/* Copyright 2017 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "zkx/service/gpu/nvptx_compiler.h"

#include <fstream>
#include <memory>
#include <tuple>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

#include "xla/tsl/platform/path.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/lib/scoped_annotation.h"
#include "xla/tsl/profiler/lib/traceme.h"
#include "zkx/service/dump.h"
#include "zkx/service/gpu/buffer_sharing.h"
#include "zkx/service/gpu/llvm_gpu_backend/nvptx_backend.h"
#include "zkx/service/gpu/ptx_compile_options_from_debug_options.h"
#include "zkx/service/gpu/target_constants.h"
#include "zkx/shape_util.h"
#include "zkx/stream_executor/cuda/assemble_compilation_provider.h"
#include "zkx/stream_executor/cuda/cuda_diagnostics.h"
#include "zkx/stream_executor/cuda/cuda_platform_id.h"
#include "zkx/stream_executor/cuda/subprocess_compilation.h"

namespace zkx::gpu {
namespace {

// Try to load ptx from files defined in the FLAGS. If successful, return true.
bool MaybeLoadPtxFromFile(const HloModuleConfig module_config,
                          const HloModule* module, std::string* ptx) {
  // If the zkx_gpu_ptx_file option is set, be explicit if a file is used
  // and warn when a file is not used to ease catching typo in filename.
  std::string prefix = FilenameFor(*module, "", *ptx);
  std::string matched_filename;
  for (const std::string& full_filename :
       module_config.debug_options().zkx_gpu_ptx_file()) {
    // To ease comparing many PTX versions, accept different suffixes then
    // the original filename.
    auto filename = tsl::io::Basename(full_filename);
    if (absl::StartsWith(filename, prefix)) {
      matched_filename = full_filename;
      VLOG(1) << "RunBackend() - Will load PTX from file: " << full_filename;
      break;
    }
  }
  if (!module_config.debug_options().zkx_gpu_ptx_file().empty() &&
      matched_filename.empty()) {
    VLOG(1) << "RunBackend() - For module with prefix '" << prefix
            << "', we did not found a PTX file to load.";
  }

  if (!matched_filename.empty()) {
    std::ifstream ifs(matched_filename, std::ifstream::in);
    *ptx = std::string(std::istreambuf_iterator<char>(ifs),
                       std::istreambuf_iterator<char>());
    CHECK(!ptx->empty()) << "Empty or non existing PTX file: "
                         << matched_filename;
    return true;
  }
  return false;
}

// Try to load textual LLVM IR from files defined in the FLAGS. If
// successful, return the llvm::Module, otherwise return nullptr.
std::unique_ptr<llvm::Module> MaybeLoadLLVMFromFile(const HloModule* module,
                                                    llvm::Module* llvm_module) {
  // If the zkx_gpu_llvm_ir_file option is set, be explicit if a file is used
  // and warn when a file is not used to ease catching typo in filename.
  if (module == nullptr) {
    return nullptr;
  }

  std::string prefix = FilenameFor(*module, "", "");
  auto zkx_gpu_llvm_ir_file =
      module->config().debug_options().zkx_gpu_llvm_ir_file();
  auto matched_filename = absl::c_find_if(
      zkx_gpu_llvm_ir_file, [prefix](const std::string& full_filename) {
        // To ease comparing many LLVM versions, accept different suffixes then
        // the original filename.
        return absl::StartsWith(tsl::io::Basename(full_filename), prefix);
      });
  if (!zkx_gpu_llvm_ir_file.empty() &&
      matched_filename == std::end(zkx_gpu_llvm_ir_file)) {
    VLOG(1) << "RunBackend() - For module with prefix '" << prefix
            << "', we did not found a LLVM file to load.";
  }

  if (matched_filename != std::end(zkx_gpu_llvm_ir_file)) {
    VLOG(1) << "RunBackend() - Will load LLVM from file: " << *matched_filename;
    llvm::LLVMContext& context = llvm_module->getContext();
    llvm::SMDiagnostic err;
    std::unique_ptr<llvm::Module> loaded_module =
        llvm::parseIRFile(*matched_filename, err, context);

    if (!loaded_module) {
      err.print("ERR", llvm::errs());
      LOG(FATAL) << "Failed to load an LLVM file. It is probably invalid LLVM.";
    }
    // Overwrite the dumped not optimized LLVM to show which one will be used.
    DumpIrIfEnabled(*module, *loaded_module, /*optimized=*/false);
    return loaded_module;
  }
  return nullptr;
}

}  // namespace

// Prints a warning if the ptx->sass JIT in the driver has known bugs.
//
// Using such a driver only a problem if we fail to use ptxas to compile our ptx
// and have to use the driver instead, so you should only call this function if
// we're going to use the driver JIT.
//
// Only prints a warning the first time it's called.
void WarnIfBadDriverJITVersion() {
  static absl::once_flag run_once;
  absl::call_once(run_once, [] {
    absl::StatusOr<se::cuda::DriverVersion> version_or_status =
        se::cuda::Diagnostician::FindKernelDriverVersion();
    if (!version_or_status.ok()) {
      LOG(WARNING) << "Couldn't read CUDA driver version.";
      return;
    }
    se::cuda::DriverVersion version = version_or_status.value();

    // The following versions of the driver JIT miscompile some address
    // calculations with large offsets (e.g. "load ptr + large_constant"),
    // b/70245379:
    //
    //  - 384.x before 384.108
    //  - 387.x before 387.40
    //  - 390.x before 390.10.
    //
    // In addition, only >= 396.20 contains ptxas >= 9.2.88, which contains the
    // fix for the "large multioutput fusions" miscompile, b/111107644.
    if (version < std::make_tuple(396, 20, 0)) {
      LOG(WARNING)
          << "*** WARNING *** Invoking the PTX->SASS JIT from driver version "
          << se::cuda::DriverVersionToString(version)
          << ", which is older than 396.20.0. These versions are known to "
             "miscompile ZKX code, leading to incorrect results or "
             "invalid-address errors.\nZKX only uses the driver JIT if it "
             "cannot find ptxas; you don't need to update your driver if "
             "you can point ZKX to ptxas 9.2.88 or newer.";
    }
  });
}

absl::StatusOr<const se::cuda::CompilationProvider*>
NVPTXCompiler::GetCompilationProvider(const DebugOptions& debug_options) {
  absl::MutexLock lock(&compilation_providers_mutex_);
  std::unique_ptr<se::cuda::CompilationProvider>& compilation_provider =
      compilation_providers_[se::cuda::CompilationProviderOptions::
                                 FromDebugOptions(debug_options)];
  if (compilation_provider == nullptr) {
    TF_ASSIGN_OR_RETURN(
        compilation_provider,
        se::cuda::AssembleCompilationProvider(
            se::cuda::CompilationProviderOptions::FromDebugOptions(
                debug_options)));
  }
  return compilation_provider.get();
}

NVPTXCompiler::NVPTXCompiler()
    : GpuCompiler(se::cuda::kCudaPlatformId, nvptx::TargetTriple(),
                  nvptx::DataLayout()) {}

HloDataflowAnalysis::CanShareBuffer NVPTXCompiler::GetCanShareBuffer(
    const se::DeviceDescription& device_description) const {
  return [&](const HloInstruction* user, const HloInstruction* operand,
             const ShapeIndex& user_index) {
    return CanShareBufferHint(user, operand, user_index, device_description);
  };
}

absl::StatusOr<GpuCompiler::BackendCompileResult>
NVPTXCompiler::CompileTargetBinary(
    const HloModuleConfig& module_config, llvm::Module* llvm_module,
    const se::DeviceDescription& device_description, bool relocatable,
    const HloModule* debug_module, const CompileOptions& options,
    std::optional<int> shard_number) {
  std::unique_ptr<llvm::Module> loaded_module =
      MaybeLoadLLVMFromFile(debug_module, llvm_module);
  llvm::Module* selected_module = nullptr;
  if (loaded_module) {
    selected_module = loaded_module.get();
  } else {
    selected_module = llvm_module;
  }

  std::string ptx;
  if (!(debug_module &&
        MaybeLoadPtxFromFile(module_config, debug_module, &ptx))) {
    // This may print multiple lines per HLO compilation because of the
    // parallelized compilation of LLVM modules.
    // TODO(chokobole): Uncomment this. Dependency: XLA_SCOPED_LOGGING_TIMER_IF
    // XLA_SCOPED_LOGGING_TIMER_IF(
    //     absl::StrCat(
    //         "NVPTXCompiler::CompileTargetBinary - CompileToPtx for ",
    //         (debug_module != nullptr ? debug_module->name() : "(unknown")),
    //     !options.is_autotuning_compilation);
    TF_ASSIGN_OR_RETURN(
        ptx, nvptx::CompileToPtx(selected_module,
                                 device_description.gpu_compute_capability(),
                                 module_config.debug_options()));

    // This won't record values for calls that error out (because if they error
    // out we have no way of telling how far through the process we got).
    // clang-format off
    // TODO(chokobole): Uncomment this. Dependency: RecordLlvmPassesAndLlvmToPtxDuration
    // clang-format on
    // RecordLlvmPassesAndLlvmToPtxDuration(end_usecs - start_usecs);

    if (DumpingEnabledForHloModule(debug_module ? debug_module->name() : "",
                                   module_config.debug_options())) {
      if (debug_module) {
        DumpToFileInDirOrStdout(*debug_module, "",
                                shard_number.has_value()
                                    ? (std::to_string(*shard_number) + ".ptx")
                                    : "ptx",
                                ptx);
      } else {
        LOG(ERROR)
            << "Dumping is not implemented since the file name cannot be "
               "inferred. Please implement (potentially MLIR) module -> "
               "filename heuristic.";
      }
    }
  }

  if (ptx.empty()) {
    return BackendCompileResult{};
  }

  TF_ASSIGN_OR_RETURN(const se::cuda::CompilationProvider* compilation_provider,
                      GetCompilationProvider(module_config.debug_options()));

  se::cuda::CompilationOptions compilation_options =
      PtxCompileOptionsFromDebugOptions(
          module_config.debug_options(),
          /*is_autotuning_compilation=*/options.is_autotuning_compilation);

  se::CudaComputeCapability cc = std::get<se::CudaComputeCapability>(
      device_description.gpu_compute_capability());

  // This may print multiple lines per HLO compilation because of the
  // parallelized compilation of LLVM modules.
  std::string module_name =
      debug_module != nullptr ? debug_module->name() : "(unknown)";
  // TODO(chokobole): Uncomment this. Dependency: XLA_SCOPED_LOGGING_TIMER_IF
  // XLA_SCOPED_LOGGING_TIMER_IF(
  //     absl::StrCat("NVPTXCompiler::CompileTargetBinary - PtxToCubin for ",
  //                  module_name),
  //     !options.is_autotuning_compilation);
  tsl::profiler::ScopedAnnotation annotation([&] {
    return absl::StrFormat("ZkxCompileGpuAsm:#module=%s#", module_name);
  });
  tsl::profiler::TraceMe activity("PTX->CUBIN",
                                  tsl::profiler::TraceMeLevel::kInfo);

  const auto record_ptx_to_cubin_metric = [&]() {
    // This won't record values for calls that error out (because if they
    // error out we have no way of telling how far through the process we
    // got).
    // TODO(chokobole): Uncomment this. Dependency: RecordPtxToCubinDuration
    // RecordPtxToCubinDuration(end_usecs - start_usecs);
  };

  if (relocatable) {
    TF_ASSIGN_OR_RETURN(se::cuda::RelocatableModule relocatable_module,
                        compilation_provider->CompileToRelocatableModule(
                            cc, ptx, compilation_options));
    record_ptx_to_cubin_metric();
    return BackendCompileResult{std::move(ptx),
                                std::move(relocatable_module.cubin)};
  }

  TF_ASSIGN_OR_RETURN(
      se::cuda::Assembly assembly,
      compilation_provider->Compile(cc, ptx, compilation_options));
  record_ptx_to_cubin_metric();
  return BackendCompileResult{std::move(ptx), std::move(assembly.cubin)};
}

absl::StatusOr<bool> NVPTXCompiler::CanUseLinkModules(
    const HloModuleConfig& hlo_module_config,
    const se::DeviceDescription& device_description) {
  TF_ASSIGN_OR_RETURN(
      const se::cuda::CompilationProvider* compilation_provider,
      GetCompilationProvider(hlo_module_config.debug_options()));
  return compilation_provider->SupportsCompileAndLink() &&
         compilation_provider->SupportsCompileToRelocatableModule();
}

absl::StatusOr<std::vector<uint8_t>> NVPTXCompiler::LinkModules(
    const se::DeviceDescription& device_description,
    se::StreamExecutor* stream_exec, std::vector<std::vector<uint8_t>> modules,
    const DebugOptions& debug_options) {
  if (modules.empty()) return std::vector<uint8_t>{};

  auto cc = std::get<se::CudaComputeCapability>(
      device_description.gpu_compute_capability());

  TF_ASSIGN_OR_RETURN(const se::cuda::CompilationProvider* compilation_provider,
                      GetCompilationProvider(debug_options));

  std::vector<se::cuda::CompilationProvider::RelocatableModuleOrPtx> inputs;
  inputs.reserve(modules.size());
  for (std::vector<uint8_t>& module : modules) {
    inputs.push_back(se::cuda::RelocatableModule{std::move(module)});
  }

  se::cuda::CompilationOptions compilation_options =
      PtxCompileOptionsFromDebugOptions(debug_options,
                                        /*is_autotuning_compilation=*/false);

  VLOG(1) << "Linking " << modules.size()
          << " modules with compilation provider "
          << compilation_provider->name();
  TF_ASSIGN_OR_RETURN(
      se::cuda::Assembly assembly,
      compilation_provider->CompileAndLink(cc, inputs, compilation_options));

  return std::move(assembly.cubin);
}

std::vector<std::string> NVPTXCompiler::GetLLVMCommandLineOptions(
    const DebugOptions& debug_options) const {
  return nvptx::GetNVPTXBackendOptions(debug_options);
}

}  // namespace zkx::gpu
