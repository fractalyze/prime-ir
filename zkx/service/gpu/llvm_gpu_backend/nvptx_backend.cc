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

#include "zkx/service/gpu/llvm_gpu_backend/nvptx_backend.h"

#include <algorithm>
#include <variant>

#include "absl/base/call_once.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "third_party/gpus/cuda/include/cuda.h"

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/profiler/lib/scoped_annotation.h"
#include "xla/tsl/profiler/lib/traceme.h"
#include "zkx/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "zkx/service/gpu/llvm_gpu_backend/nvptx_libdevice_path.h"
#include "zkx/service/llvm_ir/llvm_command_line_options.h"
#include "zkx/stream_executor/cuda/ptx_compiler_helpers.h"
#include "zkx/stream_executor/cuda/subprocess_compilation.h"

namespace zkx::gpu::nvptx {
namespace {

// Default inline threshold value to use in llvm.
const int kDefaultInlineThreshold = 1100;

// Emits the given module to PTX. target_machine is an initialized TargetMachine
// for the NVPTX target.
std::string EmitModuleToPTX(llvm::Module* module,
                            llvm::TargetMachine* target_machine) {
  tsl::profiler::ScopedAnnotation annotation([&] {
    return absl::StrFormat("ZkxEmitGpuAsm:#module=%s#",
                           module->getName().str());
  });
  std::string ptx;
  llvm::raw_string_ostream stream(ptx);
  llvm::buffer_ostream pstream(stream);
  llvm::legacy::PassManager pm;
  pm.add(new llvm::TargetLibraryInfoWrapperPass(
      llvm::Triple(module->getTargetTriple())));
  target_machine->addPassesToEmitFile(pm, pstream, nullptr,
                                      llvm::CodeGenFileType::AssemblyFile);
  pm.run(*module);
  return ptx;
}

// Links libdevice into the given module if the module needs libdevice.
absl::Status LinkLibdeviceIfNecessary(llvm::Module* module,
                                      std::string_view libdevice_path) {
  if (!CouldNeedDeviceBitcode(*module)) {
    return absl::OkStatus();
  }

  if (!tsl::Env::Default()->FileExists(libdevice_path).ok()) {
    LOG(WARNING)
        << "libdevice is required by this HLO module but was not found at "
        << libdevice_path;
    return absl::InternalError(
        absl::StrFormat("libdevice not found at %s", libdevice_path));
  }

  VLOG(1) << "Linking with libdevice from: " << libdevice_path;
  return LinkWithBitcodeVector(module, {libdevice_path});
}

absl::Status NVPTXTargetModuleLinker(llvm::Module* module,
                                     se::GpuComputeCapability gpu_version,
                                     const DebugOptions& debug_options,
                                     std::string_view device_bitcode_path) {
  // Link the input module with libdevice, to pull in implementations of some
  // builtins.
  return LinkLibdeviceIfNecessary(module, device_bitcode_path);
}

std::unique_ptr<llvm::TargetMachine> NVPTXGetTargetMachine(
    llvm::Triple target_triple, se::CudaComputeCapability compute_capability,
    const DebugOptions& debug_options) {
  absl::StatusOr<se::SemanticVersion> runtime_cuda_version =
      se::GetAsmCompilerVersion(debug_options.zkx_gpu_cuda_data_dir());

  constexpr se::SemanticVersion kCompileTimeCudaVersion{
      CUDA_VERSION / 1000, (CUDA_VERSION / 10) % 100, CUDA_VERSION % 10};

  se::SemanticVersion highest_supported_cuda_version = [&] {
    if (runtime_cuda_version.ok()) {
      return std::min(runtime_cuda_version.value(), kCompileTimeCudaVersion);
    }

    return kCompileTimeCudaVersion;
  }();

  se::SemanticVersion ptx_version =
      DetermineHighestSupportedPtxVersionFromCudaVersion(
          highest_supported_cuda_version);
  int highest_supported_ptx_version =
      ptx_version.major() * 10 + ptx_version.minor();

  VLOG(1) << "Targeting PTX version: " << highest_supported_ptx_version;
  std::string feature_str =
      absl::StrFormat("+ptx%d", highest_supported_ptx_version);

  return GetTargetMachine(target_triple, GetSmName(compute_capability),
                          debug_options, feature_str);
}

// One-time module initializer.
// Must be called only once -- DO NOT CALL DIRECTLY.
void NVPTXBackendInit() {
  // Initialize the NVPTX target; it's the only target we link with, so call its
  // specific initialization functions instead of the catch-all InitializeAll*.
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();

  // Initialize the LLVM optimization passes.
  llvm::PassRegistry* registry = llvm::PassRegistry::getPassRegistry();
  InitializePasses(registry);
}

}  // namespace

std::vector<std::string> GetNVPTXBackendOptions(
    const DebugOptions& debug_options) {
  // Feed all customized flags here, so we can override them with llvm_cl_opts
  // without redeploy the compiler for development purpose.
  std::vector<std::string> backend_llvm_opts;

  // This flag tunes a threshold in branch folding. The default threshold, which
  // is one, is not suitable for CUDA programs where branches are more expensive
  // than for CPU programs. Setting the threshold to 2 improves the latency of
  // TwoDPatchDotProductKernel_IND_3_ND_48 by over 5%, and does not affect the
  // latency of other benchmarks so far.
  //
  // I also tried setting this threshold to other values:
  // * 3-6 gives similar results as 2;
  // * >6 start hurting the performance of at least dot product kernels.
  //
  // TODO(jingyue): The current threshold only considers the number of IR
  // instructions which do not accurately reflect the true cost. We need a
  // better cost model.
  backend_llvm_opts.emplace_back("-bonus-inst-threshold=2");

  // SLPVectorizer is useful (vectorizes f16x2 ops) but slow. Most of the
  // slowness appears to be in trying to form horizontal reductions, which don't
  // exist in PTX *anyway*. Disable these. While we're here, tweak
  // SLPVectorizer so it doesn't try to create large vectors -- f16x2 are the
  // only vectors supported in PTX.
  backend_llvm_opts.emplace_back("-slp-vectorize-hor=false");
  backend_llvm_opts.emplace_back("-slp-max-reg-size=32");

  // Extra backend options must go after regular backend options in order to be
  // able for the later to override the former.
  std::vector<std::string> backend_extra_llvm_opts =
      llvm_ir::ExtractZkxBackendExtraOptions(
          debug_options.zkx_backend_extra_options());
  backend_llvm_opts.insert(backend_llvm_opts.end(),
                           backend_extra_llvm_opts.cbegin(),
                           backend_extra_llvm_opts.cend());

  return backend_llvm_opts;
}

std::string GetSmName(se::CudaComputeCapability compute_capability) {
  int compute_capability_version =
      compute_capability.major * 10 + compute_capability.minor;
  int sm_version = 30;
  // If the current compute capability isn't known, fallback to the
  // most recent version before it.
  static constexpr int kSupportedVersions[] = {120, 101, 100, 90, 89, 87, 86,
                                               80,  75,  72,  70, 62, 61, 60,
                                               53,  52,  50,  37, 35, 32, 30};
  for (int v : kSupportedVersions) {
    if (v <= compute_capability_version) {
      sm_version = v;
      break;
    }
  }

  // If the current CC isn't supported by LLVM and it is newer then
  // the max supported LLVM version, do not warn about it. The end
  // user can't do anything about this. E.g., PTX compiled for SM75 will
  // run on SM80 too.
  if (sm_version != compute_capability_version &&
      compute_capability_version < kSupportedVersions[0]) {
    LOG(WARNING) << "Unknown compute capability "
                 << compute_capability.ToString()
                 << ". Defaulting to telling LLVM that we're compiling for sm_"
                 << sm_version;
  }
  // On Hopper, default to sm_90a so that all instructions can be used. But
  // only sm_90 is forward compatible, so don't use sm_90a with newer hardware:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ptx-compatibility
  // Similarly for sm_100a, sm_101a and sm_120a (Blackwell).
  std::string_view extension =
      se::ShouldUsePtxExtension(compute_capability) ? "a" : "";
  return absl::StrCat("sm_", sm_version, extension);
}

absl::StatusOr<std::string> CompileToPtx(
    llvm::Module* module, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options,
    std::function<void(llvm::TargetMachine*)> configure_target) {
  static absl::once_flag backend_init_flag;
  absl::call_once(backend_init_flag, NVPTXBackendInit);
  std::vector<std::string> llvm_opts = GetNVPTXBackendOptions(debug_options);
  llvm_ir::LLVMCommandLineOptionsLock llvm_lock(llvm_opts);

  std::string ptx;
  {
    tsl::profiler::TraceMe activity(
        [&] { return absl::StrCat("Compiling IR:", module->getName().str()); },
        tsl::profiler::TraceMeLevel::kInfo);
    // TODO(chokobole): Uncomment this. Dependency: XLA_SCOPED_LOGGING_TIMER
    // XLA_SCOPED_LOGGING_TIMER("Compile module " + module->getName().str());

    // If the module has no functions or globals, there's nothing to compile.
    // Just return an empty string.
    if (module->empty() && module->global_empty()) {
      VLOG(2) << "Module '" << module->getName().str()
              << "' is empty. Skipping compilation.";
      return std::string();
    }

    auto compute_capability =
        std::get_if<se::CudaComputeCapability>(&gpu_version);
    if (!compute_capability) {
      return absl::InternalError(
          "Incompatible compute capability was specified.");
    }

    llvm::Triple default_target_triple("nvptx64-unknown-unknown");
    // Construct LLVM TargetMachine for NVPTX.
    std::unique_ptr<llvm::TargetMachine> target_machine = NVPTXGetTargetMachine(
        default_target_triple, *compute_capability, debug_options);

    // Apply target machine configuration from call-back if available.
    if (configure_target) {
      configure_target(target_machine.get());
    }

    // Link with libdevice, and optimize the LLVM module.
    TF_RETURN_IF_ERROR(LinkAndOptimizeModule(
        module, gpu_version, debug_options,
        LibDevicePath(debug_options.zkx_gpu_cuda_data_dir()),
        NVPTXTargetModuleLinker, default_target_triple, target_machine.get(),
        kDefaultInlineThreshold));

    // TODO(chokobole): Uncomment this. Dependency: RecordLlvmPassesDuration
    // uint64_t end_usecs = tsl::Env::Default()->NowMicros();
    // RecordLlvmPassesDuration(end_usecs - start_usecs);

    // Lower optimized LLVM module to PTX.
    ptx = EmitModuleToPTX(module, target_machine.get());

    // TODO(chokobole): Uncomment this. Dependency: RecordLlvmToPtxDuration
    // end_usecs = tsl::Env::Default()->NowMicros();
    // RecordLlvmToPtxDuration(end_usecs - start_usecs);
  }
  return ptx;
}

namespace {

constexpr se::SemanticVersion kFallbackPtxVersion{6, 5, 0};
constexpr se::SemanticVersion kMaxPtxVersion{8, 7, 0};

}  // namespace

se::SemanticVersion DetermineHighestSupportedPtxVersionFromCudaVersion(
    se::SemanticVersion cuda_version) {
  if (cuda_version < se::SemanticVersion{11, 0, 0}) {
    // For everything below CUDA 11 we just fall back to PTX 6.5.
    // We don't support CUDA below 11 anymore.
    return kFallbackPtxVersion;
  }

  // Mapping determined from
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-notes
  // Examples:
  // CUDA 11.0 -> PTX 7.0
  // CUDA 11.1 -> PTX 7.1
  // CUDA 12.0 -> PTX 8.0
  // CUDA 12.4 -> PTX 8.4
  // This versioning scheme is valid until CUDA 12.6
  if (cuda_version < se::SemanticVersion{12, 6, 0}) {
    return {cuda_version.major() - 4, cuda_version.minor(), 0};
  }
  // CUDA 12.6 -> PTX 8.5
  // CUDA 12.8 -> PTX 8.7
  if (cuda_version < se::SemanticVersion{12, 9, 0}) {
    return {cuda_version.major() - 4, cuda_version.minor() - 1, 0};
  }

  // Return maximum known PTX version.
  return kMaxPtxVersion;
}

}  // namespace zkx::gpu::nvptx
