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

#include "zkx/service/gpu/gpu_compiler.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <variant>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "mlir/IR/DialectRegistry.h"

#include "xla/tsl/platform/casts.h"
#include "xla/tsl/platform/cpu_info.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/path.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/thread_pool.h"
#include "xla/tsl/profiler/lib/scoped_annotation.h"
#include "xla/tsl/profiler/lib/traceme.h"
#include "zkx/base/logging.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/ir/hlo_module_group.h"
#include "zkx/maybe_owning.h"
#include "zkx/service/buffer_assignment.h"
#include "zkx/service/dump.h"
#include "zkx/service/gpu/compile_module_to_llvm_ir.h"
#include "zkx/service/gpu/execution_stream_assignment.h"
#include "zkx/service/gpu/gpu_executable.h"
#include "zkx/service/gpu/gpu_hlo_schedule.h"
#include "zkx/service/gpu/gpu_latency_hiding_scheduler.h"
#include "zkx/service/gpu/ir_emitter_context.h"
#include "zkx/service/gpu/ir_emitter_unnested.h"
#include "zkx/service/gpu/kernel_reuse_cache.h"
#include "zkx/service/llvm_ir/llvm_command_line_options.h"
#include "zkx/service/llvm_ir/llvm_util.h"
#include "zkx/service/slow_operation_alarm.h"
#include "zkx/status_macros.h"
#include "zkx/stream_executor/platform_manager.h"
#include "zkx/zkx_data.pb.h"

namespace zkx::gpu {
namespace {

using MaybeOwningThreadPool = MaybeOwning<tsl::thread::ThreadPool>;

MaybeOwningThreadPool CreateMaybeOwningThreadPool(
    int parallelism, tsl::thread::ThreadPool* default_thread_pool,
    int default_parallelism) {
  CHECK_GE(parallelism, 0);
  CHECK_GE(default_parallelism, 1);
  // CurrentThreadId() returns -1 if the current thread does not belong to the
  // thread pool. If the current thread belongs to the thread pool, we should
  // not be using it, because it can potentially cause deadlocks.
  CHECK(default_thread_pool == nullptr ||
        default_thread_pool->CurrentThreadId() == -1);

  auto create_thread_pool = [&](int num_threads) {
    CHECK_GE(num_threads, 1);
    return std::make_unique<tsl::thread::ThreadPool>(tsl::Env::Default(), "",
                                                     num_threads);
  };

  switch (parallelism) {
    case 0:
      if (default_thread_pool == nullptr && default_parallelism > 1) {
        return MaybeOwningThreadPool(create_thread_pool(default_parallelism));
      }
      return MaybeOwningThreadPool(default_thread_pool);
    case 1:
      return MaybeOwningThreadPool(nullptr);
    default:
      return MaybeOwningThreadPool(create_thread_pool(parallelism));
  }
}

se::GpuComputeCapability GetGpuVersion(const se::StreamExecutor* stream_exec) {
  return stream_exec->GetDeviceDescription().gpu_compute_capability();
}

class GpuThunkAotCompilationResult : public AotCompilationResult {
 public:
  static absl::StatusOr<std::unique_ptr<GpuThunkAotCompilationResult>>
  FromModule(const HloModule* hlo_module,
             const BufferAssignment* buffer_assignment,
             std::string_view asm_text, absl::Span<const uint8_t> binary) {
    CompilationResultProto proto;
    *proto.mutable_hlo_module_with_config() = hlo_module->ToProtoWithConfig();
    *proto.mutable_buffer_assignment() = buffer_assignment->ToProto();
    proto.set_asm_text(std::string(asm_text));
    proto.set_binary(binary.data(), binary.size());
    return std::unique_ptr<GpuThunkAotCompilationResult>(
        new GpuThunkAotCompilationResult(hlo_module->Clone(),
                                         std::move(proto)));
  }

  static absl::StatusOr<std::unique_ptr<GpuThunkAotCompilationResult>>
  FromString(const std::string& serialized) {
    CompilationResultProto proto;
    if (!proto.ParseFromString(serialized)) {
      return absl::InternalError(
          "Failed to parse serialized GpuThunkAotCompilationResult.");
    }

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModule> module,
        HloModule::CreateFromProtoWithConfig(proto.hlo_module_with_config()));
    return std::unique_ptr<GpuThunkAotCompilationResult>(
        new GpuThunkAotCompilationResult(std::move(module), std::move(proto)));
  }

  absl::StatusOr<std::string> SerializeAsString() const override {
    return proto_.SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Executable>> LoadExecutable(
      Compiler* compiler, const se::StreamExecutor* stream_exec) const override;

  const HloModule* optimized_module() const override { return module_.get(); }
  std::unique_ptr<HloModule> consume_optimized_module() override {
    return std::move(module_);
  }

 private:
  GpuThunkAotCompilationResult(std::unique_ptr<HloModule> module,
                               CompilationResultProto proto)
      : module_(std::move(module)), proto_(std::move(proto)) {}

  std::unique_ptr<HloModule> module_;
  CompilationResultProto proto_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<Executable>>
GpuThunkAotCompilationResult::LoadExecutable(
    Compiler* compiler, const se::StreamExecutor* stream_exec) const {
  // Recreate HloModule+HloModuleConfig from proto.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      HloModule::CreateFromProtoWithConfig(proto_.hlo_module_with_config()));

  // Recreate BufferAssignment from proto.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BufferAssignment> buffer_assignment,
      BufferAssignment::FromProto(proto_.buffer_assignment(), hlo_module.get(),
                                  compiler->BufferSizeBytesFunction(),
                                  /*can_share_buffer=*/nullptr));

  ExecutionStreamAssignment execution_stream_assignment(hlo_module.get());

  std::vector<uint8_t> binary(proto_.binary().begin(), proto_.binary().end());

  // Build the executable, which should be a thunk sequence.
  TF_ASSIGN_OR_RETURN(
      se::Platform * platform,
      se::PlatformManager::PlatformWithId(compiler->PlatformId()));
  std::string_view platform_name = platform->Name();
  const se::DeviceDescription& gpu_device_info =
      stream_exec->GetDeviceDescription();
  mlir::DialectRegistry registry;
  auto mlir_context = std::make_unique<mlir::MLIRContext>(registry);
  llvm::LLVMContext llvm_context;
  auto* gpu_compiler = dynamic_cast<GpuCompiler*>(compiler);
  if (gpu_compiler == nullptr) {
    return absl::InternalError("Compiler is not a GpuCompiler.");
  }
  auto llvm_module = std::make_unique<llvm::Module>("", llvm_context);
  llvm_module->setTargetTriple(llvm::Triple(gpu_compiler->target_triple()));
  llvm_module->setDataLayout(gpu_compiler->data_layout());
  IrEmitterContext ir_emitter_context(
      hlo_module.get(), buffer_assignment.get(), &execution_stream_assignment,
      platform_name, gpu_device_info, mlir_context.get(), llvm_module.get(),
      /*llvm_module_constants=*/nullptr,
      /*emit_kernels=*/false);

  std::string_view cache_file_path =
      hlo_module->config().debug_options().zkx_gpu_kernel_cache_file();
  if (!cache_file_path.empty() &&
      hlo_module->config()
          .debug_options()
          .zkx_gpu_enable_llvm_module_compilation_parallelism()) {
    TF_RETURN_IF_ERROR(LoadCache(ir_emitter_context, cache_file_path));
  }

  auto ir_emitter = IrEmitterUnnested::Create(&ir_emitter_context);
  TF_RETURN_IF_ERROR(
      ir_emitter->EmitHloComputation(hlo_module->entry_computation()));

  // Get all other fields required by GpuExecutable.
  std::vector<GpuExecutable::ConstantInfo> constants =
      std::move(ir_emitter_context.constants());
  TF_ASSIGN_OR_RETURN(auto output_info,
                      GetOutputInfo(*hlo_module, *buffer_assignment));
  const Shape& output_shape = hlo_module->result_shape();
  int64_t debug_buffer_assignment_show_max =
      hlo_module->config()
          .debug_options()
          .zkx_debug_buffer_assignment_show_max();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<GpuExecutable> executable,
      GpuExecutable::Create(GpuExecutable::Params{
          /*asm_text=*/proto_.asm_text(),
          /*binary=*/binary,
          /*gpu_version=*/gpu_device_info.gpu_compute_capability(),
          /*executable=*/ir_emitter->ConsumeThunkSequence(),
          /*constants=*/std::move(constants),
          /*output_info=*/std::move(output_info),
          /*module_name=*/std::move(hlo_module->name()),
          /*output_shape=*/std::move(output_shape),
          /*mlir_allocations=*/std::nullopt,
          /*buffer_assignment=*/std::move(buffer_assignment),
          /*debug_buffer_assignment_show_max=*/debug_buffer_assignment_show_max,
          /*debug_module=*/std::move(hlo_module),
          /*enable_debug_info_manager=*/true}));
  return executable;
}

GpuCompiler::GpuCompiler(se::Platform::Id platform_id,
                         const char* target_triple, const char* data_layout)
    : platform_id_(platform_id),
      target_triple_(target_triple),
      data_layout_(data_layout),
      pointer_size_(llvm::DataLayout(data_layout)
                        .getPointerSize(0 /* default address space */)) {}

namespace {

void CheckNotScheduled(HloModule* hlo_module) {
  if (hlo_module->has_schedule() &&
      !hlo_module->config().debug_options().zkx_disable_all_hlo_passes()) {
    LOG(WARNING) << "\nThe current HLO module " << hlo_module->name()
                 << " is scheduled and optimized. \n"
                 << "It is not expected to run optimization passes again.\n"
                    "Use a test method like RunAndCompareNoHloPasses() or "
                 << "the zkx_disable_all_hlo_passes flag.";
  }
}

void LogDebugOptions(HloModule* hlo_module) {
  // LOG_LINES is used instead of LOG since the message can exceed the
  // maximum line length, which results in the message being truncated.
  ZKX_VLOG_LINES(
      1, absl::StrFormat("GpuCompilationEnvironment of hlo_module %s:\n%s",
                         hlo_module->name(),
                         hlo_module->config().debug_options().DebugString()));
}

}  // namespace

// Runs optimization passes on the given HLO module.
absl::Status GpuCompiler::OptimizeHloModule(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    const CompileOptions& options, const TargetConfig& gpu_target_config) {
  tsl::profiler::TraceMe traceme("GpuCompiler::OptimizeHloModule");

  CheckNotScheduled(hlo_module);
  LogDebugOptions(hlo_module);

  MaybeOwningThreadPool thread_pool = CreateMaybeOwningThreadPool(
      /*parallelism=*/hlo_module->config()
          .debug_options()
          .zkx_gpu_force_compilation_parallelism(),
      /*default_thread_pool=*/options.thread_pool,
      /*default_parallelism=*/tsl::port::MaxParallelism());

  // TODO(chokobole): Uncomment this. Dependency: RunPreSPMDPartitionerPasses
  // TF_RETURN_IF_ERROR(RunPreSPMDPartitionerPasses(hlo_module));
  // TODO(chokobole): Uncomment this. Dependency: RunSPMDPasses
  // TF_RETURN_IF_ERROR(RunSPMDPasses(hlo_module, gpu_target_config,
  //                                  layout_insensitive_algsimp_opts));
  // TODO(chokobole): Uncomment this. Dependency: RunOptimizationPasses
  // TF_RETURN_IF_ERROR(RunOptimizationPasses(hlo_module, gpu_target_config,
  //                                          layout_insensitive_algsimp_opts));
  se::GpuComputeCapability gpu_version =
      gpu_target_config.device_description.gpu_compute_capability();
  // clang-format off
      // TODO(chokobole): Uncomment this. Dependency: RunCollectiveOptimizationPasses
  // clang-format on
  // TF_RETURN_IF_ERROR(RunCollectiveOptimizationPasses(
  //     hlo_module, layout_insensitive_algsimp_opts, gpu_version));

  // Run target-specific HLO optimization passes for convolution
  // canonicalization.
  if (stream_exec != nullptr) {
    gpu_version = GetGpuVersion(stream_exec);
  }

  // TODO(chokobole): Uncomment this. Dependency: RunLayoutAssignmentPasses
  // TF_RETURN_IF_ERROR(
  //     RunLayoutAssignmentPasses(hlo_module, gpu_version, dnn_version,
  //                               gpu_target_config.device_description));
  // TODO(chokobole): Uncomment this. Dependency: RunLayoutNormalizationPasses
  // TF_RETURN_IF_ERROR(RunLayoutNormalizationPasses(hlo_module, gpu_version));

  // Run target-specific HLO optimization passes after layout assignment.
  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: OptimizeHloPostLayoutAssignment
  // clang-format on
  // TF_RETURN_IF_ERROR(OptimizeHloPostLayoutAssignment(
  //     hlo_module, stream_exec, options, gpu_target_config,
  //     thread_pool.get_mutable()));

  // This is a "low effort, high impact" fusion that should be run first.
  // TODO(chokobole): Uncomment this. Dependency: RunDynamicSliceFusionPasses
  // TF_RETURN_IF_ERROR(RunDynamicSliceFusionPasses(hlo_module, PlatformId()));

  // TODO(chokobole): Uncomment this. Dependency: RunFusionPasses
  // TF_RETURN_IF_ERROR(RunFusionPasses(hlo_module, gpu_target_config,
  //                                    thread_pool.get_mutable(),
  //                                    ShapeSizeBytesFunction()));
  // TODO(chokobole): Uncomment this. Dependency: RunPostFusionPasses
  // TF_RETURN_IF_ERROR(RunPostFusionPasses(
  //     hlo_module, gpu_target_config.device_description, pointer_size_));
  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: RunAsyncCollectivesConversionPasses
  // clang-format on
  // TF_RETURN_IF_ERROR(RunAsyncCollectivesConversionPasses(hlo_module));
  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: RunPostFusionSimplificationPasses
  // clang-format on
  // TF_RETURN_IF_ERROR(RunPostFusionSimplificationPasses(
  //     hlo_module, layout_insensitive_algsimp_opts, gpu_version,
  //     gpu_target_config));

  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: RunPostFusionVerificationPasses
  // clang-format on
  // TF_RETURN_IF_ERROR(RunPostFusionVerificationPasses(
  //     hlo_module, stream_exec, options, gpu_target_config));
  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: RunCollectiveScheduleLinearizerPasses
  // clang-format on
  // TF_RETURN_IF_ERROR(
  //     RunCollectiveScheduleLinearizerPasses(hlo_module, stream_exec));
  // TODO(chokobole): Uncomment this. Dependency: RunAsyncDotPasses
  // TF_RETURN_IF_ERROR(RunAsyncDotPasses(hlo_module));

  return absl::OkStatus();
}  // NOLINT(readability/fn_size)

// Modifies the given HLO module so that it will be accepted by IrEmitter.
// Unlike optimization passes, the passes are necessary for correctness.
absl::Status GpuCompiler::PrepareHloModuleForIrEmitting(
    HloModule* hlo_module, const se::DeviceDescription& device_description) {
  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: PrepareHloModuleForIrEmittingPipeline
  // clang-format on
  // return PrepareHloModuleForIrEmittingPipeline(
  //            *hlo_module, GetCanShareBuffer(device_description),
  //            device_description)
  //     .Run(hlo_module)
  //     .status();
  return absl::OkStatus();
}

// Returns the TargetConfig, either from the module debug options, or from the
// CompilationOptions, or if both of those are absent, from the attached GPU.
// static
absl::StatusOr<Compiler::TargetConfig> GpuCompiler::GetTargetConfig(
    const Compiler::CompileOptions& options, const DebugOptions& debug_opts,
    se::StreamExecutor* executor) {
  if (options.target_config.has_value()) {
    return *options.target_config;
  }
  if (!debug_opts.zkx_gpu_target_config_filename().empty()) {
    std::string gpu_target_config_string;
    TF_RETURN_IF_ERROR(tsl::ReadFileToString(
        tsl::Env::Default(), debug_opts.zkx_gpu_target_config_filename(),
        &gpu_target_config_string));
    se::GpuTargetConfigProto gpu_target_config_proto;
    if (!google::protobuf::TextFormat::ParseFromString(
            gpu_target_config_string, &gpu_target_config_proto)) {
      return absl::FailedPreconditionError(
          "Failed to parse GpuTargetConfigProto");
    }

    return Compiler::TargetConfig{gpu_target_config_proto};
  }
  if (executor) {
    Compiler::TargetConfig target_config = Compiler::TargetConfig{executor};
    int64_t device_memory_size =
        target_config.device_description.device_memory_size();
    // Checking for device_memory_size == -1 is how we detect that we are
    // running on Nvidia's software simulator. When running on simulation,
    // the config from StreamExecutor is inaccurate, so we must load the
    // hard-coded config from a file.
    if (device_memory_size == -1) {
      return absl::FailedPreconditionError(
          "When running on an NVIDIA simulation device, you must use "
          "--zkx_gpu_target_config_filename to pass in target information. "
          "The target config from StreamExecutor is inaccurate.");
    }
    return target_config;
  }
  return absl::InternalError(
      "Either GPU has to be attached, or --zkx_gpu_target_config_filename "
      "has to be specified to specify the target to compile for.");
}

absl::StatusOr<std::unique_ptr<HloModule>> GpuCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  const DebugOptions debug_opts = module->config().debug_options();
  TF_RETURN_IF_ERROR(LoadAutotuneResultsFromFile(debug_opts));
  bool is_deviceless = options.target_config.has_value() ||
                       !debug_opts.zkx_gpu_target_config_filename().empty();

  TF_ASSIGN_OR_RETURN(TargetConfig gpu_target_config,
                      GetTargetConfig(options, debug_opts, stream_exec));
  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: MaybeUploadUnoptimizedGpuSymbols
  // clang-format on
  // const std::optional<std::string>
  // unoptimized_fingerprint =
  //     MaybeUploadUnoptimizedGpuSymbols(module.get(),
  //                                      gpu_target_config.ToProto());

  // We dump the post-optimization HLO in RunBackend so no need to dump it here.
  // TODO(chokobole): Uncomment this. Dependency: XLA_SCOPED_LOGGING_TIMER_IF
  // XLA_SCOPED_LOGGING_TIMER_IF(
  //     absl::StrCat("GpuCompiler::RunHloPasses for ", module->name()),
  //     !options.is_autotuning_compilation);
  tsl::profiler::TraceMe activity(
      [&] { return absl::StrCat("HLO Transforms:", module->name()); },
      tsl::profiler::TraceMeLevel::kInfo);

  TF_RETURN_IF_ERROR(OptimizeHloModule(module.get(),
                                       is_deviceless ? nullptr : stream_exec,
                                       options, gpu_target_config));

  TF_RETURN_IF_ERROR(PrepareHloModuleForIrEmitting(
      module.get(), gpu_target_config.device_description));

  const auto* cuda_cc = std::get_if<se::CudaComputeCapability>(
      &gpu_target_config.device_description.gpu_compute_capability());
  if (cuda_cc != nullptr && cuda_cc->IsAtLeastAmpere()) {
    // This needs to run after every pass affecting fusions, which includes
    // `CopyFusion`, which itself must run in the
    // `PrepareHloModuleForIrEmitting` pipeline.
    // TODO(chokobole): Uncomment this. Dependency: FusionDispatchPipeline
    // TF_RETURN_IF_ERROR(
    //     FusionDispatchPipeline(gpu_target_config.device_description,
    //                            ShapeSizeBytesFunction())
    //         .Run(module.get())
    //         .status());
  }

  // This won't record values for calls that error out (because if they error
  // out we have no way of telling how far through the process we got).
  // TODO(chokobole): Uncomment this. Dependency: RecordHloPassesDuration
  // RecordHloPassesDuration(end_usecs - start_usecs);

  DumpHloModuleMetadataIfEnabled({module.get()});

  // TODO(chokobole): Uncomment this. Dependency: AutotuneResults
  // AutotuneResults autotune_results;
  // TF_ASSIGN_OR_RETURN(
  //     AutotuneConfig autotune_config,
  //     GetAutotuneConfig(stream_exec, debug_opts, options,
  //     gpu_target_config));
  // if (!is_deviceless) {
  //   // TODO(chokobole): Uncomment this. Dependency: SerializeAutotuneResults
  //   TF_RETURN_IF_ERROR(
  //       AutotunerUtil::SerializeAutotuneResults(&autotune_results));
  //   TF_RETURN_IF_ERROR(SerializeAutotuneResultsToFile(debug_opts));
  // }
  // const std::optional<std::string> optimized_fingerprint =
  //     MaybeUploadOptimizedGpuSymbols(module.get(), autotune_results);
  // if (unoptimized_fingerprint.has_value() &&
  //     optimized_fingerprint.has_value()) {
  //   MaybeUploadGpuSymbolMapping(*unoptimized_fingerprint,
  //                               *optimized_fingerprint);
  // }

  if (DumpingEnabledForHloModule(*module)) {
    // clang-format off
    // TODO(chokobole): Uncomment this. Dependency: AutotunerUtil::SerializeAutotuneResults
    // clang-format on
    // TF_ASSIGN_OR_RETURN(
    //     std::string autotune_results,
    //     AutotunerUtil::SerializeAutotuneResults(/*as_textproto=*/true));
    // DumpToFileInDirOrStdout(*module, "", "autotune_results.pbtxt",
    //                         autotune_results);
  }

  return std::move(module);
}

namespace {

void NullDiagnosticHandler(const llvm::DiagnosticInfo* diag_info,
                           void* context) {
  std::string error_string;
  llvm::raw_string_ostream string_printer(error_string);
  llvm::DiagnosticPrinterRawOStream diagnostic_printer(string_printer);
  diag_info->print(diagnostic_printer);

  VLOG(5) << error_string;
}

std::unique_ptr<llvm::Module> CopyToContext(const llvm::Module& module,
                                            llvm::LLVMContext& context) {
  // We are setting llvm::SmallString's InternalLen to 0, because we want to
  // allocate its buffer on the heap. We use llvm::SmallString instead of
  // std::string, because llvm::raw_svector_ostream is a bit faster than
  // llvm::raw_string_ostream.
  llvm::SmallString<0> bitcode;
  llvm::raw_svector_ostream bitcode_ostream(bitcode);
  llvm::WriteBitcodeToFile(module, bitcode_ostream);

  llvm::Expected<std::unique_ptr<llvm::Module>> new_module =
      llvm::parseBitcodeFile(
          llvm::MemoryBufferRef(llvm::StringRef(bitcode.data(), bitcode.size()),
                                "split_module"),
          context);
  CHECK(new_module) << "Failed to parse bitcode "
                    << llvm::toString(new_module.takeError());

  return std::move(new_module.get());
}

}  // namespace

absl::StatusOr<GpuCompiler::BackendCompileResult>
GpuCompiler::CompileSingleModule(
    const HloModuleConfig& module_config,
    const se::DeviceDescription& device_description,
    const HloModule* debug_module, llvm::Module* llvm_module, bool relocatable,
    const CompileOptions& options, std::optional<int> shard_number) {
  {
    // This may print multiple lines per HLO compilation because of the
    // parallelized compilation of LLVM modules.
    // TODO(chokobole): Uncomment this. Dependency: XLA_SCOPED_LOGGING_TIMER_IF
    // XLA_SCOPED_LOGGING_TIMER_IF(
    //     absl::StrCat(
    //         "GpuCompiler::RunBackend - Running LLVM verifier for ",
    //         (debug_module != nullptr ? debug_module->name() : "(unknown)")),
    //     VLOG_IS_ON(4) && !options.is_autotuning_compilation);

    llvm_module->getContext().setDiagnosticHandlerCallBack(
        NullDiagnosticHandler, nullptr);

    std::string err;
    llvm::raw_string_ostream err_stream(err);

    // verifyModule() returns true if the module is broken.
    TF_RET_CHECK(!llvm::verifyModule(*llvm_module, &err_stream))
        << "Invalid LLVM IR before optimizations:\n"
        << err_stream.str()
        << "\nThis probably indicates a bug in the HLO -> LLVM IR "
           "lowering. Rerun with --zkx_dump_to to get the IR"
        << (debug_module
                ? absl::StrCat(" and looks for files with name containing: *",
                               FilenameFor(*debug_module, "", ""), "*")
                : ".");
  }

  TF_ASSIGN_OR_RETURN(
      BackendCompileResult result,
      CompileTargetBinary(module_config, llvm_module, device_description,
                          relocatable, debug_module, options, shard_number));

  const bool should_dump = DumpingEnabledForHloModule(
      debug_module ? debug_module->name() : "", module_config.debug_options());

  if (should_dump) {
    if (debug_module) {
      DumpIrIfEnabled(
          *debug_module, *llvm_module,
          /*optimized=*/true,
          shard_number.has_value() ? std::to_string(*shard_number) : "");
    } else {
      LOG(ERROR) << "Dumping is not implemented since the file name cannot be "
                    "inferred. Please implement (potentially MLIR) module -> "
                    "filename heuristic.";
    }
  }

  if (user_post_optimization_hook_) {
    user_post_optimization_hook_(*llvm_module);
  }

  return result;
}

namespace {

int CountFunctions(const llvm::Module& module) {
  int num_functions = 0;
  for (const llvm::Function& func : module.functions()) {
    if (!func.isDeclaration() &&
        func.getLinkage() == llvm::GlobalValue::LinkageTypes::ExternalLinkage) {
      ++num_functions;
    }
  }
  return num_functions;
}

// Returns the name of the single function in the module or empty string if it's
// not a single-function module.
std::string SingleFunctionName(const llvm::Module& module) {
  std::string name;
  for (const llvm::Function& func : module.functions()) {
    if (!func.isDeclaration() &&
        func.getLinkage() == llvm::GlobalValue::LinkageTypes::ExternalLinkage) {
      if (name.empty()) {
        // First function in a module: name the module with it.
        name = func.getName().str();
      } else {
        // Not the first function - the module is not cacheable.
        return "";
      }
    }
  }
  return name;
}

}  // namespace

absl::StatusOr<GpuCompiler::BackendCompileResult> GpuCompiler::CompileAndLink(
    const HloModuleConfig& module_config,
    CompileModuleResults& compile_module_results,
    const se::DeviceDescription& device_description,
    se::StreamExecutor* stream_exec, const CompileOptions& options,
    const HloModule* debug_module) {
  llvm::Module* llvm_module = &*compile_module_results.llvm_module;

  bool force_module_split =
      module_config.debug_options().zkx_llvm_force_inline_before_split();
  if (force_module_split) {
    for (llvm::Function& func : llvm_module->functions()) {
      if (func.getNumUses() > 0 && !func.isDeclaration()) {
        VLOG(4) << absl::StrFormat("Inlining function %s with %d users.\n",
                                   func.getName().str(), func.getNumUses());
        std::vector<llvm::CallInst*> calls_to_inline;
        for (auto* user : func.users()) {
          if (auto* call = llvm::dyn_cast<llvm::CallInst>(user)) {
            calls_to_inline.push_back(call);
          }
        }
        for (auto* call_to_inline : calls_to_inline) {
          llvm::InlineFunctionInfo inline_function_info;
          if (!llvm::InlineFunction(*call_to_inline, inline_function_info)
                   .isSuccess()) {
            return absl::InternalError("Can not inline function " +
                                       func.getName().str());
          }
        }
      }
    }
  }

  // Record the name of some constant global variables and their initializers.
  // We'll change the linkage type of these variables from external to internal
  // to ensure constant-folding works properly after calling llvm::SplitModule.
  llvm::DenseMap<llvm::StringRef, llvm::Constant*> const_initializer_map;
  llvm::Module& module_with_constants =
      (compile_module_results.llvm_module_constants == nullptr)
          ? *llvm_module
          : *compile_module_results.llvm_module_constants;
  for (llvm::GlobalVariable& gv : module_with_constants.globals()) {
    if (gv.hasName() && gv.isConstant() && gv.hasInitializer() &&
        gv.hasExternalLinkage()) {
      llvm::Constant* initializer = gv.getInitializer();
      unsigned int num_elements = 0;
      if (auto* caz =
              llvm::dyn_cast<llvm::ConstantAggregateZero>(initializer)) {
        num_elements = caz->getElementCount().getFixedValue();
      } else if (auto* cds = llvm::dyn_cast<llvm::ConstantDataSequential>(
                     initializer)) {
        num_elements = cds->getNumElements();
      }
      if (num_elements > 0) {
        const_initializer_map[gv.getName()] = initializer;
      }
    }
  }

  DumpIrIfEnabled(*debug_module, *llvm_module,
                  /*optimized=*/false, "inlined");

  std::string_view cache_path =
      module_config.debug_options().zkx_gpu_kernel_cache_file();
  const bool use_cache = !cache_path.empty();

  struct NamedModule {
    // The string is the function name for single-function modules (used to
    // cache them), empty for all other modules.
    std::string name;
    std::unique_ptr<llvm::Module> module;
  };
  std::vector<NamedModule> llvm_modules;
  MaybeOwningThreadPool thread_pool = CreateMaybeOwningThreadPool(
      /*parallelism=*/module_config.debug_options()
          .zkx_gpu_force_compilation_parallelism(),
      /*default_thread_pool=*/options.thread_pool,
      /*default_parallelism=*/1);
  // Only single-function module are cacheable -> for caching try to get 1
  // function per module. If caching is not used limit the number of modules to
  // the number of threads.
  int num_modules = CountFunctions(*llvm_module);
  if (thread_pool.get() != nullptr && !use_cache) {
    num_modules = std::max(1, std::min(thread_pool->NumThreads(), num_modules));
  }
  if (compile_module_results.llvm_module_constants != nullptr) {
    llvm_modules.reserve(num_modules + 1);
    llvm_modules.push_back(
        {"", std::move(compile_module_results.llvm_module_constants)});
  } else {
    llvm_modules.reserve(num_modules);
  }
  int single_function_module_count = 0;
  llvm::SplitModule(
      *llvm_module, num_modules,
      [&](std::unique_ptr<llvm::Module> module) {
        // Change the linkage type of some global constant variables to internal
        for (llvm::GlobalVariable& gv : module->globals()) {
          if (gv.hasName() && gv.isConstant() && !gv.hasInitializer() &&
              const_initializer_map.count(gv.getName()) != 0) {
            gv.setInitializer(const_initializer_map[gv.getName()]);
            gv.setLinkage(llvm::GlobalValue::InternalLinkage);
          }
        }
        const std::string name = SingleFunctionName(*module);
        if (!name.empty()) {
          ++single_function_module_count;
        }
        llvm_modules.push_back({name, std::move(module)});
      },
      /*PreserveLocals=*/true, /*RoundRobin=*/true);
  VLOG(2) << "Single-function cacheable modules: "
          << single_function_module_count << " / " << llvm_modules.size();

  struct NamedCompileResult {
    // Single function name or empty just like for llvm_modules.
    std::string name;
    absl::StatusOr<BackendCompileResult> result;
  };
  std::vector<NamedCompileResult> compile_results(llvm_modules.size());
  if (thread_pool.get() != nullptr) {
    absl::BlockingCounter counter(llvm_modules.size());
    for (int i = 0; i < llvm_modules.size(); ++i) {
      thread_pool.get_mutable()->Schedule(
          [&compile_results, i, &llvm_modules, &counter, this, &module_config,
           &device_description, &debug_module, &options] {
            // Each thread has its own context to avoid race conditions.
            llvm::LLVMContext new_context;
            std::unique_ptr<llvm::Module> new_module =
                CopyToContext(*llvm_modules.at(i).module, new_context);
            compile_results.at(i) = {
                llvm_modules.at(i).name,
                CompileSingleModule(module_config, device_description,
                                    debug_module, new_module.get(),
                                    /*relocatable=*/true, options,
                                    /*shard_number=*/i)};
            counter.DecrementCount();
          });
    }
    counter.Wait();
  } else {
    for (int i = 0; i < llvm_modules.size(); ++i) {
      compile_results.at(i) = {
          llvm_modules.at(i).name,
          CompileSingleModule(module_config, device_description, debug_module,
                              &*llvm_modules.at(i).module,
                              /*relocatable=*/true, options,
                              /*shard_number=*/i)};
    }
  }

  std::string ptx_snippets;
  std::vector<std::vector<uint8_t>> binaries_to_link;
  binaries_to_link.reserve(compile_results.size());
  std::vector<KernelReuseCache::NamedBinary> binaries_to_cache;
  binaries_to_cache.reserve(single_function_module_count);
  for (const auto& [name, maybe_result] : compile_results) {
    TF_ASSIGN_OR_RETURN(auto result, maybe_result);
    if (result.binary.empty()) {
      continue;
    }
    ptx_snippets += result.asm_text;
    ptx_snippets += "\n";
    binaries_to_link.push_back(result.binary);
    if (!name.empty()) {
      binaries_to_cache.push_back({name, result.binary});
    }
  }

  if (use_cache) {
    std::string resolved_path;
    if (!tsl::io::ResolveTestPrefixes(cache_path, resolved_path)) {
      return absl::FailedPreconditionError(
          absl::StrFormat("File path can not be resolved: %s", cache_path));
    }
    // current_cache contains new kernels from the current compilation and
    // kernels to reuse from previous compilations if some were loaded from the
    // cache file.
    const CompilationCacheProto& current_cache =
        compile_module_results.kernel_compilation_cache;
    const bool cache_file_exists =
        tsl::Env::Default()->FileExists(resolved_path).ok();
    if (cache_file_exists) {
      // Pick reused binaries from previous compilations needed to link the
      // current executable.
      int loaded_kernel_count = 0;
      for (const auto& [name, entry] : current_cache.entries()) {
        if (llvm_module->getFunction(name) != nullptr) {
          VLOG(5) << "Using the just compiled kernel for " << name;
          TF_RET_CHECK(entry.binary().empty())
              << name
              << " is a just compiled kernel and is not expected to have a "
                 "binary yet.";
          continue;
        }
        const uint8_t* binary =
            reinterpret_cast<const uint8_t*>(entry.binary().data());
        binaries_to_link.push_back(
            std::vector<uint8_t>(binary, binary + entry.binary().size()));
        VLOG(5) << "Using " << name << " from cache: " << entry.binary().size();
        ++loaded_kernel_count;
      }
      VLOG(2) << "Using " << loaded_kernel_count << " / "
              << current_cache.entries_size() << " cached kernels.";
    }
    if (!binaries_to_cache.empty()) {
      TF_RETURN_IF_ERROR(
          UpdateDiskKernelCache(resolved_path, /*do_append=*/cache_file_exists,
                                current_cache, binaries_to_cache));
    }
  }

  auto maybe_backend_result =
      LinkModules(device_description, stream_exec, std::move(binaries_to_link),
                  module_config.debug_options());
  if (!maybe_backend_result.ok()) {
    LOG(ERROR) << "The CUDA linking API did not work. Please use ZKX_FLAGS="
                  "--zkx_gpu_enable_llvm_module_compilation_parallelism=false "
                  "to bypass it, but expect to get longer compilation time due "
                  "to the lack of multi-threading. Original error: "
               << maybe_backend_result.status();
    return maybe_backend_result.status();
  }
  VLOG(4) << "Binary size after linking [B]: " << maybe_backend_result->size();
  compile_module_results.kernel_compilation_cache.Clear();
  return BackendCompileResult{ptx_snippets, std::move(*maybe_backend_result)};
}

absl::StatusOr<GpuCompiler::CompileResultWithMetadata>
GpuCompiler::CompileToBackendResult(
    HloModule* module, llvm::LLVMContext* llvm_context,
    se::StreamExecutor* executor, const CompileOptions& options,
    const se::DeviceDescription& gpu_device_info) {
  tsl::profiler::TraceMe traceme("GpuCompiler::CompileToBackendResult");

  // TODO(chokobole): Uncomment this. Dependency: RunPreSchedulingPasses
  // TF_RETURN_IF_ERROR(RunPreSchedulingPasses(module, executor,
  // gpu_device_info));
  TF_ASSIGN_OR_RETURN(
      ScheduleMetadata schedule_metadata,
      ScheduleGpuModule(module, pointer_size_, gpu_device_info));
  // TODO(chokobole): Uncomment this. Dependency: RunPostSchedulingPipelines
  // TF_RETURN_IF_ERROR(RunPostSchedulingPipelines(
  //     module, schedule_metadata.scheduler_mem_limit, gpu_device_info));
  std::ignore = schedule_metadata;

  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      se::PlatformManager::PlatformWithId(PlatformId()));

  // Test whether LinkModules is supported.
  bool can_use_link_modules = (executor != nullptr);
  if (can_use_link_modules) {
    TF_ASSIGN_OR_RETURN(can_use_link_modules,
                        CanUseLinkModules(module->config(), gpu_device_info));
  }
  const bool split_modules =
      can_use_link_modules &&
      module->config()
          .debug_options()
          .zkx_gpu_enable_llvm_module_compilation_parallelism();
  const bool use_cache =
      split_modules &&
      !module->config().debug_options().zkx_gpu_kernel_cache_file().empty();

  CompileModuleResults compile_module_results;

  {
    llvm_ir::LLVMCommandLineOptionsLock llvm_options_lock(
        GetLLVMCommandLineOptions(module->config().debug_options()));
    // Compile the module
    TF_ASSIGN_OR_RETURN(
        compile_module_results,
        CompileModuleToLlvmIr(
            module, llvm_context, target_triple_, data_layout_,
            platform->Name(), platform->id(), gpu_device_info,
            GetCanShareBuffer(gpu_device_info), BufferSizeBytesFunction(),
            /*split_constants_module=*/use_cache));
  }

  if (user_pre_optimization_hook_) {
    user_pre_optimization_hook_(*compile_module_results.llvm_module);
    if (compile_module_results.llvm_module_constants != nullptr) {
      user_pre_optimization_hook_(
          *compile_module_results.llvm_module_constants);
    }
  }

  DumpIrIfEnabled(*module, *compile_module_results.llvm_module,
                  /*optimized=*/false);
  if (compile_module_results.llvm_module_constants != nullptr) {
    DumpIrIfEnabled(*module, *compile_module_results.llvm_module_constants,
                    /*optimized=*/false, "constants");
  }

  BackendCompileResult backend_result;
  // Disable multi-threading during deviceless AOT compilation.
  // TODO(anlunx): Enable multi-threading once deviceless AOT compilation is
  // enabled.
  if (split_modules) {
    TF_ASSIGN_OR_RETURN(
        backend_result,
        CompileAndLink(module->config(), compile_module_results,
                       gpu_device_info, executor, options, module));
  } else {
    CHECK(compile_module_results.llvm_module_constants == nullptr);
    TF_ASSIGN_OR_RETURN(
        backend_result,
        CompileSingleModule(module->config(), gpu_device_info, module,
                            &*compile_module_results.llvm_module,
                            /*relocatable=*/false, options,
                            /*shard_number=*/std::nullopt));
  }
  // TODO(chokobole): Uncomment this. Dependency: RecordZkxDeviceBinarySize
  // RecordZkxDeviceBinarySize(backend_result.binary.size());
  if (DumpingEnabledForHloModule(*module)) {
    DumpToFileInDirOrStdout(
        *module, "", "thunk_sequence.txt",
        compile_module_results.executable->ToString(/*indent=*/0));
  }

  return CompileResultWithMetadata{std::move(backend_result),
                                   std::move(compile_module_results)};
}

absl::StatusOr<std::unique_ptr<Executable>> GpuCompiler::RunBackend(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  tsl::profiler::ScopedAnnotation backend_annotation{[&] {
    return absl::StrFormat("ZkxCompileBackend:#module=%s,program_id=%d#",
                           module->name(), module->unique_id());
  }};

  // TODO(chokobole): Uncomment this. Dependency: RecordGpuCompilerStacktrace
  // RecordGpuCompilerStacktrace();

  const DebugOptions& debug_opts = module->config().debug_options();
  TF_ASSIGN_OR_RETURN(TargetConfig gpu_target_config,
                      GetTargetConfig(options, debug_opts, stream_exec));

  if (DumpingEnabledForHloModule(*module)) {
    std::string textproto;
    google::protobuf::TextFormat::PrintToString(gpu_target_config.ToProto(),
                                                &textproto);
    DumpToFileInDirOrStdout(*module, "", "gpu_target_config.pbtxt", textproto);
  }

  if (!options.is_autotuning_compilation) {
    VLOG(1) << "Starting to compile HLO module " << module->name();
  }

  // TODO(chokobole): Uncomment this. Dependency: XLA_SCOPED_LOGGING_TIMER_IF
  // XLA_SCOPED_LOGGING_TIMER_IF(
  //     absl::StrCat("GpuCompiler::RunBackend for ", module->name()),
  //     !options.is_autotuning_compilation);
  std::string slow_compilation_msg =
      absl::StrCat("Compiling module ", module->name());
  auto slow_compile_alarm = SlowCompilationAlarm(slow_compilation_msg);

  if (options.is_autotuning_compilation) {
    if (module->config().debug_options().zkx_embed_ir_in_executable()) {
      LOG(WARNING) << "Doing autotuning compilations with "
                      "zkx_embed_ir_in_executable wastes memory!";
    }
  }

  llvm::LLVMContext llvm_context;
  const se::DeviceDescription& gpu_device_info =
      gpu_target_config.device_description;

  if (module->config().hlo_profiling_enabled() || VLOG_IS_ON(1)) {
    // TODO(chokobole): Uncomment this. Dependency: GpuHloCostAnalysis
    // HloCostAnalysis::Options cost_analysis_options{ShapeSizeBytesFunction()};
    // cost_analysis_options.set_bytes_per_second(
    //     gpu_device_info.memory_bandwidth());
    // GpuHloCostAnalysis cost_analysis(cost_analysis_options, gpu_device_info);
    // TF_RETURN_IF_ERROR(module->entry_computation()->Accept(&cost_analysis));
    // if (!options.is_autotuning_compilation) {
    //   VLOG(1) << "HLO memory read+written: "
    //           << tsl::strings::HumanReadableNumBytes(
    //                  cost_analysis.bytes_accessed());
    // }
    // if (module->config().hlo_profiling_enabled()) {
    //   LOG(ERROR) << "--zkx_hlo_profile for GPU is unsupported.";
    // }
  }

  TF_ASSIGN_OR_RETURN(
      CompileResultWithMetadata res,
      CompileToBackendResult(module.get(), &llvm_context, stream_exec, options,
                             gpu_device_info));

  if (DumpingEnabledForHloModule(*module)) {
    DumpToFileInDirOrStdout(
        *module, "", "thunk_sequence.txt",
        res.compile_module_results.executable->ToString(/*indent=*/0));
  }

  // The module is being moved into the GpuExecutable below and we need to
  // read a few config values from the module, before it becomes invalid.
  bool embed_ir_in_executable =
      module->config().debug_options().zkx_embed_ir_in_executable();
  int64_t debug_buffer_assignment_show_max =
      module->config().debug_options().zkx_debug_buffer_assignment_show_max();

  tsl::profiler::ScopedAnnotation annotation([&] {
    return absl::StrFormat("ZkxCreateGpuExecutable:#module=%s#",
                           module->name());
  });
  TF_ASSIGN_OR_RETURN(
      auto gpu_executable,
      GpuExecutable::Create(GpuExecutable::Params{
          /*asm_text=*/(options.is_autotuning_compilation &&
                        !res.backend_result.binary.empty())
              ? std::string()
              : std::move(res.backend_result.asm_text),
          /*binary=*/std::move(res.backend_result.binary),
          /*gpu_version=*/gpu_device_info.gpu_compute_capability(),
          /*executable=*/std::move(res.compile_module_results.executable),
          /*constants=*/std::move(res.compile_module_results.constants),
          /*output_info=*/std::move(res.compile_module_results.output_info),
          /*module_name=*/std::move(res.compile_module_results.module_name),
          /*output_shape=*/std::move(res.compile_module_results.output_shape),
          /*mlir_allocations=*/
          (res.compile_module_results.use_original_allocations
               ? std::optional<std::vector<BufferAllocation>>()
               : std::move(res.compile_module_results.allocations)),
          /*buffer_assignment=*/
          std::move(res.compile_module_results.buffer_assignment),
          /*debug_buffer_assignment_show_max=*/
          debug_buffer_assignment_show_max,
          /*debug_module=*/options.is_autotuning_compilation
              ? std::unique_ptr<HloModule>()
              : std::move(module),
          /*enable_debug_info_manager=*/!options.is_autotuning_compilation}));

  if (embed_ir_in_executable) {
    std::string ir_module_string_before_opt =
        llvm_ir::DumpToString(res.compile_module_results.llvm_module.get());
    gpu_executable->set_ir_module_string(ir_module_string_before_opt);
    DCHECK_NE("", ir_module_string_before_opt);
  }

  // TODO(chokobole): Uncomment this. Dependency: IncrementCompiledProgramsCount
  // IncrementCompiledProgramsCount();

  if (!options.is_autotuning_compilation && gpu_executable->has_module()) {
    // Dump computation proto state and buffer assignment for
    // CompiledMemoryAnalysis.
    auto hlo_proto = std::make_unique<HloProto>();
    *hlo_proto->mutable_buffer_assignment() =
        gpu_executable->buffer_assignment()->ToProto();
    gpu_executable->set_hlo_proto(std::move(hlo_proto));
    gpu_executable->set_debug_info(
        gpu_executable->buffer_assignment()->StatsString(
            /*report_total_fragmentation=*/true));
  }

  return static_cast<std::unique_ptr<Executable>>(std::move(gpu_executable));
}

absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
GpuCompiler::CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                                const AotCompilationOptions& options) {
  // Check that we are on the platform (CUDA or ROCm) that was chosen for AOT
  // compilation.
  CHECK_EQ(options.PlatformId(), PlatformId());

  std::vector<std::unique_ptr<HloModule>> modules =
      module_group->ConsumeModules();

  std::vector<std::unique_ptr<HloModule>> optimized_modules;
  optimized_modules.reserve(modules.size());

  for (std::unique_ptr<HloModule>& module : modules) {
    if (!module->has_schedule()) {
      tsl::profiler::ScopedAnnotation annotation{[&] {
        return absl::StrFormat("ZkxCompile:#module=%s,program_id=%d#",
                               module->name(), module->unique_id());
      }};
      CompileOptions compile_options;
      compile_options.device_allocator = options.device_allocator();
      compile_options.target_config = options.target_config();
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<HloModule> optimized_module,
          RunHloPasses(std::move(module), options.executor(), compile_options));
      optimized_modules.push_back(std::move(optimized_module));
    } else {
      optimized_modules.push_back(std::move(module));
    }
  }

  modules = std::move(optimized_modules);

  std::vector<std::unique_ptr<AotCompilationResult>> results;

  const std::optional<Compiler::TargetConfig>& target_config =
      options.target_config();
  CHECK(target_config.has_value() || options.executor() != nullptr);
  const se::DeviceDescription& gpu_device_info =
      target_config.has_value() ? target_config->device_description
                                : options.executor()->GetDeviceDescription();
  for (const std::unique_ptr<HloModule>& module : modules) {
    llvm::LLVMContext llvm_context;
    TF_ASSIGN_OR_RETURN(
        CompileResultWithMetadata res,
        CompileToBackendResult(module.get(), &llvm_context, options.executor(),
                               {options.device_allocator()}, gpu_device_info));

    // Create GpuThunkAotCompilationResult if thunk runtime is enabled.
    TF_ASSIGN_OR_RETURN(
        results.emplace_back(),
        GpuThunkAotCompilationResult::FromModule(
            module.get(), res.compile_module_results.buffer_assignment.get(),
            res.backend_result.asm_text, res.backend_result.binary));
  }

  return std::move(results);
}

HloCostAnalysis::ShapeSizeFunction GpuCompiler::ShapeSizeBytesFunction() const {
  // Capture just the pointer size, not the entire GpuCompiler object.
  return [pointer_size = pointer_size_](const Shape& shape) {
    return GetSizeOfShape(shape, pointer_size);
  };
}

absl::StatusOr<std::unique_ptr<AotCompilationResult>> GpuCompiler::Export(
    Executable* executable) const {
  auto* gpu_executable = tensorflow::down_cast<GpuExecutable*>(executable);
  if (!gpu_executable) return absl::InternalError("GpuExecutable is null");

  return GpuThunkAotCompilationResult::FromModule(
      &gpu_executable->module(), gpu_executable->buffer_assignment(),
      gpu_executable->text(), gpu_executable->binary());
}

absl::Status GpuCompiler::LoadAutotuneResultsFromFile(
    const DebugOptions& debug_options) {
  // We are doing this before the timer is started.
  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: AutotunerUtil::LoadAutotuneResultsFromFile
  // clang-format on
  // if (std::string_view file_path =
  //         debug_options.zkx_gpu_load_autotune_results_from();
  //     !file_path.empty()) {
  //   static absl::once_flag once;
  //   absl::Status status = absl::OkStatus();
  //   absl::call_once(once, [&file_path, &status] {
  //     status = AutotunerUtil::LoadAutotuneResultsFromFile(file_path);
  //   });
  //   TF_RETURN_IF_ERROR(status);
  // }
  // return absl::OkStatus();
  return absl::UnimplementedError(
      "not implemented for LoadAutotuneResultsFromFile");
}

absl::Status GpuCompiler::SerializeAutotuneResultsToFile(
    const DebugOptions& debug_options) {
  // We are doing this after the timer is finished.
  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: AutotunerUtil::SerializeAutotuneResultsToFile
  // clang-format on
  // if (std::string_view file_path =
  //         debug_options.zkx_gpu_dump_autotune_results_to();
  //     !file_path.empty()) {
  //   // Warning: This writes the autotune results at every compilation,
  //   // possibly multiple times per process.
  //   TF_RETURN_IF_ERROR(
  //       AutotunerUtil::SerializeAutotuneResultsToFile(file_path));
  // }
  // return absl::OkStatus();
  return absl::UnimplementedError(
      "not implemented for SerializeAutotuneResultsToFile");
}

absl::StatusOr<std::unique_ptr<AotCompilationResult>>
GpuCompiler::LoadAotCompilationResult(
    const std::string& serialized_aot_result) {
  return LoadAotCompilationResultStatic(serialized_aot_result);
}

// static
absl::StatusOr<std::unique_ptr<AotCompilationResult>>
GpuCompiler::LoadAotCompilationResultStatic(
    const std::string& serialized_aot_result) {
  return GpuThunkAotCompilationResult::FromString(serialized_aot_result);
}

}  // namespace zkx::gpu
