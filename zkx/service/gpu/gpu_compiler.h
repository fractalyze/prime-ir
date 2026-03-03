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

#ifndef ZKX_SERVICE_GPU_GPU_COMPILER_H_
#define ZKX_SERVICE_GPU_GPU_COMPILER_H_

#include <stdint.h>

#include <optional>
#include <string>
#include <vector>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include "xla/tsl/platform/thread_pool.h"
#include "zkx/hlo/analysis/hlo_dataflow_analysis.h"
#include "zkx/service/gpu/buffer_sharing.h"
#include "zkx/service/gpu/compile_module_to_llvm_ir.h"
#include "zkx/service/gpu/transforms/algebraic_simplifier.h"
#include "zkx/service/llvm_compiler.h"
#include "zkx/shape_util.h"
#include "zkx/stream_executor/device_description.h"
#include "zkx/stream_executor/platform.h"
#include "zkx/stream_executor/stream_executor.h"
#include "zkx/zkx.pb.h"

namespace zkx::gpu {

// The GPU compiler generates efficient GPU executables.
class GpuCompiler : public LlvmCompiler {
 public:
  GpuCompiler(se::Platform::Id platform_id, const char* target_triple,
              const char* data_layout);

  using LlvmCompiler::Compile;

  // An attached device is passed in via stream_exec. We get GPU configuration
  // from the attached device OR from the `options` struct (in which case the
  // attached device is ignored during the compilation).
  // If you call this directly, follow it with RunBackend rather than Compile.
  absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  absl::StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     AotCompilationOptions const& options) override;

  se::Platform::Id PlatformId() const override { return platform_id_; }

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override;

  // Returns a (deserialized) AotCompilationResult from a serialized
  // AotCompilationResult.
  absl::StatusOr<std::unique_ptr<AotCompilationResult>>
  LoadAotCompilationResult(const std::string& serialized_aot_result) override;

  // Stateless version of the same function.
  static absl::StatusOr<std::unique_ptr<AotCompilationResult>>
  LoadAotCompilationResultStatic(const std::string& serialized_aot_result);

  absl::StatusOr<std::unique_ptr<AotCompilationResult>> Export(
      Executable* executable) const override;

  absl::Status RunPostSchedulingPipelines(
      HloModule* module, int64_t scheduler_mem_limit,
      const se::DeviceDescription& gpu_device_info) const;

  std::string target_triple() const { return target_triple_; }
  std::string data_layout() const { return data_layout_; }

  const char* GetDataLayout() const { return data_layout_; }

  const char* GetTargetTriple() const { return target_triple_; }

  int64_t GetPointerSize() const { return pointer_size_; }

  static absl::StatusOr<Compiler::TargetConfig> GetTargetConfig(
      const Compiler::CompileOptions& options, const DebugOptions& debug_opts,
      se::StreamExecutor* executor);

  virtual HloDataflowAnalysis::CanShareBuffer GetCanShareBuffer(
      const se::DeviceDescription& device_description) const {
    return [&](const HloInstruction* user, const HloInstruction* operand,
               const ShapeIndex& user_index) -> std::optional<bool> {
      return FusionCanShareBufferHint(user, operand, user_index,
                                      device_description);
    };
  }

  virtual absl::StatusOr<bool> CanUseLinkModules(
      const HloModuleConfig& config,
      const se::DeviceDescription& device_description) {
    return false;
  }

  static AlgebraicSimplifierOptions GetAlgebraicSimplifierOptions(
      const HloModuleConfig& config);

 protected:
  struct BackendCompileResult {
    std::string asm_text;
    std::vector<uint8_t> binary;
  };

  // During compilation with device, stream_exec != null and autotune_results
  // == null. During deviceless AOT compilation, stream_exec == null and
  // autotune_results != null.
  // thread_pool is used to speed up compilation during autotuning.
  virtual absl::Status OptimizeHloPostLayoutAssignment(
      HloModule* hlo_module, se::StreamExecutor* stream_exec,
      const CompileOptions& options, const TargetConfig& gpu_target_config,
      tsl::thread::ThreadPool* thread_pool);

 private:
  struct CompileResultWithMetadata {
    BackendCompileResult backend_result;
    CompileModuleResults compile_module_results;
  };

  // Schedule and compile the module.
  absl::StatusOr<CompileResultWithMetadata> CompileToBackendResult(
      HloModule* module, llvm::LLVMContext* llvm_context,
      se::StreamExecutor* executor, const CompileOptions& options,
      const se::DeviceDescription& gpu_device_info);

  absl::StatusOr<BackendCompileResult> CompileAndLink(
      const HloModuleConfig& module_config,
      CompileModuleResults& compile_module_results,
      const se::DeviceDescription& device_description,
      se::StreamExecutor* stream_exec, const CompileOptions& options,
      const HloModule* debug_module);

  absl::StatusOr<BackendCompileResult> CompileSingleModule(
      const HloModuleConfig& module_config,
      const se::DeviceDescription& device_description,
      const HloModule* debug_module, llvm::Module* llvm_module,
      bool relocatable, const CompileOptions& options,
      std::optional<int> shard_number);

  absl::Status LoadAutotuneResultsFromFile(const DebugOptions& debug_options);
  absl::Status SerializeAutotuneResultsToFile(
      const DebugOptions& debug_options);

  absl::Status RunPreSchedulingPasses(
      HloModule* module, se::StreamExecutor* stream_exec,
      const se::DeviceDescription& gpu_device_info);
  absl::Status RunCollectiveScheduleLinearizerPasses(
      HloModule* hlo_module, se::StreamExecutor* stream_exec);

  // During compilation with device, stream_exec != null and autotune_results
  // == null. During deviceless AOT compilation, stream_exec == null and
  // autotune_results != null.
  absl::Status OptimizeHloModule(HloModule* hlo_module,
                                 se::StreamExecutor* stream_exec,
                                 const CompileOptions& options,
                                 const TargetConfig& gpu_target_config);

  // TODO(timshen): Replace `debug_module` with some portable debug information
  // that accommodates both HLO and MLIR.
  virtual absl::StatusOr<BackendCompileResult> CompileTargetBinary(
      const HloModuleConfig& module_config, llvm::Module* llvm_module,
      const se::DeviceDescription& device_description, bool relocatable,
      const HloModule* debug_module, const CompileOptions& options,
      std::optional<int> shard_number) = 0;

  absl::Status PrepareHloModuleForIrEmitting(
      HloModule* hlo_module, const se::DeviceDescription& device_description);

  virtual absl::StatusOr<std::vector<uint8_t>> LinkModules(
      const se::DeviceDescription& device_description,
      se::StreamExecutor* stream_exec,
      std::vector<std::vector<uint8_t>> modules,
      const DebugOptions& debug_options) {
    return absl::UnimplementedError("LinkModules is not implemented.");
  }

  se::Platform::Id platform_id_;

  // The triple that represents our target.
  const char* target_triple_;

  // The data layout of the emitted module.
  const char* data_layout_;

  // The size in bytes of a pointer. Used by ShapeSizeBytesFunction.
  const int64_t pointer_size_;

  GpuCompiler(const GpuCompiler&) = delete;
  GpuCompiler& operator=(const GpuCompiler&) = delete;

  // Returns the LLVM command line options that we use for compilation.
  // THey need to be set globally whenever we call into LLVM.
  virtual std::vector<std::string> GetLLVMCommandLineOptions(
      const DebugOptions& debug_options) const = 0;
};

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_GPU_COMPILER_H_
