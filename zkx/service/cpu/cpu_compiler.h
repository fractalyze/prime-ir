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

#ifndef ZKX_SERVICE_CPU_CPU_COMPILER_H_
#define ZKX_SERVICE_CPU_CPU_COMPILER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "llvm/Target/TargetMachine.h"

#include "zkx/backends/cpu/codegen/target_machine_features.h"
#include "zkx/cpu_function_runtime.h"
#include "zkx/hlo/ir/hlo_schedule.h"
#include "zkx/service/buffer_assignment.h"
#include "zkx/service/cpu/cpu_executable.h"
#include "zkx/service/hlo_profile_printer_data.pb.h"
#include "zkx/service/llvm_compiler.h"

namespace zkx::cpu {

class CpuExecutable;

// This class wraps the configurability options that LLVM exposes including: the
// target triple, the target cpu and the target features. It also includes the
// desired linkage name for the computation entry point.
class CpuAotCompilationOptions : public AotCompilationOptions {
 public:
  // Relocation models available for compilation.
  enum class RelocationModel {
    // Corresponds to the -fno-pic compiler option.
    Static,
    // Corresponds to the -fpic compiler option.
    SmallPic,
    // Corresponds to the -fPIC compiler option.
    BigPic,
    // Corresponds to the -fpie compiler option.
    SmallPie,
    // Corresponds to the -fPIE compiler option.
    BigPie
  };

  CpuAotCompilationOptions(std::string triple, std::string cpu_name,
                           std::string features, std::string entry_point_name,
                           RelocationModel relocation_model)
      : triple_(std::move(triple)),
        cpu_name_(std::move(cpu_name)),
        features_(std::move(features)),
        entry_point_name_(std::move(entry_point_name)),
        relocation_model_(relocation_model) {}

  ~CpuAotCompilationOptions() override = default;

  se::Platform::Id PlatformId() const override;

  // The triple used for compilation, similar to clang's -target flag.
  const std::string& triple() const { return triple_; }
  // The CPU name used for compilation, similar to clang's -mcpu flag.
  const std::string& cpu_name() const { return cpu_name_; }
  // The target features used for compilation ("+avx2", "+neon", etc).
  const std::string& features() const { return features_; }
  // The name to be used for the compiled code's entry point.
  const std::string& entry_point_name() const { return entry_point_name_; }
  // The relocation model used for compilation.
  RelocationModel relocation_model() const { return relocation_model_; }

 private:
  const std::string triple_;
  const std::string cpu_name_;
  const std::string features_;
  const std::string entry_point_name_;
  const RelocationModel relocation_model_;
};

class CpuAotCompilationResult : public AotCompilationResult {
 public:
  CpuAotCompilationResult(
      ObjectFileData object_file_data,
      std::vector<cpu_function_runtime::BufferInfo> buffer_infos,
      int64_t result_buffer_index, std::unique_ptr<HloModule> module,
      std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data)
      : object_file_data_(std::move(object_file_data)),
        buffer_infos_(std::move(buffer_infos)),
        result_buffer_index_(result_buffer_index),
        module_(std::move(module)),
        hlo_profile_printer_data_(std::move(hlo_profile_printer_data)) {}
  ~CpuAotCompilationResult() override = default;

  HloProfilePrinterData* hlo_profile_printer_data() const {
    return hlo_profile_printer_data_.get();
  }

  const ObjectFileData& object_file_data() const { return object_file_data_; }
  const std::vector<cpu_function_runtime::BufferInfo>& buffer_infos() const {
    return buffer_infos_;
  }
  int64_t result_buffer_index() const { return result_buffer_index_; }

  const HloModule* optimized_module() const override { return module_.get(); }
  std::unique_ptr<HloModule> consume_optimized_module() override {
    return std::move(module_);
  }

 private:
  // Contains the compiled computation: an object file.
  const ObjectFileData object_file_data_;

  // A list of BufferInfo objects describing the buffers used by the ZKX
  // computation.
  const std::vector<cpu_function_runtime::BufferInfo> buffer_infos_;

  // Contains which buffer index into |buffer_sizes| was designated to the
  // result of the computation.  This buffer should be passed into the output
  // parameter when calling the compiled computation.
  const int64_t result_buffer_index_;

  // Contains the optimized HLO module.
  std::unique_ptr<HloModule> module_;

  // Contains an instance of HloProfilePrinterData if HLO profiling is enabled,
  // otherwise is nullptr.
  std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data_;
};

// CPU-targeting implementation of the ZKX Compiler interface.
//
// The compiler translates ZKX HLO code into LLVM IR and uses LLVM's JIT
// infrastructure to create an executable "blob" that can then be returned
// wrapped in CpuExecutable and actually invoked.
class CpuCompiler : public LlvmCompiler {
 public:
  CpuCompiler() = default;
  ~CpuCompiler() override = default;

  absl::StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<se::StreamExecutor*>> executors,
      const CompileOptions& options) override;

  absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* executor,
      const CompileOptions& options) override;

  absl::StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* executor,
      const CompileOptions& options) override;

  absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     const AotCompilationOptions& options) override {
    return absl::UnimplementedError("...");
  }

  se::Platform::Id PlatformId() const override;

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override {
    return CpuExecutable::ShapeSizeBytes;
  }

  absl::StatusOr<std::unique_ptr<AotCompilationResult>> Export(
      Executable* executable) const override;

  // Returns a (deserialized) AotCompilationResult from a serialized
  // AotCompilationResult.
  absl::StatusOr<std::unique_ptr<AotCompilationResult>>
  LoadAotCompilationResult(const std::string& serialized_aot_result) override;

  absl::StatusOr<HloSchedule> CreateHloSchedule(
      const HloModule& hlo_module) const;

  absl::StatusOr<std::unique_ptr<BufferAssignment>> CreateBufferAssignment(
      const HloModule& module) const;

 private:
  // Runs the HLO passes which are necessary for both optimizations and
  // correctness.
  absl::Status RunHloPasses(HloModule* module, bool is_aot_compile,
                            llvm::TargetMachine* target_machine,
                            const CompileOptions& compile_options);

  // Runs HLO passes up to and including layout assignment.
  absl::Status RunHloPassesThroughLayoutAssn(
      HloModule* module, bool /*is_aot_compile*/,
      TargetMachineFeatures* target_machine_features);

  // Runs HLO passes after layout assignment.
  absl::Status RunHloPassesAfterLayoutAssn(
      HloModule* module, bool is_aot_compile,
      TargetMachineFeatures* target_machine_features,
      const CompileOptions& compile_options);

  absl::StatusOr<std::unique_ptr<CpuExecutable>> CompileCpuExecutable(
      std::unique_ptr<HloModule> module);

  CpuCompiler(const CpuCompiler&) = delete;
  CpuCompiler& operator=(const CpuCompiler&) = delete;
};

}  // namespace zkx::cpu

#endif  // ZKX_SERVICE_CPU_CPU_COMPILER_H_
