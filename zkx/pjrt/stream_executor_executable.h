/* Copyright 2023 The OpenXLA Authors.
Copyright 2026 The ZKX Authors.

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

#ifndef ZKX_PJRT_STREAM_EXECUTOR_EXECUTABLE_H_
#define ZKX_PJRT_STREAM_EXECUTOR_EXECUTABLE_H_

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "zkx/pjrt/pjrt_executable.h"

namespace zkx {

class StreamExecutorExecutable : public PjRtExecutable {
 public:
  StreamExecutorExecutable(
      const CompileOptions& compile_options,
      std::vector<std::unique_ptr<AotCompilationResult>> executables,
      int num_replicas, int num_partitions, std::string_view name,
      std::string_view fingerprint,
      std::optional<std::vector<std::vector<std::string_view>>>
          output_memory_kinds)
      : compile_options_(compile_options),
        aot_executables_(std::move(executables)),
        num_replicas_(num_replicas),
        num_partitions_(num_partitions),
        name_(name),
        fingerprint_(fingerprint),
        output_memory_kinds_(std::move(output_memory_kinds)) {}

  absl::StatusOr<std::string> SerializeExecutable() const override;

  std::string_view name() const override { return name_; }
  int num_replicas() const override { return num_replicas_; }
  int num_partitions() const override { return num_partitions_; }
  absl::StatusOr<CompileOptions> GetCompileOptions() const override {
    return compile_options_;
  }
  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override {
    return absl::UnimplementedError("GetHloModules is not supported.");
  }

  absl::StatusOr<std::vector<std::vector<std::string_view>>>
  GetOutputMemoryKinds() const override {
    if (output_memory_kinds_.has_value()) {
      return *output_memory_kinds_;
    }
    return absl::UnimplementedError("GetOutputMemoryKinds is not supported.");
  }
  absl::StatusOr<absl::flat_hash_map<std::string, PjRtValueType>>
  GetCostAnalysis() const override {
    return absl::UnimplementedError("GetCostAnalysis is not supported.");
  }

  int64_t SizeOfGeneratedCodeInBytes() const override { return 0; }

  const CompileOptions& compile_options() const { return compile_options_; }
  std::vector<std::unique_ptr<AotCompilationResult>>& aot_executables() {
    return aot_executables_;
  }

  absl::StatusOr<std::string> FingerprintExecutable() const override {
    return fingerprint_;
  }

 private:
  CompileOptions compile_options_;
  std::vector<std::unique_ptr<AotCompilationResult>> aot_executables_;
  int num_replicas_;
  int num_partitions_;
  std::string name_;
  std::string fingerprint_;
  std::optional<std::vector<std::vector<std::string_view>>>
      output_memory_kinds_;
};

}  // namespace zkx

#endif  // ZKX_PJRT_STREAM_EXECUTOR_EXECUTABLE_H_
