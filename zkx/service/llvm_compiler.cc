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

#include "zkx/service/llvm_compiler.h"

#include <utility>

#include "absl/strings/str_format.h"

#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/lib/scoped_annotation.h"

namespace zkx {

absl::StatusOr<std::vector<std::unique_ptr<Executable>>> LlvmCompiler::Compile(
    std::unique_ptr<HloModuleGroup> module_group,
    std::vector<std::vector<se::StreamExecutor*>> executors,
    const CompileOptions& options) {
  std::vector<std::unique_ptr<Executable>> result;
  std::vector<std::unique_ptr<HloModule>> modules =
      module_group->ConsumeModules();
  for (size_t i = 0; i < modules.size(); i++) {
    tsl::profiler::ScopedAnnotation annotation{[&] {
      return absl::StrFormat("ZlxCompile:#module=%s,program_id=%d#",
                             modules[i]->name(), modules[i]->unique_id());
    }};
    TF_ASSIGN_OR_RETURN(modules[i], RunHloPasses(std::move(modules[i]),
                                                 executors[i][0], options));
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<Executable> executable,
        RunBackend(std::move(modules[i]), executors[i][0], options));
    result.push_back(std::move(executable));
  }

  return std::move(result);
}

}  // namespace zkx
