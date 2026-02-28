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

#ifndef ZKX_PJRT_GPU_SE_GPU_PJRT_COMPILER_H_
#define ZKX_PJRT_GPU_SE_GPU_PJRT_COMPILER_H_

#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"

#include "zkx/hlo/builder/zkx_computation.h"
#include "zkx/pjrt/pjrt_compiler.h"
#include "zkx/pjrt/pjrt_executable.h"
#include "zkx/stream_executor/platform.h"

namespace zkx {

// Implements the interfaces that are needed for the registered compiler.
class StreamExecutorGpuCompiler : public PjRtCompiler {
 public:
  // Constructs a compiler for the default "gpu" platform.
  explicit StreamExecutorGpuCompiler() = default;

  // Constructs a compiler for the given platform.
  explicit StreamExecutorGpuCompiler(se::Platform::Id platform_id);

  // Setting CompileOptions.TargetConfig field will trigger deviceless
  // compilation, which will not query the GPU attached to the machine.
  // In this case, the `client` argument could be left as `nullptr`.
  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, const ZkxComputation& computation,
      const PjRtTopologyDescription& topology, PjRtClient* client) override;

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, mlir::ModuleOp module,
      const PjRtTopologyDescription& topology, PjRtClient* client) override;

 private:
  std::optional<se::Platform::Id> requested_platform_id_;
};

}  // namespace zkx

#endif  // ZKX_PJRT_GPU_SE_GPU_PJRT_COMPILER_H_
