/* Copyright 2024 The OpenXLA Authors.
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
#ifndef ZKX_BACKENDS_GPU_CODEGEN_TOOLS_TEST_LIB_H_
#define ZKX_BACKENDS_GPU_CODEGEN_TOOLS_TEST_LIB_H_

#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "mlir/IR/MLIRContext.h"

#include "zkx/backends/gpu/codegen/emitters/emitter_base.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/service/gpu/hlo_fusion_analysis.h"
#include "zkx/stream_executor/device_description.h"

namespace zkx::gpu {

// Loads a test module from the given filename, ensuring it has a single fusion.
// If the file contains more than one fusion, the function fails. If the file
// contains no fusions, the function generates a fusion from the entry
// computation.
absl::StatusOr<std::unique_ptr<HloModule>> LoadTestModule(
    std::string_view filename);

// Returns the MLIR fusion emitter for the given module, which should have been
// loaded using LoadTestModule.
struct EmitterData {
  HloFusionInstruction* fusion;
  std::optional<se::DeviceDescription> device;
  std::optional<HloFusionAnalysis> analysis;
  std::unique_ptr<EmitterBase> emitter;
};
absl::StatusOr<std::unique_ptr<EmitterData>> GetEmitter(
    const HloModule& module);

// Returns an MLIR context with all the dialects needed for testing.
mlir::MLIRContext GetMlirContextForTest();

}  // namespace zkx::gpu

#endif  // ZKX_BACKENDS_GPU_CODEGEN_TOOLS_TEST_LIB_H_
