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
#ifndef ZKX_BACKENDS_GPU_CODEGEN_COPY_H_
#define ZKX_BACKENDS_GPU_CODEGEN_COPY_H_

#include <vector>

#include "absl/status/statusor.h"

#include "zkx/backends/gpu/codegen/fusion_emitter.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/service/buffer_assignment.h"
#include "zkx/service/gpu/hlo_fusion_analysis.h"
#include "zkx/service/gpu/ir_emitter_context.h"

namespace zkx::gpu {

// Special case of a fusion consisting only of `kCopy` instructions that can be
// implemented using `memcpy`s.
class MemcpyFusion : public FusionInterface {
 public:
  MemcpyFusion(const HloFusionAnalysis& analysis,
               const BufferAssignment* buffer_assignment)
      : analysis_(analysis), buffer_assignment_(buffer_assignment) {}

  absl::StatusOr<FusionEmissionResult> Emit(
      IrEmitterContext& ir_emitter_context,
      const HloFusionInstruction& fusion) const final;

 private:
  const HloFusionAnalysis& analysis_;
  const BufferAssignment* buffer_assignment_;  // not owned
};

}  // namespace zkx::gpu

#endif  // ZKX_BACKENDS_GPU_CODEGEN_COPY_H_
