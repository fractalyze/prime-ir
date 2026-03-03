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

#include "zkx/service/gpu/transforms/algebraic_simplifier.h"

#include "absl/log/check.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/service/gpu/matmul_utils.h"

namespace zkx::gpu {

bool GpuAlgebraicSimplifierVisitor::ShouldStrengthReduceDotToReduce(
    const HloInstruction* hlo) {
  if (!options_.enable_dot_strength_reduction()) {
    return false;
  }

  const HloDotInstruction* dot = DynCast<HloDotInstruction>(hlo);
  if (dot == nullptr) {
    return false;
  }

  const HloInstruction* lhs = dot->operand(0);
  const HloInstruction* rhs = dot->operand(1);
  DotDimensionNumbers dnums = dot->dot_dimension_numbers();
  bool lhs_is_vector = (dnums.lhs_batch_dimensions_size() +
                            dnums.lhs_contracting_dimensions_size() ==
                        lhs->shape().rank());
  bool rhs_is_vector = (dnums.rhs_batch_dimensions_size() +
                            dnums.rhs_contracting_dimensions_size() ==
                        rhs->shape().rank());
  // Strength-reduce vector-vector dots since they are not supported by
  // GemmFusion.
  if (lhs_is_vector && rhs_is_vector) {
    return true;
  }

  absl::StatusOr<bool> is_too_small =
      IsMatrixMultiplicationTooSmallForRewriting(*hlo, /*threshold=*/10000000);
  CHECK_OK(is_too_small.status());
  return is_too_small.value();
}

}  // namespace zkx::gpu
