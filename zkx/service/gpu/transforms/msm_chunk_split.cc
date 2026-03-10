/* Copyright 2026 The ZKX Authors.

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

#include "zkx/service/gpu/transforms/msm_chunk_split.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "absl/log/log.h"

#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/primitive_util.h"
#include "zkx/shape_util.h"

namespace zkx::gpu {

// static
int MsmChunkSplit::GetOptimalC(int64_t msm_size) {
  // Mirrors ICICLE's get_optimal_c() from cuda_msm.cuh.
  if (msm_size <= 1) return 1;
  double c = std::ceil(std::log2(static_cast<double>(msm_size))) - 4.0;
  return static_cast<int>(std::min(std::max(c, 1.0), 20.0));
}

// static
int64_t MsmChunkSplit::EstimateMsmMemoryBytes(int64_t msm_size,
                                              int64_t scalar_bytes,
                                              int64_t base_bytes,
                                              int64_t projective_bytes,
                                              int32_t precompute_factor) {
  // Mirrors ICICLE's compute_required_memory() from cuda_msm.cuh.
  int c = GetOptimalC(msm_size);
  int nof_bms = (kBitsize + c - 1) / c;
  int pf = std::max(precompute_factor, static_cast<int32_t>(1));
  int nof_bms_after_precomp = (nof_bms + pf - 1) / pf;

  // Per-element costs (scale with N):
  int64_t scalars_mem = scalar_bytes * msm_size;
  // 7 unsigned ints per element per bucket module (sorting indices + temp).
  int64_t indices_mem =
      static_cast<int64_t>(7) * sizeof(uint32_t) * msm_size * nof_bms;
  int64_t points_mem = base_bytes * msm_size * pf;

  // Fixed cost (independent of N, depends on c):
  // 4 projective points per bucket × 2^c buckets × nof_bms_after_precomp.
  int64_t buckets_mem = static_cast<int64_t>(4) * projective_bytes *
                        (int64_t{1} << c) * nof_bms_after_precomp;

  return scalars_mem + indices_mem + points_mem + buckets_mem;
}

int64_t MsmChunkSplit::MaxMsmSize(int64_t scalar_bytes, int64_t base_bytes,
                                  int64_t projective_bytes,
                                  int32_t precompute_factor) const {
  int64_t budget =
      static_cast<int64_t>(device_memory_bytes_ * memory_fraction_);

  // Binary search: find largest N where EstimateMsmMemoryBytes(N) <= budget.
  // We need binary search because c = f(N), making the formula non-linear.
  int64_t lo = 1, hi = kMaxChunkSize;
  while (lo < hi) {
    int64_t mid = lo + (hi - lo + 1) / 2;
    if (EstimateMsmMemoryBytes(mid, scalar_bytes, base_bytes, projective_bytes,
                               precompute_factor) <= budget) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

absl::StatusOr<bool> MsmChunkSplit::Run(
    HloModule* module,
    const absl::flat_hash_set<std::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation : module->MakeComputationPostOrder()) {
    if (computation->IsFusionComputation()) continue;

    // Collect MSMs to split (don't modify while iterating).
    std::vector<HloMsmInstruction*> to_split;
    for (HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() != HloOpcode::kMsm) continue;
      auto* msm = Cast<HloMsmInstruction>(instr);
      // Only split non-batched MSMs.
      if (msm->batch_size() > 1) continue;

      int64_t n = msm->operand(0)->shape().dimensions(0);
      int64_t scalar_bytes =
          primitive_util::ByteWidth(msm->operand(0)->shape().element_type());
      int64_t base_bytes =
          primitive_util::ByteWidth(msm->operand(1)->shape().element_type());
      // ICICLE uses projective (jacobian) points internally for buckets.
      PrimitiveType base_type = msm->operand(1)->shape().element_type();
      int64_t projective_bytes = primitive_util::ByteWidth(
          primitive_util::AffineToJacobianType(base_type));
      int64_t max_size = MaxMsmSize(scalar_bytes, base_bytes, projective_bytes,
                                    msm->precompute_factor());

      if (n > max_size) {
        to_split.push_back(msm);
      }
    }

    for (HloMsmInstruction* msm : to_split) {
      int64_t n = msm->operand(0)->shape().dimensions(0);
      int64_t scalar_bytes =
          primitive_util::ByteWidth(msm->operand(0)->shape().element_type());
      int64_t base_bytes =
          primitive_util::ByteWidth(msm->operand(1)->shape().element_type());
      PrimitiveType base_type = msm->operand(1)->shape().element_type();
      int64_t projective_bytes = primitive_util::ByteWidth(
          primitive_util::AffineToJacobianType(base_type));
      int64_t chunk_size = MaxMsmSize(
          scalar_bytes, base_bytes, projective_bytes, msm->precompute_factor());
      // Ensure at least 1 element per chunk.
      chunk_size = std::max(chunk_size, int64_t{1});
      int64_t num_chunks = (n + chunk_size - 1) / chunk_size;

      int64_t estimated_bytes =
          EstimateMsmMemoryBytes(n, scalar_bytes, base_bytes, projective_bytes,
                                 msm->precompute_factor());
      int64_t budget =
          static_cast<int64_t>(device_memory_bytes_ * memory_fraction_);
      VLOG(1) << "Splitting MSM of size " << n << " into " << num_chunks
              << " chunks of ≤" << chunk_size << " (estimated "
              << estimated_bytes / (1 << 20) << " MB, budget "
              << budget / (1 << 20) << " MB)";

      HloInstruction* scalars = msm->mutable_operand(0);
      HloInstruction* bases = msm->mutable_operand(1);
      PrimitiveType scalar_type = scalars->shape().element_type();
      PrimitiveType bases_type = bases->shape().element_type();

      // Create chunk MSMs and accumulate via EC point addition.
      HloInstruction* accumulator = nullptr;
      for (int64_t i = 0; i < num_chunks; ++i) {
        int64_t start = i * chunk_size;
        int64_t end = std::min(start + chunk_size, n);
        int64_t chunk_n = end - start;

        // Slice scalars[start:end].
        Shape scalar_slice_shape = ShapeUtil::MakeShape(scalar_type, {chunk_n});
        HloInstruction* scalar_slice = computation->AddInstruction(
            HloInstruction::CreateSlice(scalar_slice_shape, scalars,
                                        /*start_indices=*/{start},
                                        /*limit_indices=*/{end},
                                        /*strides=*/{1}));

        // Slice bases[start:end].
        Shape bases_slice_shape = ShapeUtil::MakeShape(bases_type, {chunk_n});
        HloInstruction* bases_slice = computation->AddInstruction(
            HloInstruction::CreateSlice(bases_slice_shape, bases,
                                        /*start_indices=*/{start},
                                        /*limit_indices=*/{end},
                                        /*strides=*/{1}));

        // Create chunk MSM with same config.
        Shape chunk_result_shape = msm->shape();  // scalar EC point
        HloInstruction* chunk_msm =
            computation->AddInstruction(HloInstruction::CreateMsm(
                chunk_result_shape, scalar_slice, bases_slice,
                msm->window_bits(), msm->precompute_factor(), msm->bitsize(),
                /*batch_size=*/0, /*are_points_shared=*/false));

        if (accumulator == nullptr) {
          accumulator = chunk_msm;
        } else {
          // EC point addition returns jacobian type in MLIR lowering.
          // Convert both operands to jacobian so the HLO binary op has
          // matching operand and result types.
          PrimitiveType result_type = chunk_result_shape.element_type();
          if (primitive_util::IsAffineEcPointType(result_type)) {
            PrimitiveType jac_type =
                primitive_util::AffineToJacobianType(result_type);
            Shape jac_shape = ShapeUtil::MakeShape(jac_type, {});
            if (primitive_util::IsAffineEcPointType(
                    accumulator->shape().element_type())) {
              accumulator = computation->AddInstruction(
                  HloInstruction::CreateConvert(jac_shape, accumulator));
            }
            chunk_msm = computation->AddInstruction(
                HloInstruction::CreateConvert(jac_shape, chunk_msm));
            accumulator =
                computation->AddInstruction(HloInstruction::CreateBinary(
                    jac_shape, HloOpcode::kAdd, accumulator, chunk_msm));
          } else {
            accumulator =
                computation->AddInstruction(HloInstruction::CreateBinary(
                    chunk_result_shape, HloOpcode::kAdd, accumulator,
                    chunk_msm));
          }
        }
      }

      // If the accumulator is in jacobian form (from EC addition), convert
      // back to the original affine type.
      if (accumulator->shape().element_type() != msm->shape().element_type()) {
        accumulator = computation->AddInstruction(
            HloInstruction::CreateConvert(msm->shape(), accumulator));
      }

      // Replace the original MSM with the accumulated result.
      TF_RETURN_IF_ERROR(computation->ReplaceInstruction(msm, accumulator));
      changed = true;
    }
  }

  return changed;
}

}  // namespace zkx::gpu
