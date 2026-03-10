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

#include "zkx/service/gpu/transforms/msm_batch_fusion.h"

#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"

#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/shape_util.h"

namespace zkx::gpu {
namespace {

// Key for grouping MSM instructions that can be batched together.
struct MsmGroupKey {
  const HloInstruction* bases;
  int32_t window_bits;
  int32_t precompute_factor;
  int32_t bitsize;

  template <typename H>
  friend H AbslHashValue(H h, const MsmGroupKey& key) {
    return H::combine(std::move(h), key.bases, key.window_bits,
                      key.precompute_factor, key.bitsize);
  }

  bool operator==(const MsmGroupKey& other) const {
    return bases == other.bases && window_bits == other.window_bits &&
           precompute_factor == other.precompute_factor &&
           bitsize == other.bitsize;
  }
};

// Returns true if there is a data dependency from `a` to `b` (i.e., `b`
// transitively depends on `a`).
bool HasDataDependency(const HloInstruction* a, const HloInstruction* b) {
  // Simple check: if b uses a's output directly or transitively.
  // We do a BFS from b backwards through operands.
  absl::flat_hash_set<const HloInstruction*> visited;
  std::vector<const HloInstruction*> worklist = {b};
  while (!worklist.empty()) {
    const HloInstruction* cur = worklist.back();
    worklist.pop_back();
    if (cur == a) return true;
    if (!visited.insert(cur).second) continue;
    for (const HloInstruction* op : cur->operands()) {
      worklist.push_back(op);
    }
  }
  return false;
}

// Returns true if any instruction in `group` has a data dependency on any
// other instruction in the group.
bool HasInternalDependency(const std::vector<const HloMsmInstruction*>& group) {
  for (size_t i = 0; i < group.size(); ++i) {
    for (size_t j = i + 1; j < group.size(); ++j) {
      if (HasDataDependency(group[i], group[j]) ||
          HasDataDependency(group[j], group[i])) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace

absl::StatusOr<bool> MsmBatchFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<std::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation : module->MakeComputationPostOrder()) {
    if (!computation->IsFusionComputation()) {
      // Group MSM instructions by (bases, config).
      absl::flat_hash_map<MsmGroupKey, std::vector<const HloMsmInstruction*>>
          groups;

      for (HloInstruction* instr : computation->instructions()) {
        if (instr->opcode() != HloOpcode::kMsm) continue;
        auto* msm = Cast<HloMsmInstruction>(instr);
        // Only batch non-batched MSMs (batch_size == 0 or 1).
        if (msm->batch_size() > 1) continue;

        MsmGroupKey key{msm->operand(1), msm->window_bits(),
                        msm->precompute_factor(), msm->bitsize()};
        groups[key].push_back(msm);
      }

      for (auto& [key, group] : groups) {
        if (group.size() < 2) continue;
        if (HasInternalDependency(group)) continue;

        int batch_size = group.size();
        VLOG(1) << "Fusing " << batch_size
                << " MSMs sharing bases=" << key.bases->name();

        // All scalars must have the same shape [N].
        const Shape& scalar_shape = group[0]->operand(0)->shape();
        int64_t n = scalar_shape.dimensions(0);
        PrimitiveType scalar_type = scalar_shape.element_type();

        // Verify all MSMs have matching scalar dimensions.
        bool compatible = true;
        for (const auto* msm : group) {
          const Shape& s = msm->operand(0)->shape();
          if (s.dimensions(0) != n || s.element_type() != scalar_type) {
            compatible = false;
            break;
          }
        }
        if (!compatible) continue;

        // Create concatenated scalars: [N] * B → [B*N].
        std::vector<HloInstruction*> scalar_operands;
        scalar_operands.reserve(batch_size);
        for (const auto* msm : group) {
          scalar_operands.push_back(
              const_cast<HloInstruction*>(msm->operand(0)));
        }

        Shape concat_shape =
            ShapeUtil::MakeShape(scalar_type, {batch_size * n});
        HloInstruction* concat_scalars =
            computation->AddInstruction(HloInstruction::CreateConcatenate(
                concat_shape, scalar_operands, /*dimension=*/0));

        // Result: [batch_size] points.
        PrimitiveType result_type = group[0]->shape().element_type();
        Shape batch_result_shape =
            ShapeUtil::MakeShape(result_type, {batch_size});

        // Create the batched MSM.
        HloInstruction* bases =
            const_cast<HloInstruction*>(group[0]->operand(1));
        auto batched_msm =
            computation->AddInstruction(HloInstruction::CreateMsm(
                batch_result_shape, concat_scalars, bases, key.window_bits,
                key.precompute_factor, key.bitsize, batch_size,
                /*are_points_shared=*/true));

        // Replace each original MSM with a slice of the batched result.
        for (int i = 0; i < batch_size; ++i) {
          Shape slice_shape = ShapeUtil::MakeShape(result_type, {});
          // Slice [i, i+1) then reshape to scalar.
          Shape slice_1d = ShapeUtil::MakeShape(result_type, {1});
          HloInstruction* slice =
              computation->AddInstruction(HloInstruction::CreateSlice(
                  slice_1d, batched_msm, /*start_indices=*/{i},
                  /*limit_indices=*/{i + 1}, /*strides=*/{1}));
          HloInstruction* reshape = computation->AddInstruction(
              HloInstruction::CreateReshape(slice_shape, slice));
          TF_RETURN_IF_ERROR(computation->ReplaceInstruction(
              const_cast<HloInstruction*>(
                  static_cast<const HloInstruction*>(group[i])),
              reshape));
        }

        changed = true;
      }
    }
  }

  return changed;
}

}  // namespace zkx::gpu
