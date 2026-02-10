/* Copyright 2020 The OpenXLA Authors.
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

#include "zkx/service/gpu/transforms/horizontal_input_fusion.h"

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/log.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/service/gpu/gpu_fusible.h"
#include "zkx/service/hlo_creation_utils.h"
#include "zkx/shape.h"
#include "zkx/shape_util.h"
#include "zkx/util.h"

namespace zkx::gpu {

namespace {

// Gets the representative input shape of the multi-output fusion.
Shape GetInputShapeForMultiOutputFusion(
    const HloInstruction& instr, const se::DeviceDescription& device_info) {
  // Get the HLO that determines the emitter used for lowering.
  const HloInstruction* real_hero =
      GetRealHeroForMultiOutputFusion(instr, device_info);
  if (real_hero->operands().empty()) {
    // Simply return an empty shape if the representative node has no input
    // operands.
    return Shape();
  } else {
    return real_hero->operand(0)->shape();
  }
}

class HorizontalInputFusionImpl {
 public:
  explicit HorizontalInputFusionImpl(HloComputation* computation,
                                     const se::DeviceDescription& d)
      : computation_(computation), device_info_(d) {}

  ~HorizontalInputFusionImpl() = default;

  absl::StatusOr<bool> Run();

 private:
  HloComputation* computation_;
  const se::DeviceDescription& device_info_;
};

std::vector<HloInstruction*> FindAndSortFusionCandidates(
    HloInstruction* consumer, const se::DeviceDescription& device_info) {
  absl::flat_hash_set<HloInstruction*> fusion_instr_set;
  std::vector<HloInstruction*> fusion_instrs;
  for (HloInstruction* opnd : consumer->operands()) {
    HloInstruction* predecessor = opnd->LatestNonGteAncestor();
    // Find out the input fusion instructions whose only consumer is `consumer`.
    // This guarantees that fusing these candidates will never create cycles, as
    // there is no back edge.
    if (!predecessor->IsCustomFusion() &&
        IsInputFusibleReduction(*predecessor, device_info) &&
        IsConsumerTheOnlyNonRootUser(*predecessor, *consumer)) {
      if (fusion_instr_set.insert(predecessor).second) {
        fusion_instrs.push_back(predecessor);
      }
    }
  }

  std::sort(
      fusion_instrs.begin(), fusion_instrs.end(),
      [&](const HloInstruction* a, const HloInstruction* b) {
        Shape shape_a = GetInputShapeForMultiOutputFusion(*a, device_info);
        Shape shape_b = GetInputShapeForMultiOutputFusion(*b, device_info);
        auto tuple_for_op = [](const Shape& shape, const HloInstruction* op) {
          // Sort shapes according to dimensions, so that the same input
          // shapes will be placed adjacent each other.
          // Sort `fusion_instrs` according to instruction counts, because
          // we'd like to fuse together computations of similar sizes.
          return std::tuple{shape.rank(), shape.dimensions(),
                            GetInstrCountOfFusible(*op), op->unique_id()};
        };
        return tuple_for_op(shape_a, a) < tuple_for_op(shape_b, b);
      });

  return fusion_instrs;
}

absl::StatusOr<bool> HorizontalInputFusionImpl::Run() {
  bool changed = false;
  ZKX_VLOG_LINES(3, computation_->ToString());

  // Using def-to-use order is sound since we do not modify users.
  std::vector<HloInstruction*> def_to_use_order =
      computation_->MakeInstructionPostOrder();
  for (HloInstruction* consumer : def_to_use_order) {
    auto candidates = FindAndSortFusionCandidates(consumer, device_info_);
    if (candidates.size() <= 1) {
      continue;
    }

    // Convert candidates into fusions if needed.
    for (size_t j = 0; j < candidates.size(); ++j) {
      if (candidates[j]->opcode() != HloOpcode::kFusion) {
        TF_ASSIGN_OR_RETURN(
            HloInstruction * fusion_instr,
            MakeFusionInstruction(candidates[j],
                                  HloInstruction::FusionKind::kInput));
        candidates[j] = fusion_instr;
        changed = true;
      }
    }

    size_t fusion_anchor_id = 0;
    for (size_t j = 1; j < candidates.size(); ++j) {
      HloInstruction* fusion_anchor = candidates[fusion_anchor_id];
      HloInstruction* fused = candidates[j];
      if (ShapesCompatibleForMultiOutputFusion(*fusion_anchor, *fused,
                                               device_info_) &&
          FusionFitsInBudget(*fusion_anchor, *fused, device_info_)) {
        VLOG(3) << "Fuse " << fused->ToString() << " into "
                << fusion_anchor->ToString();
        fusion_anchor->MergeFusionInstructionIntoMultiOutput(fused);
        changed = true;
      } else {
        // Update the `fusion_anchor_id` since `fused` is either not
        // compatible or not beneficial to be fused with current fusion anchor.
        VLOG(3) << j - fusion_anchor_id - 1 << " instructions are fused.";
        fusion_anchor_id = j;
      }
    }
  }

  return changed;
}

}  // namespace

absl::StatusOr<bool> HorizontalInputFusion::RunOnComputation(
    HloComputation* computation) {
  HorizontalInputFusionImpl horizontal_fusion_impl(computation, device_info_);
  return horizontal_fusion_impl.Run();
}

absl::StatusOr<bool> HorizontalInputFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<std::string_view>& execution_threads) {
  bool any_changed = false;
  VLOG(2) << "Run horizontal input fusion.";
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool changed, RunOnComputation(comp));
    any_changed |= changed;
  }

  return any_changed;
}

}  // namespace zkx::gpu
