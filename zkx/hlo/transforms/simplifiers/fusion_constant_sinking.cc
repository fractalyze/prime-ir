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

#include "zkx/hlo/transforms/simplifiers/fusion_constant_sinking.h"

#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/base/logging.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/hlo/transforms/simplifiers/hlo_dce.h"
#include "zkx/shape_util.h"
#include "zkx/util.h"

namespace zkx {

// Given the fusion instruction and the operand to the fusion, checks:
//   1. the operand is scalar and constant
//   2. the parameter instruction representing the operand is not used in any
//   fusion instructions with a single operand.
// if the checks hold, it returns the parameter instruction representing the
// operand in the fusion computation, otherwise nullopt.
bool CanSink(HloInstruction* fusion, const HloInstruction* operand) {
  if (!fusion->IsLoopFusion() && !fusion->IsOutputFusion()) {
    return false;
  }

  if (fusion->operand_count() == 1) {
    return false;
  }

  if (!ShapeUtil::IsScalar(operand->shape()) || !operand->IsConstant()) {
    return false;
  }

  int64_t operand_idx = fusion->operand_index(operand);
  HloInstruction* fused_param = fusion->fused_parameter(operand_idx);
  for (HloInstruction* user : fused_param->users()) {
    // Fusions with single operands are not considered because the nested
    // computation will be left without any parameters
    if (user->opcode() == HloOpcode::kFusion && user->operand_count() == 1) {
      return false;
    }
  }
  return true;
}

bool ProcessScalar(HloInstruction* scalar) {
  if (!ShapeUtil::IsScalar(scalar->shape()) || !scalar->IsConstant()) {
    return false;
  }
  bool processed = false;
  std::vector<HloInstruction*> sinkable_users;
  for (HloInstruction* use : scalar->users()) {
    if (CanSink(use, scalar)) {
      sinkable_users.push_back(use);
    }
  }
  for (HloInstruction* use : sinkable_users) {
    HloInstruction* fused_scalar = use->FuseInstruction(scalar);
    processed = true;
    ProcessScalar(fused_scalar);
  }
  return processed;
}

absl::StatusOr<bool> FusionConstantSinking::Run(
    HloModule* module,
    const absl::flat_hash_set<std::string_view>& execution_threads) {
  VLOG(3) << "HLO module before FusionConstantSinking:";
  ZKX_VLOG_LINES(3, module->ToString());

  bool changed = false;
  for (HloComputation* c : module->MakeNonfusionComputations()) {
    for (HloInstruction* i : c->MakeInstructionPostOrder()) {
      changed |= ProcessScalar(i);
    }
  }

  if (changed) {
    TF_ASSIGN_OR_RETURN(bool dce, HloDCE{}.Run(module, execution_threads));
    changed |= dce;
  }

  VLOG(3) << "HLO module after FusionConstantSinking:";
  ZKX_VLOG_LINES(3, module->ToString());
  return changed;
}

}  // namespace zkx
