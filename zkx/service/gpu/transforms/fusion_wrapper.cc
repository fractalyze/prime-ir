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

#include "zkx/service/gpu/transforms/fusion_wrapper.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

#include "xla/tsl/platform/errors.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/service/gpu/gpu_fusible.h"

namespace zkx::gpu {

// static
bool FusionWrapper::MustWrapInstruction(const HloInstruction& instruction) {
  const HloOpcode opcode = instruction.opcode();
  switch (opcode) {
    case HloOpcode::kAbs:
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kBroadcast:
    case HloOpcode::kClamp:
    case HloOpcode::kClz:
    case HloOpcode::kCompare:
    case HloOpcode::kConcatenate:
    case HloOpcode::kConvert:
    case HloOpcode::kDivide:
    case HloOpcode::kDot:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kGather:
    case HloOpcode::kIota:
    case HloOpcode::kMap:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kOr:
    case HloOpcode::kPad:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kPower:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kRemainder:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kScatter:
    case HloOpcode::kSelect:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kSign:
    case HloOpcode::kSlice:
    case HloOpcode::kSubtract:
    case HloOpcode::kTranspose:
    case HloOpcode::kXor:
      return true;
    default:
      return false;
  }
}
// NOTE: The following XLA opcodes (at 8bac4a2c) are omitted from
// MustWrapInstruction because they are not present in the ZKX HloOpcode enum:
// kAtan2, kCbrt, kCeil, kComplex, kConvolution, kCos, kErf, kExp, kExpm1,
// kFloor, kImag, kIsFinite, kLog, kLog1p, kReal, kReducePrecision,
// kRoundNearestAfz, kRoundNearestEven, kRsqrt, kSin, kSqrt,
// kStochasticConvert, kTan, kTanh.

HloInstruction::FusionKind FusionWrapper::ChooseFusionKind(
    const HloInstruction& producer, const HloInstruction& consumer) {
  return gpu::ChooseFusionKind(producer, consumer, device_description_);
}

// Run() logic is inlined from xla::emitters::FusionWrapperBase::RunImpl(),
// since FusionWrapperBase is not ported to ZKX.
absl::StatusOr<bool> FusionWrapper::Run(
    HloModule* module,
    const absl::flat_hash_set<std::string_view>& execution_threads) {
  bool changed = false;
  for (auto* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (!MustWrapInstruction(*instruction)) {
        continue;
      }
      auto* fusion_instruction =
          computation->AddInstruction(HloInstruction::CreateFusion(
              instruction->shape(),
              ChooseFusionKind(*instruction, *instruction), instruction));
      const std::string_view wrapped_opcode =
          HloOpcodeString(instruction->opcode());
      module->SetAndUniquifyInstrName(fusion_instruction,
                                      absl::StrCat("wrapped_", wrapped_opcode));
      module->SetAndUniquifyComputationName(
          fusion_instruction->fused_instructions_computation(),
          absl::StrCat("wrapped_", wrapped_opcode, "_computation"));
      if (module->has_schedule()) {
        module->schedule().replace_instruction(computation, instruction,
                                               fusion_instruction);
      }
      TF_RETURN_IF_ERROR(
          fusion_instruction->CopyAllControlDepsFrom(instruction));
      TF_RETURN_IF_ERROR(instruction->DropAllControlDeps());
      TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(fusion_instruction));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(instruction));
      changed = true;
    }
  }
  return changed;
}

}  // namespace zkx::gpu
