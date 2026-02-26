/* Copyright 2017 The OpenXLA Authors.
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

#include "zkx/hlo/transforms/simplifiers/zero_sized_hlo_elimination.h"

#include <memory>
#include <vector>

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "zkx/literal_util.h"
#include "zkx/shape.h"
#include "zkx/shape_util.h"

namespace zkx {
namespace {

class ZeroSizedHloEliminationTest : public HloHardwareIndependentTestBase {
 protected:
  ZeroSizedHloEliminationTest()
      : HloHardwareIndependentTestBase(),
        builder_("zero_sized_computation"),
        zero_sized_param_(
            builder_.AddInstruction(HloInstruction::CreateParameter(
                0, ShapeUtil::MakeShape(S32, {3, 0}), "zero sized param"))) {}

  absl::StatusOr<bool> RunZeroSizedElimination() {
    auto module = CreateNewVerifiedModule("zero_sized_elimination_test_module");
    module->AddEntryComputation(builder_.Build());
    return ZeroSizedHloElimination{}.Run(module.get());
  }

  HloComputation::Builder builder_;
  HloInstruction* zero_sized_param_;
};

TEST_F(ZeroSizedHloEliminationTest, EliminatedZeroSizedOp) {
  builder_.AddInstruction(HloInstruction::CreateUnary(
      zero_sized_param_->shape(), HloOpcode::kNegate, zero_sized_param_));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunZeroSizedElimination());
  EXPECT_TRUE(changed);
}

TEST_F(ZeroSizedHloEliminationTest, DoesNotEliminateParameter) {
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunZeroSizedElimination());
  EXPECT_FALSE(changed);
}

TEST_F(ZeroSizedHloEliminationTest, DoesNotEliminateSideEffects) {
  auto token = builder_.AddInstruction(HloInstruction::CreateToken());
  auto send = builder_.AddInstruction(HloInstruction::CreateSend(
      zero_sized_param_, token, /*channel_id*/ 0, /*is_host_transfer=*/false));
  builder_.AddInstruction(HloInstruction::CreateSendDone(
      send, send->channel_id(), /*is_host_transfer=*/false));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunZeroSizedElimination());
  EXPECT_FALSE(changed);
}

TEST_F(ZeroSizedHloEliminationTest, DoesNotEliminateConstant) {
  builder_.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1({})));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunZeroSizedElimination());
  EXPECT_FALSE(changed);
}

TEST_F(ZeroSizedHloEliminationTest, ZeroSizedInstructionWithoutLayoutFolded) {
  Shape op_shape = ShapeUtil::MakeShape(S32, {4, 0});
  op_shape.clear_layout();
  HloInstruction* param1 = builder_.AddInstruction(
      HloInstruction::CreateParameter(1, op_shape, "zero sized param 1"));
  HloInstruction* param2 = builder_.AddInstruction(
      HloInstruction::CreateParameter(2, op_shape, "zero sized param 2"));
  builder_.AddInstruction(
      HloInstruction::CreateBinary(op_shape, HloOpcode::kAdd, param1, param2));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunZeroSizedElimination());
  EXPECT_TRUE(changed);
}

}  // namespace
}  // namespace zkx
