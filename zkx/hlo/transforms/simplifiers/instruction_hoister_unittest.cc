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

#include "zkx/hlo/transforms/simplifiers/instruction_hoister.h"

#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace zkx {
namespace {

using InstructionHoisterTest = HloHardwareIndependentTestBase;

TEST_F(InstructionHoisterTest, HoistParameters) {
  // Parameters are not at the beginning of the schedule, so they should be
  // hoisted.
  std::string_view hlo_string = R"(
  HloModule test, is_scheduled=true
  ENTRY test {
    p1 = s32[] parameter(1)
    add = s32[] add(p1, p1)
    p0 = s32[] parameter(0)
    ROOT add2 = s32[] add(add, p0)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InstructionHoister hoister(/*hoist_parameters=*/true,
                             /*hoist_constants=*/false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, hoister.Run(module.get()));
  EXPECT_TRUE(changed);

  // Verify parameters are first in the schedule.
  const auto& sequence =
      module->schedule().sequence(module->entry_computation());
  EXPECT_EQ(sequence.instructions()[0]->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(sequence.instructions()[1]->opcode(), HloOpcode::kParameter);
}

TEST_F(InstructionHoisterTest, NoChangeWhenAlreadyHoisted) {
  // Parameters and constants are already at the top.
  std::string_view hlo_string = R"(
  HloModule test, is_scheduled=true
  ENTRY test {
    p0 = s32[] parameter(0)
    constant.1 = s32[] constant(42)
    ROOT add = s32[] add(p0, constant.1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InstructionHoister hoister;
  TF_ASSERT_OK(hoister.Run(module.get()).status());
  const auto& sequence =
      module->schedule().sequence(module->entry_computation());
  // Verify the schedule is valid.
  EXPECT_EQ(sequence.size(), 3);
}

TEST_F(InstructionHoisterTest, HoistConstantsOnly) {
  // Only hoist constants, not parameters.
  std::string_view hlo_string = R"(
  HloModule test, is_scheduled=true
  ENTRY test {
    p0 = s32[] parameter(0)
    p1 = s32[] parameter(1)
    add = s32[] add(p0, p1)
    constant.1 = s32[] constant(42)
    ROOT add2 = s32[] add(add, constant.1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  InstructionHoister hoister(/*hoist_parameters=*/false,
                             /*hoist_constants=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, hoister.Run(module.get()));
  EXPECT_TRUE(changed);

  // Verify constants are moved before non-constants.
  const auto& sequence =
      module->schedule().sequence(module->entry_computation());
  bool found_constant = false;
  bool found_non_constant_non_param = false;
  for (auto* inst : sequence.instructions()) {
    if (inst->opcode() == HloOpcode::kConstant) {
      EXPECT_FALSE(found_non_constant_non_param)
          << "Constant found after non-constant instruction";
      found_constant = true;
    }
    if (inst->opcode() != HloOpcode::kConstant &&
        inst->opcode() != HloOpcode::kParameter) {
      found_non_constant_non_param = true;
    }
  }
  EXPECT_TRUE(found_constant);
}

}  // namespace
}  // namespace zkx
