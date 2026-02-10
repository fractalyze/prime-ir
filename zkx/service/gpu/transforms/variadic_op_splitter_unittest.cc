/* Copyright 2018 The OpenXLA Authors.
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

#include "zkx/service/gpu/transforms/variadic_op_splitter.h"

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/parser/hlo_parser.h"
#include "zkx/literal_util.h"
#include "zkx/service/pattern_matcher.h"
#include "zkx/shape_util.h"
#include "zkx/tests/hlo_test_base.h"
#include "zkx/util.h"

namespace zkx::gpu {
namespace {
using match::Concatenate;

class VariadicOpSplitterTest : public HloTestBase {};

TEST_F(VariadicOpSplitterTest, DontSplit) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    p0 = s16[30,41] parameter(0)
    p1 = s16[30,41] parameter(1)
    ROOT result = s16[60, 41] concatenate(p0, p1), dimensions={0}
  })")
                    .value();
  EXPECT_FALSE(VariadicOpSplitter().Run(module.get()).value());
}

TEST_F(VariadicOpSplitterTest, SplitInto2) {
  auto builder = HloComputation::Builder(TestName());
  auto operand = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int32_t>({42})));
  std::vector<HloInstruction*> concat_operands(255, operand);
  builder.AddInstruction(HloInstruction::CreateConcatenate(
      ShapeUtil::MakeShape(S32, {255}), concat_operands, 0));
  auto module = CreateNewVerifiedModule();
  auto entry_computation = module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(VariadicOpSplitter().Run(module.get()).value());
  EXPECT_TRUE(Match(entry_computation->root_instruction(),
                    Concatenate().WithNumOperands(128).WithOperand(
                        0, Concatenate().WithNumOperands(128))));
}

TEST_F(VariadicOpSplitterTest, SplitInto3) {
  auto builder = HloComputation::Builder(TestName());
  auto operand = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int32_t>({42})));
  std::vector<HloInstruction*> concat_operands(256, operand);
  builder.AddInstruction(HloInstruction::CreateConcatenate(
      ShapeUtil::MakeShape(S32, {256}), concat_operands, 0));
  auto module = CreateNewVerifiedModule();
  auto entry_computation = module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(VariadicOpSplitter().Run(module.get()).value());
  EXPECT_TRUE(Match(entry_computation->root_instruction(),
                    Concatenate(Concatenate().WithNumOperands(128),
                                Concatenate().WithNumOperands(128))));
}

}  // namespace
}  // namespace zkx::gpu
