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

#include "zkx/hlo/transforms/simplifiers/hlo_cse.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/substitute.h"
#include "gmock/gmock.h"

#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/hlo/parser/hlo_parser.h"
#include "zkx/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "zkx/hlo/testlib/pattern_matcher_gmock.h"
#include "zkx/hlo/utils/hlo_matchers.h"
#include "zkx/layout_util.h"
#include "zkx/literal.h"
#include "zkx/literal_util.h"
#include "zkx/service/pattern_matcher.h"
#include "zkx/shape_util.h"
#include "zkx/zkx_data.pb.h"

namespace zkx {
namespace {

namespace op = zkx::testing::opcode_matchers;
namespace m = zkx::match;

class HloCseTest : public HloHardwareIndependentTestBase {
 protected:
  HloCseTest() {}
};

TEST_F(HloCseTest, CombineTwoConstants) {
  // Test that two identical constants are commoned.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(42)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(42)));
  builder.AddInstruction(HloInstruction::CreateBinary(
      constant1->shape(), HloOpcode::kAdd, constant1, constant2));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, computation->instruction_count());

  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_TRUE(cse.Run(module.get()).value());

  EXPECT_EQ(2, computation->instruction_count());
  HloInstruction* constant = *computation->instructions().begin();
  EXPECT_EQ(42, constant->literal().Get<int32_t>({}));
}

TEST_F(HloCseTest, CombineTwoConstantsDifferentLayouts) {
  // Test that two identical constants with different layouts are *not*
  // combined.
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2WithLayout<int32_t>(
          {{1, 2}, {3, 4}}, LayoutUtil::MakeLayout({0, 1}))));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2WithLayout<int32_t>(
          {{1, 2}, {3, 4}}, LayoutUtil::MakeLayout({1, 0}))));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      constant1->shape(), HloOpcode::kAdd, constant1, constant2));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, computation->instruction_count());
  EXPECT_THAT(add, op::Add(constant1, constant2));

  HloCSE cse(/*is_layout_sensitive=*/true);
  EXPECT_FALSE(cse.Run(module.get()).value());

  EXPECT_EQ(3, computation->instruction_count());
  EXPECT_THAT(add, op::Add(constant1, constant2));
}

TEST_F(HloCseTest, ConstantsSameValueDifferentType) {
  // Test that constants with the same value but different type are *not*
  // commoned.
  auto builder = HloComputation::Builder(TestName());
  std::vector<HloInstruction*> constants;
  constants.push_back(builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint32_t>(42))));
  constants.push_back(builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(42))));
  constants.push_back(builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint64_t>(42))));
  constants.push_back(builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(42))));
  // Duplicate the uint64 constant to verify something happens.
  constants.push_back(builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint64_t>(42))));

  const Shape shape_r0 = ShapeUtil::MakeShape(S32, {});
  for (int64_t i = 0; i < constants.size(); ++i) {
    constants[i] = builder.AddInstruction(
        HloInstruction::CreateConvert(shape_r0, constants[i]));
  }
  HloInstruction* root = builder.AddInstruction(HloInstruction::CreateBinary(
      shape_r0, HloOpcode::kAdd, constants[0], constants[1]));
  for (int64_t i = 2; i < constants.size(); ++i) {
    root = builder.AddInstruction(HloInstruction::CreateBinary(
        shape_r0, HloOpcode::kAdd, root, constants[i]));
  }

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  // 5 constants + 5 converts + 4 adds = 14
  EXPECT_EQ(14, computation->instruction_count());

  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_TRUE(cse.Run(module.get()).value());

  // CSE will remove the second uint64(42) and the corresponding
  // convert/cast: 14 - 2 = 12
  EXPECT_EQ(12, computation->instruction_count());
}

TEST_F(HloCseTest, NonscalarConstants) {
  // Test that identical nonscalar constants are merged.
  auto builder = HloComputation::Builder(TestName());
  auto common_constant1 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}})));
  auto common_constant2 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}})));
  // Create a constant which has the same shape but a different value.
  auto uncommon_constant =
      builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR2<int32_t>({{2, 4}, {6, 8}})));

  // Tie the constants together with a tuple. This makes it easier to refer to
  // the constant instructions via their use.
  auto tuple = builder.AddInstruction(HloInstruction::CreateTuple(
      {common_constant1, common_constant2, uncommon_constant}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(4, computation->instruction_count());
  EXPECT_THAT(tuple,
              op::Tuple(common_constant1, common_constant2, uncommon_constant));

  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_TRUE(cse.Run(module.get()).value());

  EXPECT_EQ(3, computation->instruction_count());
  auto first_operand = tuple->operand(0);
  EXPECT_THAT(first_operand,
              ::testing::AnyOf(common_constant1, common_constant2));
  EXPECT_THAT(tuple,
              op::Tuple(first_operand, first_operand, uncommon_constant));
}

TEST_F(HloCseTest, IdenticalInstructions) {
  // Test that three identical instructions are commoned.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(42)));
  auto abs1 = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kAbs, constant));
  auto abs2 = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kAbs, constant));
  auto abs3 = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kAbs, constant));
  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({abs1, abs2, abs3}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(5, computation->instruction_count());
  EXPECT_THAT(tuple, op::Tuple(abs1, abs2, abs3));

  HloCSE cse(/*is_layout_sensitive=*/true);
  EXPECT_TRUE(cse.Run(module.get()).value());

  EXPECT_EQ(3, computation->instruction_count());
  auto first_operand = tuple->operand(0);
  EXPECT_THAT(first_operand, ::testing::AnyOf(abs1, abs2, abs3));
  EXPECT_THAT(tuple, op::Tuple(first_operand, first_operand, first_operand));
}

// Test two identical while loops with same inputs
TEST_F(HloCseTest, WhileLoopsIdenticalConditionsAndBodiesSameInput) {
  const char* const hlo_string = R"(
    HloModule WhileLoopsIdenticalConditionsAndBodiesSameInput

    %body (param: (s32[], s32[])) -> (s32[], s32[]) {
      %param = (s32[], s32[]) parameter(0)
      %gte0 = get-tuple-element(%param), index=0
      %gte1 = get-tuple-element(%param), index=1
      %add = add(%gte0, %gte1)
      ROOT %tuple = tuple(%gte0, %add)
    }

    %condition {
      %param.1 = (s32[], s32[]) parameter(0)
      ROOT %constant = pred[] constant(false)
    }

    %condition.1 {
      %param.2 = (s32[], s32[]) parameter(0)
      ROOT %constant.1 = pred[] constant(false)
    }

    ENTRY %WhileLoopsIdenticalConditionsAndBodiesSameInput {
      %c0 = s32[] constant(1)
      %c1 = s32[] constant(2)
      %t = tuple(c0, c1)
      %while = while(%t), condition=%condition, body=%body
      %while.1 = while(%t), condition=%condition.1, body=%body
      ROOT r = tuple(while, while.1)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  auto computation = m->entry_computation();

  EXPECT_EQ(6, computation->instruction_count());
  HloCSE cse(true);
  EXPECT_TRUE(cse.Run(m.get()).value());
  EXPECT_EQ(5, computation->instruction_count());
}

// Test two while loops with same conditions, same inputs, but different
// bodies
TEST_F(HloCseTest, WhileLoopsIdenticalConditionsSameInputAndDifferentBodies) {
  const char* const hlo_string = R"(
    HloModule WhileLoopsIdenticalConditionsSameInputAndDifferentBodies

    %body {
      %param = (s32[], s32[]) parameter(0)
      %get-tuple-element = get-tuple-element(%param), index=0
      %get-tuple-element.1 = get-tuple-element(%param), index=1
      %add = add(%get-tuple-element, %get-tuple-element.1)
      ROOT %tuple = tuple(%get-tuple-element, %add)
    }

    %body2 {
      %param.1 = (s32[], s32[]) parameter(0)
      %get-tuple-element.2 = get-tuple-element(%param.1), index=0
      %get-tuple-element.3 = get-tuple-element(%param.1), index=1
      %sub = subtract(%get-tuple-element.2, %get-tuple-element.3)
      ROOT %tuple.2 = tuple(%get-tuple-element.2, %sub)
    }

    %condition (param.2: (s32[], s32[])) -> pred[] {
      %param.2 = (s32[], s32[]) parameter(0)
      ROOT %constant = pred[] constant(false)
    }

    %condition.1 (param.3: (s32[], s32[])) -> pred[] {
      %param.3 = (s32[], s32[]) parameter(0)
      ROOT %constant.1 = pred[] constant(false)
    }

    ENTRY %WhileLoopsIdenticalConditionsSameInputAndDifferentBodies {
      %constant.2 = s32[] constant(1)
      %constant.3 = s32[] constant(2)
      %tuple.1 = tuple(s32[] %constant.2, s32[] %constant.3)
      %while = while(%tuple.1), condition=%condition, body=%body
      ROOT %while.1 = while(%tuple.1), condition=%condition.1, body=%body2
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  auto computation = m->entry_computation();

  EXPECT_EQ(5, computation->instruction_count());
  HloCSE cse(true);
  EXPECT_FALSE(cse.Run(m.get()).value());
  EXPECT_EQ(5, computation->instruction_count());
}

// Test two while loops with identical bodies and same inputs, but different
// conditions
TEST_F(HloCseTest, WhileLoopsIdenticalBodiesAndInputDifferentConditions) {
  const char* const hlo_string = R"(
    HloModule WhileLoopsIdenticalBodiesAndInputDifferentConditions

    %body {
      %param = (s32[], s32[]) parameter(0)
      %get-tuple-element = get-tuple-element(%param), index=0
      %get-tuple-element.1 = get-tuple-element((s32[], s32[]) %param), index=1
      %add = add(%get-tuple-element, %get-tuple-element.1)
      ROOT %tuple = tuple(%get-tuple-element, %add)
    }

    %condition {
      %param.1 = (s32[], s32[]) parameter(0)
      ROOT %constant = pred[] constant(false)
    }

    %condition.1 {
      %param.2 = (s32[], s32[]) parameter(0)
      ROOT %constant.1 = pred[] constant(true)
    }

    ENTRY %WhileLoopsIdenticalBodiesAndInputDifferentConditions {
      %constant.2 = s32[] constant(1)
      %constant.3 = s32[] constant(2)
      %tuple.1 = tuple(%constant.2, %constant.3)
      %while = while(%tuple.1), condition=%condition, body=%body
      ROOT %while.1 = while(%tuple.1), condition=%condition.1, body=%body
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  auto computation = m->entry_computation();

  EXPECT_EQ(5, computation->instruction_count());
  HloCSE cse(true);
  EXPECT_FALSE(cse.Run(m.get()).value());
  EXPECT_EQ(5, computation->instruction_count());
}

TEST_F(HloCseTest, IdenticalInstructionsDifferentLayoutsSensitive) {
  // Test that two identical instructions with different layouts are *not*
  // commoned if the pass is layout sensitive.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}})));

  auto abs1 = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kAbs, constant));
  *abs1->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  auto abs2 = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kAbs, constant));
  *abs2->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({1, 0});

  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({abs1, abs2}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(4, computation->instruction_count());
  EXPECT_THAT(tuple, op::Tuple(abs1, abs2));

  HloCSE cse(/*is_layout_sensitive=*/true);
  EXPECT_FALSE(cse.Run(module.get()).value());

  EXPECT_EQ(4, computation->instruction_count());
  EXPECT_THAT(tuple, op::Tuple(abs1, abs2));
}

TEST_F(HloCseTest, IdenticalInstructionsDifferentLayoutsInsensitive) {
  // Test that two identical instructions with different layouts are commoned if
  // the pass is layout insensitive.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}})));

  auto abs1 = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kAbs, constant));
  *abs1->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  auto abs2 = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kAbs, constant));
  *abs2->mutable_shape()->mutable_layout() = LayoutUtil::MakeLayout({1, 0});

  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({abs1, abs2}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(4, computation->instruction_count());
  EXPECT_THAT(tuple, op::Tuple(abs1, abs2));

  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_TRUE(cse.Run(module.get()).value());

  EXPECT_EQ(3, computation->instruction_count());
  auto first_operand = tuple->operand(0);
  EXPECT_THAT(first_operand, ::testing::AnyOf(abs1, abs2));
  EXPECT_THAT(tuple, op::Tuple(first_operand, first_operand));
}

TEST_F(HloCseTest, FusionInternalCSE) {
  // Test that we can CSE expressions that live within a fusion node
  // computation.
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  const Shape shape_r0 = ShapeUtil::MakeShape(S32, {});
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape_r0, "p0"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape_r0, "p1"));
  auto add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_r0, HloOpcode::kAdd, param0, param1));
  auto add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_r0, HloOpcode::kAdd, param0, param1));
  auto mul = builder.AddInstruction(
      HloInstruction::CreateBinary(shape_r0, HloOpcode::kMultiply, add1, add2));

  auto computation = module->AddEntryComputation(builder.Build());
  auto fused_computation =
      computation
          ->CreateFusionInstruction({mul, add1, add2},
                                    HloInstruction::FusionKind::kLoop)
          ->fused_instructions_computation();

  EXPECT_EQ(5, fused_computation->instruction_count());
  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_TRUE(cse.Run(module.get()).value());
  EXPECT_EQ(4, fused_computation->instruction_count());

  auto root = fused_computation->root_instruction();
  EXPECT_THAT(root, op::Multiply(root->operand(0), root->operand(0)));
}

TEST_F(HloCseTest, IdenticalExpressions) {
  // Test that two identical expressions are commoned. Build the following
  // computation:
  //
  //   constant = 42
  //   negate1 = neg(constant)
  //   abs1 = abs(constant)
  //   add1 = add(negate1, abs1)
  //   negate2 = neg(constant)
  //   abs2 = abs(constant)
  //   add2 = add(negate2, abs2)
  //   tuple = tuple(add1, add2)
  //
  // The *1 instructions should be merged with the *2 instructions.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(42)));

  auto negate1 = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kNegate, constant));
  auto abs1 = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kAbs, constant));
  auto add1 = builder.AddInstruction(HloInstruction::CreateBinary(
      constant->shape(), HloOpcode::kAdd, negate1, abs1));

  auto negate2 = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kNegate, constant));
  auto abs2 = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kAbs, constant));
  auto add2 = builder.AddInstruction(HloInstruction::CreateBinary(
      constant->shape(), HloOpcode::kAdd, negate2, abs2));

  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({add1, add2}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_EQ(8, computation->instruction_count());
  EXPECT_THAT(tuple, op::Tuple(op::Add(negate1, abs1), op::Add(negate2, abs2)));

  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_TRUE(cse.Run(module.get()).value());

  EXPECT_EQ(5, computation->instruction_count());
  auto operand = tuple->operand(0);
  EXPECT_THAT(tuple, op::Tuple(operand, operand));
  EXPECT_THAT(operand, op::Add(op::Negate(), op::Abs()));
}

// TODO(jeong0982): Enable DoNotCombineRng test once kRng opcode and
// HloInstruction::CreateRng are ported.

// TODO(jeong0982): Enable DoNotCombineOpsWithDifferentShardings test once
// HloInstruction::CreateCustomCall is ported.

// TODO(jeong0982): Enable DoNotCombineCallsToImpureFunctions test once kRng
// opcode and HloInstruction::CreateRng are ported.

TEST_F(HloCseTest, CompareComputations) {
  const char* const hlo_string = R"(
    HloModule m

    add_computation {
      add_lhs = s32[] parameter(0)
      add_rhs = s32[] parameter(1)
      ROOT add_root = add(add_lhs, add_rhs)
    }

    add_computation2 {
      add_lhs2 = s32[] parameter(0)
      add_rhs2 = s32[] parameter(1)
      ROOT add_root2 = add(add_lhs2, add_rhs2)
    }

    ENTRY entry {
      p = s32[10]{0} parameter(0)
      c = s32[] constant(0)
      r1 = reduce(p, c), dimensions={0}, to_apply=add_computation
      r2 = reduce(p, c), dimensions={0}, to_apply=add_computation2
      ROOT f2 = tuple(r1, r2)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  HloCSE cse(/*is_layout_sensitive=*/false);
  EXPECT_TRUE(cse.Run(m.get()).value());
  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand(0), root->operand(1));
}

// TODO(jeong0982): Enable Domain test once HloDomainInstruction and
// HloInstruction::CreateDomain are ported.
// TEST_F(HloCseTest, Domain) { ... }

TEST_F(HloCseTest, Iota) {
  const char* const hlo_string = R"(
    HloModule m

    ENTRY entry {
      i1 = s64[16,16] iota(), iota_dimension=0
      i2 = s64[16,16] iota(), iota_dimension=0
      i3 = s64[17,16] iota(), iota_dimension=0
      i4 = s64[16,16] iota(), iota_dimension=1
      ROOT root = (s64[16,16], s64[16,16], s64[17,16], s64[16,16]) tuple(i1, i2, i3, i4)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  HloCSE cse(/*is_layout_sensitive=*/false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&cse, m.get()));
  EXPECT_TRUE(changed);
  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand(0), root->operand(1));
  EXPECT_NE(root->operand(0), root->operand(2));
  EXPECT_NE(root->operand(0), root->operand(3));
}

TEST_F(HloCseTest, OptimizationBarrier) {
  const char* const hlo_string = R"(
    HloModule m

    ENTRY entry {
      %param.0 = s32[] parameter(0)
      %param.1 = s32[] parameter(1)
      %add.0 = s32[] add(%param.0, %param.1)
      %cse_tmp.0 = (s32[], s32[], s32[]) tuple(%param.0, %param.1, %add.0)
      %cse_tmp.1 = (s32[], s32[], s32[]) opt-barrier(%cse_tmp.0)

      %param.0.1 = s32[] get-tuple-element(%cse_tmp.1), index=0
      %param.1.1 = s32[] get-tuple-element(%cse_tmp.1), index=1
      %add.0.1 = s32[] get-tuple-element(%cse_tmp.1), index=2

      %add.1 = s32[] add(%param.0.1, %param.1.1)
      ROOT %add.2 = s32[] add(%add.1, %add.0.1)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  HloCSE cse(/*is_layout_sensitive=*/false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&cse, m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(HloCseTest, OnlyScalar) {
  const char* const hlo_string = R"(
    HloModule m

    ENTRY entry {
      %const1 = s32[] constant(1)
      %const2 = s32[] constant(1)
      %const3 = s32[2] constant({1,2})
      %const4 = s32[2] constant({1,2})
      %add.0 = s32[] add(%const1, %const2)
      %add.1 = s32[2] add(%const3, %const4)
      ROOT out = (s32[], s32[2]) tuple(%add.0, %add.1)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  HloCSE cse(/*is_layout_sensitive=*/false, /*only_fusion_computations=*/false,
             /*ignore_control_dependencies=*/false, /*only_scalars=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&cse, m.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(absl::c_count_if(m->entry_computation()->instructions(),
                             [](const HloInstruction* instruction) {
                               return instruction->IsConstant();
                             }),
            3);
}

// TODO(jeong0982): Enable HloCseCustomCallTest, CustomCallCalledComputations,
// and CustomCallSideEffects tests once HloInstruction::CreateCustomCall is
// ported.

TEST_F(HloCseTest, IgnoreControlDependencies) {
  const char* const hlo_string = R"(
    HloModule m

    %add {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      ROOT x = s32[] add(p0, p1)
    }

    ENTRY entry {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)

      ar0 = s32[] all-reduce(p0), replica_groups={}, to_apply=%add
      ar1 = s32[] all-reduce(p1), replica_groups={}, to_apply=%add, control-predecessors={ar0}
      ar2 = s32[] all-reduce(p0), replica_groups={}, to_apply=%add, control-predecessors={ar1}
      ROOT root = tuple(ar0, ar1, ar2)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  HloCSE cse(/*is_layout_sensitive=*/false, /*only_fusion_computations=*/false,
             /*ignore_control_dependencies=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&cse, m.get()));

  SCOPED_TRACE(absl::StrCat("Module after CSE:\n", m->ToString()));
  EXPECT_EQ(changed, true);
}

TEST_F(HloCseTest, MultiOutputFusion) {
  const char* const hlo_string = R"(
    HloModule m

    f {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      add.0 = s32[] add(p0, p1)
      add.1 = s32[] add(p0, p1)
      ROOT res = (s32[], s32[]) tuple(add.0, add.1)
    }

    ENTRY entry {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      fusion = (s32[], s32[]) fusion(p0, p1), kind=kLoop, calls=f
      gte0 = s32[] get-tuple-element(fusion), index=0
      gte1 = s32[] get-tuple-element(fusion), index=1
      ROOT res = (s32[], s32[]) tuple(gte0, gte1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));
  HloCSE cse(/*is_layout_sensitive=*/false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&cse, m.get()));

  SCOPED_TRACE(absl::StrCat("Module after CSE:\n", m->ToString()));
  EXPECT_EQ(changed, true);
  HloInstruction* root = m->entry_computation()->root_instruction();
  HloInstruction* add0;
  HloInstruction* add1;
  HloInstruction* gte0;
  HloInstruction* gte1;
  ASSERT_THAT(root, GmockMatch(m::Tuple(m::GetTupleElement(&gte0),
                                        m::GetTupleElement(&gte1))));
  EXPECT_EQ(gte0, gte1);
  EXPECT_EQ(gte0->tuple_index(), 0);
  const HloInstruction* fusion = gte0->operand(0);
  ASSERT_THAT(
      fusion->fused_expression_root(),
      GmockMatch(m::Tuple(m::Add(&add0, m::Parameter(0), m::Parameter(1)),
                          m::Add(&add1, m::Parameter(0), m::Parameter(1)))));
  EXPECT_EQ(add0, add1);
}

class HloCseCommutativeOpTest
    : public HloCseTest,
      public ::testing::WithParamInterface<std::string /*op*/> {};

TEST_P(HloCseCommutativeOpTest, DoIt) {
  std::string op = GetParam();
  const char* kModuleStr = R"(
    HloModule m

    ENTRY test {
      p0 = s32[10] parameter(0)
      p1 = s32[10] parameter(1)
      op1 = s32[10] $0(p0, p1)
      op2 = s32[10] $0(p1, p0)
      ROOT t = tuple(op1, op2)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           absl::Substitute(kModuleStr, op)));
  ASSERT_TRUE(HloCSE(/*is_layout_sensitive=*/false).Run(module.get()).value());
  SCOPED_TRACE(module->ToString());

  const HloInstruction* op0;
  const HloInstruction* op1;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Op(&op0), m::Op(&op1))));
  EXPECT_EQ(op0, op1);
}

INSTANTIATE_TEST_SUITE_P(AlgebraicSimplifierCanonicalizeCommutativeTestSuite,
                         HloCseCommutativeOpTest,
                         ::testing::Values("add", "multiply", "and", "or",
                                           "xor", "minimum", "maximum"));

}  // namespace
}  // namespace zkx
