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
#include "zkx/hlo/evaluator/hlo_evaluator.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/array2d.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "zkx/literal.h"
#include "zkx/literal_util.h"
#include "zkx/service/dynamic_dimension_inference.h"
#include "zkx/shape_util.h"
#include "zkx/tests/literal_test_util.h"

namespace zkx {
namespace {

// Test fixture for the HloEvaluator.
class HloEvaluatorTest : public HloHardwareIndependentTestBase {
 public:
  HloEvaluatorTest() = default;

  absl::StatusOr<Literal> Evaluate(
      absl::Span<const Literal* const> arg_literals = {}) {
    return evaluator_.Evaluate(*m_->entry_computation(), arg_literals);
  }

  void TestUnaryOp(HloOpcode opcode, Literal expected, Literal input) {
    HloComputation::Builder b(TestName());
    auto c1 =
        b.AddInstruction(HloInstruction::CreateConstant(std::move(input)));
    b.AddInstruction(HloInstruction::CreateUnary(expected.shape(), opcode, c1));
    m_->AddEntryComputation(b.Build());

    TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
  }

  void TestBinaryOp(HloOpcode opcode, Literal expected, Literal lhs,
                    Literal rhs) {
    HloComputation::Builder b(TestName());
    auto c1 = b.AddInstruction(HloInstruction::CreateConstant(std::move(lhs)));
    auto c2 = b.AddInstruction(HloInstruction::CreateConstant(std::move(rhs)));
    b.AddInstruction(
        HloInstruction::CreateBinary(expected.shape(), opcode, c1, c2));
    m_->AddEntryComputation(b.Build());

    TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
  }

  void TestTernaryOp(HloOpcode opcode, Literal expected, Literal src0,
                     Literal src1, Literal src2) {
    HloComputation::Builder b(TestName());
    auto operand0 =
        b.AddInstruction(HloInstruction::CreateConstant(std::move(src0)));
    auto operand1 =
        b.AddInstruction(HloInstruction::CreateConstant(std::move(src1)));
    auto operand2 =
        b.AddInstruction(HloInstruction::CreateConstant(std::move(src2)));
    b.AddInstruction(HloInstruction::CreateTernary(
        expected.shape(), opcode, operand0, operand1, operand2));
    m_->AddEntryComputation(b.Build());

    TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
  }

  void TestEvaluateInstruction(HloInstruction* instruction,
                               const Literal& expected) {
    TF_ASSERT_OK_AND_ASSIGN(Literal result, evaluator_.Evaluate(instruction));
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
  }

  void TestRecursivelyEvaluateInstruction(HloInstruction* instruction,
                                          const Literal& expected) {
    TF_ASSERT_OK_AND_ASSIGN(
        Literal result,
        evaluator_.Evaluate(
            instruction, /*precomputed_analyses=*/{},
            /*recursively_evaluate_nonconstant_operands=*/true));
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
  }

  void TestRecursiveEvaluationFailure(HloInstruction* instruction) {
    absl::StatusOr<Literal> result =
        evaluator_.Evaluate(instruction, /*precomputed_analyses=*/{},
                            /*recursively_evaluate_nonconstant_operands=*/true);
    EXPECT_TRUE(!result.ok());
  }

 protected:
  HloEvaluator evaluator_;
  std::unique_ptr<HloModule> m_ = CreateNewVerifiedModule();
};

// Verifies that clamping of int64_t does not cause loss of precision
TEST_F(HloEvaluatorTest, DoesClampInt64) {
  auto ones = [](int bits) { return (int64_t{1} << bits) - 1; };

  auto low =
      LiteralUtil::CreateR2<int64_t>({{0, ones(54)}, {ones(54), ones(58)}});
  auto value = LiteralUtil::CreateR2<int64_t>({{0, ones(56)}, {0, ones(58)}});
  auto high = LiteralUtil::CreateR2<int64_t>(
      {{ones(54), ones(55)}, {ones(56), ones(58)}});

  Shape shape = low.shape();
  HloComputation::Builder b(TestName());
  auto c1 = b.AddInstruction(HloInstruction::CreateConstant(std::move(low)));
  auto c2 = b.AddInstruction(HloInstruction::CreateConstant(std::move(value)));
  auto c3 = b.AddInstruction(HloInstruction::CreateConstant(std::move(high)));
  b.AddInstruction(
      HloInstruction::CreateTernary(shape, HloOpcode::kClamp, c1, c2, c3));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected =
      LiteralUtil::CreateR2<int64_t>({{0, ones(55)}, {ones(54), ones(58)}});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise addition with 2 operands.
TEST_F(HloEvaluatorTest, DoesAdd) {
  auto lhs = LiteralUtil::CreateR2<int64_t>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64_t>({{2, 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int64_t>({{3, 4}, {-96, 8}});
  TestBinaryOp(HloOpcode::kAdd, std::move(expected), std::move(lhs),
               std::move(rhs));
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise and with 2 operands.
TEST_F(HloEvaluatorTest, DoesAnd) {
  auto lhs = LiteralUtil::CreateR2<int64_t>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64_t>({{2, 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int64_t>({{0, 0}, {4, 4}});
  TestBinaryOp(HloOpcode::kAnd, std::move(expected), std::move(lhs),
               std::move(rhs));
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise or with 2 operands.
TEST_F(HloEvaluatorTest, DoesOr) {
  auto lhs = LiteralUtil::CreateR2<int64_t>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64_t>({{2, 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int64_t>({{3, 4}, {-100, 4}});
  TestBinaryOp(HloOpcode::kOr, std::move(expected), std::move(lhs),
               std::move(rhs));
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise xor with 2 operands.
TEST_F(HloEvaluatorTest, DoesXor) {
  auto lhs = LiteralUtil::CreateR2<int64_t>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64_t>({{2, 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int64_t>({{3, 4}, {-104, 0}});
  TestBinaryOp(HloOpcode::kXor, std::move(expected), std::move(lhs),
               std::move(rhs));
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise multiply with 2 operands.
TEST_F(HloEvaluatorTest, DoesMultiply) {
  auto lhs = LiteralUtil::CreateR2<int32_t>({{-1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int32_t>(
      {{std::numeric_limits<int32_t>::min(), 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int32_t>(
      {{std::numeric_limits<int32_t>::min(), 0}, {-400, 16}});
  TestBinaryOp(HloOpcode::kMultiply, std::move(expected), std::move(lhs),
               std::move(rhs));
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise divide with 2 operands.
TEST_F(HloEvaluatorTest, DoesDivideInt64) {
  auto lhs = LiteralUtil::CreateR2<int64_t>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64_t>({{2, 4}, {4, 4}});
  auto expected = LiteralUtil::CreateR2<int64_t>({{0, 0}, {-25, 1}});
  TestBinaryOp(HloOpcode::kDivide, std::move(expected), std::move(lhs),
               std::move(rhs));
}

TEST_F(HloEvaluatorTest, DoesClampS64) {
  auto low = LiteralUtil::CreateR1<int64_t>(
      {-8616761059752331528LL, 6780561065411491190LL, -8616761059752331528LL});
  auto value = LiteralUtil::CreateR1<int64_t>(
      {-6780561065411491190LL, 6780561065411491180LL, 4241131823772864090LL});
  auto high = LiteralUtil::CreateR1<int64_t>(
      {-6780561065411491180LL, 8616761059752331528LL, 3832151243857508051LL});
  auto expected = LiteralUtil::CreateR1<int64_t>(
      {-6780561065411491190LL, 6780561065411491190LL, 3832151243857508051LL});
  TestTernaryOp(HloOpcode::kClamp, std::move(expected), std::move(low),
                std::move(value), std::move(high));
}

// Verifies that HloEvaluator evaluates a HLO instruction that performs
// element-wise abs op with 1 operand.
TEST_F(HloEvaluatorTest, DoesAbsR2) {
  auto operand = LiteralUtil::CreateR2<int64_t>({{1, -20}, {-100, 4}});
  auto expected = LiteralUtil::CreateR2<int64_t>({{1, 20}, {100, 4}});
  TestUnaryOp(HloOpcode::kAbs, std::move(expected), std::move(operand));
}

TEST_F(HloEvaluatorTest, DoesNegateR2) {
  auto operand = LiteralUtil::CreateR2<int32_t>(
      {{0, std::numeric_limits<int32_t>::min()}, {-1, 4}});
  auto expected = LiteralUtil::CreateR2<int32_t>(
      {{0, std::numeric_limits<int>::min()}, {1, -4}});
  TestUnaryOp(HloOpcode::kNegate, std::move(expected), std::move(operand));
}

TEST_F(HloEvaluatorTest, DoesNotR2) {
  auto operand =
      LiteralUtil::CreateR2<int32_t>({{0, std::numeric_limits<int>::min()},
                                      {-1, std::numeric_limits<int>::max()}});
  auto expected =
      LiteralUtil::CreateR2<int32_t>({{-1, std::numeric_limits<int>::max()},
                                      {0, std::numeric_limits<int>::min()}});
  TestUnaryOp(HloOpcode::kNot, std::move(expected), std::move(operand));
}

// Verifies that HloEvaluator evaluates a HLO Computation with non-parameter nor
// constant operands.
TEST_F(HloEvaluatorTest, DoesTraverseInstructions) {
  auto lhs = LiteralUtil::CreateR2<int64_t>({{1, 0}, {-100, 4}});
  auto rhs = LiteralUtil::CreateR2<int64_t>({{2, 4}, {4, 4}});
  auto rhs2 = LiteralUtil::CreateR2<int64_t>({{1, -20}, {-100, 4}});
  std::vector<const Literal*> args = {&lhs, &rhs, &rhs2};

  Shape shape = ShapeUtil::MakeShape(S64, {2, 2});

  HloComputation::Builder b(TestName());
  auto param_lhs =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "lhs"));
  auto param_rhs =
      b.AddInstruction(HloInstruction::CreateParameter(1, shape, "rhs"));
  auto lhs_instruction = b.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd, param_lhs, param_rhs));

  auto param_rhs2 =
      b.AddInstruction(HloInstruction::CreateParameter(2, shape, "rhs2"));
  b.AddInstruction(HloInstruction::CreateBinary(shape, HloOpcode::kAdd,
                                                lhs_instruction, param_rhs2));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate(args));

  auto expected = LiteralUtil::CreateR2<int64_t>({{4, -16}, {-196, 12}});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

// Verifies Broadcast operation is correctly evaluated.
TEST_F(HloEvaluatorTest, DoesBroadcast) {
  HloComputation::Builder b(TestName());
  auto input_literal = LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}, {5, 6}});
  auto output_literal = LiteralUtil::CreateR3<int32_t>(
      {{{1, 2}, {3, 4}, {5, 6}}, {{1, 2}, {3, 4}, {5, 6}}});
  HloInstruction* literal_instruction = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  b.AddInstruction(HloInstruction::CreateBroadcast(
      output_literal.shape(), literal_instruction, {1, 2}));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({}));

  EXPECT_TRUE(LiteralTestUtil::Equal(result, output_literal));
}

TEST_F(HloEvaluatorTest, DoesBroadcastScalar) {
  HloComputation::Builder b(TestName());
  auto input_literal = LiteralUtil::CreateR0<int32_t>(111);
  auto output_literal = LiteralUtil::CreateR2<int32_t>(
      {{111, 111}, {111, 111}, {111, 111}, {111, 111}, {111, 111}, {111, 111}});

  HloInstruction* literal_instruction = b.AddInstruction(
      HloInstruction::CreateConstant(std::move(input_literal)));
  // Broadcast dimension should be empty in the case of scalars.
  b.AddInstruction(HloInstruction::CreateBroadcast(
      output_literal.shape(), literal_instruction,
      /*broadcast_dimensions=*/{}));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({}));

  EXPECT_TRUE(LiteralTestUtil::Equal(result, output_literal));
}

TEST_F(HloEvaluatorTest, DoesConcatenateSimple) {
  HloComputation::Builder b(TestName());

  HloInstruction* operand1 = b.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int64_t>({{-1, -2}, {100, 200}})));
  HloInstruction* operand2 = b.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int64_t>({{-2, -3}, {-100, -200}})));

  std::vector<HloInstruction*> operands = {operand1, operand2};

  Shape shape = ShapeUtil::MakeShape(S64, {4, 2});
  b.AddInstruction(HloInstruction::CreateConcatenate(shape, operands, 0));

  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected = LiteralUtil::CreateR2<int64_t>(
      {{-1, -2}, {100, 200}, {-2, -3}, {-100, -200}});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest, ConcatenateHandlesShapeWithZeroElement) {
  HloComputation::Builder b(TestName());

  HloInstruction* operand1 = b.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<int64_t>({100, 200})));
  HloInstruction* operand2 = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int64_t>({})));

  std::vector<HloInstruction*> operands = {operand1, operand2};

  Shape shape = ShapeUtil::MakeShape(S64, {2});
  b.AddInstruction(HloInstruction::CreateConcatenate(shape, operands, 0));

  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected = LiteralUtil::CreateR1<int64_t>({100, 200});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

// Note: ZKX's PaddingConfig does not support interior_padding (XLA does).
// This helper only sets edge_padding_low and edge_padding_high.
PaddingConfig CreatePaddingConfig(
    std::initializer_list<std::array<int64_t, 2>> padding_dimensions) {
  PaddingConfig padding_config;

  for (auto& paddings_per_dim : padding_dimensions) {
    auto dimension = padding_config.add_dimensions();
    dimension->set_edge_padding_low(paddings_per_dim[0]);
    dimension->set_edge_padding_high(paddings_per_dim[1]);
  }
  return padding_config;
}

TEST_F(HloEvaluatorTest, Pad2DIntegerArrayWithZeroDimension) {
  auto operand = LiteralUtil::CreateR2<int32_t>({{}, {}});
  HloComputation::Builder b(TestName());
  auto operand_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(operand)));

  constexpr int32_t kPadValue = 10;
  auto pad_value = LiteralUtil::CreateR0<int32_t>(kPadValue);
  auto padding_value_instruction =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(pad_value)));

  // Note: XLA version uses {{{1, 0, 2}}, {{0, 2, 1}}} with interior_padding.
  // ZKX doesn't support interior_padding, so we use only edge padding.
  auto padding_config = CreatePaddingConfig({{{1, 0}}, {{0, 2}}});
  Shape shape = ShapeUtil::MakeShape(S32, {3, 2});
  b.AddInstruction(HloInstruction::CreatePad(
      shape, operand_instruction, padding_value_instruction, padding_config));
  m_->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate());

  auto expected =
      LiteralUtil::CreateR2<int32_t>({{10, 10}, {10, 10}, {10, 10}});

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(HloEvaluatorTest, EvaluateGather_TensorFlowGatherV1) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherV1

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[2,3] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1, 3}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {7, 8, 9}}), result));
}

TEST_F(HloEvaluatorTest, EvaluateGather_TensorFlowGatherV2) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherV2

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[3,2] gather(operand, indices),
      offset_dims={0},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=1,
      slice_sizes={3, 1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{1, 3}, {4, 6}, {7, 9}}), result));
}

TEST_F(HloEvaluatorTest, EvaluateGather_TensorFlowGatherMultipleBatchDims) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherMultipleBatchDims

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,3,2] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=2,
      slice_sizes={3, 1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR2<int32_t>({{0, 2}, {2, 1}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR3<int32_t>(
          {{{1, 3}, {4, 6}, {7, 9}}, {{3, 2}, {6, 5}, {9, 8}}}),
      result));
}

TEST_F(HloEvaluatorTest, EvaluateGather_TensorFlowGatherNd) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherNd

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,2] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0,1},
      start_index_map={0,1},
      index_vector_dim=1,
      slice_sizes={1,1,2}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR3<int32_t>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal start_indices = LiteralUtil::CreateR2<int32_t>({{0, 0}, {1, 0}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{-1, 1}, {-4, 4}}), result));
}

TEST_F(HloEvaluatorTest,
       EvaluateGather_TensorFlowGatherNdNonDefaultIndexVectorDim) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherNd

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,2] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0,1},
      start_index_map={0,1},
      index_vector_dim=0,
      slice_sizes={1,1,2}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR3<int32_t>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                      {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                      {{-7, 7}, {-8, 8}, {-9, 9}}});
  Literal start_indices = LiteralUtil::CreateR2<int32_t>({{0, 0}, {1, 0}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{-2, 2}, {-1, 1}}), result));
}

TEST_F(HloEvaluatorTest, EvaluateGather_DynamicSlice) {
  const char* hlo_text = R"(
HloModule DynamicSlice

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[1,1] gather(operand, indices),
      offset_dims={0,1},
      collapsed_slice_dims={},
      start_index_map={0,1},
      index_vector_dim=0,
      slice_sizes={1,1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR1<int32_t>({1, 1});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR2<int32_t>({{5}}), result));
}

TEST_F(HloEvaluatorTest, EvaluateGather_BatchDynamicSlice) {
  const char* hlo_text = R"(
HloModule BatchDynamicSlice

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,1,1] gather(operand, indices),
      offset_dims={1,2},
      collapsed_slice_dims={},
      start_index_map={0,1},
      index_vector_dim=0,
      slice_sizes={1,1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal start_indices = LiteralUtil::CreateR2<int32_t>({{2, 1}, {1, 1}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR3<int32_t>({{{8}}, {{5}}}), result));
}

TEST_F(HloEvaluatorTest, EvaluateGather_ZeroDimBounds) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherV1

ENTRY main {
  operand = s32[3,0] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[2,0] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1, 0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand = LiteralUtil::CreateR2<int32_t>({{}, {}, {}});
  Literal start_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR2<int32_t>({{}, {}}), result));
}

TEST_F(HloEvaluatorTest, EvaluateGather_NoOutputWindowDims) {
  const std::string hlo_text = R"(
HloModule GatherXd

ENTRY main {
  operand = s32[3] parameter(0)
  indices = s32[2,2,1] parameter(1)
  ROOT gather = s32[2,2] gather(operand, indices),
      offset_dims={},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=2,
      slice_sizes={1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));

  Literal operand = LiteralUtil::CreateR1<int32_t>({0, 1, 2});
  Literal start_indices =
      LiteralUtil::CreateR3<int32_t>({{{0}, {1}}, {{2}, {1}}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{0, 1}, {2, 1}}), result));
}

TEST_F(HloEvaluatorTest, EvaluateGather_ExplicitBatchDims) {
  const std::string hlo_text = R"(
HloModule gather

ENTRY main {
  operand = s32[3,2,1,3] parameter(0)
  indices = s32[3,2] parameter(1)
  ROOT gather = s32[3,2,2] gather(operand, indices),
      offset_dims={2},
      collapsed_slice_dims={2},
      start_index_map={0},
      index_vector_dim=2,
      slice_sizes={2,1,1,1},
      operand_batching_dims={1,3},
      start_indices_batching_dims={1,0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));

  Literal operand =
      LiteralUtil::CreateR4<int32_t>({{{{1, 2, 3}}, {{4, 5, 6}}},
                                      {{{7, 8, 9}}, {{10, 11, 12}}},
                                      {{{13, 14, 15}}, {{16, 17, 18}}}});
  Literal start_indices =
      LiteralUtil::CreateR2<int32_t>({{1, 0}, {0, 1}, {1, 0}});
  Literal expected_result = LiteralUtil::CreateR3<int32_t>(
      {{{7, 13}, {4, 10}}, {{2, 8}, {11, 17}}, {{9, 15}, {6, 12}}});

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand, &start_indices}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_result, result));
}

// Note: Original XLA test uses f32; converted to s32 for ZKX.
TEST_F(HloEvaluatorTest, EvaluateGather_GetDiagonal) {
  const std::string hlo_text = R"(
HloModule module

ENTRY %module {
  %operand = s32[4,4] parameter(0)
  %indices = s32[4,1] iota(), iota_dimension=0
  ROOT %gather = s32[4,1] gather(%operand, %indices), offset_dims={},
    collapsed_slice_dims={1}, start_index_map={1}, operand_batching_dims={0},
    start_indices_batching_dims={0}, index_vector_dim=2, slice_sizes={1,1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));

  Literal operand = LiteralUtil::CreateR2<int32_t>(
      {{0, 1, 2, 3}, {10, 11, 12, 13}, {20, 21, 22, 23}, {30, 31, 32, 33}});
  Literal expected_result =
      LiteralUtil::CreateR2<int32_t>({{0}, {11}, {22}, {33}});

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Evaluate({&operand}));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_result, result));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_TensorFlowScatterV1_Update) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {70, 80, 90}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {4, 5, 6}, {70, 80, 90}}),
      result));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_TensorFlowScatterV2_Update) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterV2

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  ROOT rhs = s32[] parameter(1)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[3,2] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={0},
      inserted_window_dims={1},
      scatter_dims_to_operand_dims={1},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 30}, {40, 60}, {70, 90}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{10, 2, 30}, {40, 5, 60}, {70, 8, 90}}),
      result));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_TensorFlowScatter_Add) {
  const char* hlo_text = R"(
HloModule TensorFlowScatter

add_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=add_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {70, 80, 90}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{11, 22, 33}, {4, 5, 6}, {77, 88, 99}}),
      result));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_TensorFlowScatter_Mul) {
  const char* hlo_text = R"(
HloModule TensorFlowScatter

mul_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT mul = s32[] multiply(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=mul_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({0, 2});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {70, 80, 90}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR2<int32_t>(
                                 {{10, 40, 90}, {4, 5, 6}, {490, 640, 810}}),
                             result));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_TensorFlowScatter_RepeatedIndices) {
  const char* hlo_text = R"(
HloModule TensorFlowScatter

add_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=add_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR1<int32_t>({1, 1});
  Literal updates =
      LiteralUtil::CreateR2<int32_t>({{10, 20, 30}, {70, 80, 90}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {84, 105, 126}, {7, 8, 9}}),
      result));
}

TEST_F(HloEvaluatorTest, EvaluateScatter_TensorFlowScatter_MultipleBatchDims) {
  const char* hlo_text = R"(
HloModule TensorFlowScatterMultipleBatchDims

add_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(s32[] lhs, s32[] rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  updates = s32[2,3,2] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=add_s32,
      update_window_dims={1},
      inserted_window_dims={1},
      scatter_dims_to_operand_dims={1},
      index_vector_dim=2
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal operand =
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  Literal scatter_indices = LiteralUtil::CreateR2<int32_t>({{0, 2}, {2, 1}});
  Literal updates = LiteralUtil::CreateR3<int32_t>(
      {{{10, 30}, {40, 60}, {70, 90}}, {{5, 5}, {5, 5}, {5, 5}}});
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&operand, &scatter_indices, &updates}));
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR2<int32_t>(
                                 {{11, 7, 38}, {44, 10, 71}, {77, 13, 104}}),
                             result));
}

// Check that s32 under/overflow doesn't trigger a ubsan failure.
TEST_F(HloEvaluatorTest, Int32Overflow) {
  const std::string_view hlo_text = R"(
HloModule Test

ENTRY main {
  c1 = s32[] constant(1073741824)  // 2^30
  sum = s32[] add(c1, c1)  // 2^31, i.e. INT_MIN

  c2 = s32[] constant(-2147483648)  // -2^31
  sub = s32[] subtract(c2, c1)  // -2^31 - 2^30, underflows

  c3 = u32[] constant(4294967295)
  c4 = u32[] constant(33)

  mul = s32[] multiply(c1, c1)

  pow = u32[] power(c3, c4)
  ROOT tuple = (s32[], s32[], s32[], u32[]) tuple(sum, sub, mul, pow)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(auto literal, Evaluate({}));
  std::vector<Literal> actual = literal.DecomposeTuple();
  ASSERT_EQ(actual.size(), 4);

  uint32_t pow30 = uint32_t{1} << 30;
  uint32_t pow31 = uint32_t{1} << 31;
  EXPECT_EQ(actual[0].GetFirstElement<int32_t>(), static_cast<int32_t>(pow31));
  EXPECT_EQ(actual[1].GetFirstElement<int32_t>(),
            static_cast<int32_t>(-(pow31 + pow30)));
  EXPECT_EQ(actual[2].GetFirstElement<int32_t>(),
            static_cast<int32_t>(pow31 * pow31));
  EXPECT_EQ(actual[3].GetFirstElement<uint32_t>(), uint32_t{4294967295});
}

TEST_F(HloEvaluatorTest, GetDimensionSize) {
  const std::string_view hlo_text = R"(
HloModule Test

ENTRY main {
  size = s32[] parameter(0)

  data = s32[4] parameter(1)

  data_dynamic = s32[<=4] set-dimension-size(data, size), dimensions={0}

  sum = s32[<=4] add(data_dynamic, data)

  ROOT dynamic_size = s32[] get-dimension-size(sum), dimensions={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));

  TF_ASSERT_OK_AND_ASSIGN(DynamicDimensionInference dynamic_dimension_inference,
                          DynamicDimensionInference::Run(m_.get()));

  evaluator_.set_dynamic_dimension_inference(&dynamic_dimension_inference);
  Literal size_arg = LiteralUtil::CreateR0<int32_t>(3);
  Literal data_arg = LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4});

  TF_ASSERT_OK_AND_ASSIGN(Literal actual, Evaluate({&size_arg, &data_arg}));

  EXPECT_EQ(actual.GetFirstElement<int32_t>(), static_cast<int32_t>(3));
}

// Check that we get a useful error if we pass inputs of the wrong shape.
TEST_F(HloEvaluatorTest, EvaluateWithWrongInputShapes) {
  const std::string_view hlo_text = R"(
HloModule Test

ENTRY main {
  p0 = s32[1] parameter(0)
  ROOT sum = s32[1] add(p0, p0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal input_wrong_shape = LiteralUtil::CreateR1<int32_t>({0, 1});

  EXPECT_EQ(
      HloEvaluator().Evaluate(*m_, {&input_wrong_shape}).status().message(),
      "Shape mismatch at parameter 0. Computation expected s32[1]{0}, "
      "but arg was s32[2]{0}.");
  EXPECT_EQ(HloEvaluator()
                .Evaluate(*m_->entry_computation(), {&input_wrong_shape})
                .status()
                .message(),
            "Shape mismatch at parameter 0. Computation expected s32[1]{0}, "
            "but arg was s32[2]{0}.");
}

// Check that we get a useful error if we pass too many or too few inputs.
TEST_F(HloEvaluatorTest, EvaluateWithWrongNumberOfInputs) {
  const std::string_view hlo_text = R"(
HloModule Test

ENTRY main {
  p0 = s32[1] parameter(0)
  ROOT sum = s32[1] add(p0, p0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal input = LiteralUtil::CreateR1<int32_t>({0});

  EXPECT_EQ(HloEvaluator().Evaluate(*m_, {&input, &input}).status().message(),
            "Expected 1 argument, but got 2.");
  EXPECT_EQ(HloEvaluator()
                .Evaluate(*m_->entry_computation(), {&input, &input})
                .status()
                .message(),
            "Expected 1 argument, but got 2.");
}

// Tests when a custom_call handler returns an error.
TEST_F(HloEvaluatorTest, EvaluateCustomCall_HandlerError) {
  const std::string_view hlo_text = R"(
    HloModule EvaluateCustomCall_HandlerError
    ENTRY kernel_entry {
      parameter.0 = u32[2,2]{1,0} parameter(0)
      ROOT test_root = (u32[2,2]{1,0}) custom-call(parameter.0),
          custom_call_target="_my_custom_call"
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal arg = LiteralUtil::CreateR2<uint32_t>({{1, 2}, {3, 4}});
  HloEvaluator evaluator;
  evaluator.set_custom_call_handler(
      [](const HloInstruction* custom_call,
         absl::Span<const Literal*> operands) -> absl::StatusOr<Literal> {
        return absl::InternalError("Test error");
      });
  EXPECT_EQ(evaluator.Evaluate(*m_, {&arg}).status().code(),
            absl::StatusCode::kInternal);
}

// Tests the custom_call handler on calls with many inputs.
// We sum the operands so that we can verify the operand and output literals
// are properly mapped for access.
TEST_F(HloEvaluatorTest, EvaluateCustomCall_ManyInputs) {
  const std::string_view hlo_text = R"(
    HloModule EvaluateCustomCall_ManyInputs
    ENTRY kernel_entry {
      parameter.0 = u32[1]{0} parameter(0)
      parameter.1 = u32[1]{0} parameter(1)
      ROOT test_root = u32[1]{0} custom-call(parameter.0, parameter.1),
          custom_call_target="_my_custom_call"
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  Literal arg0 = LiteralUtil::CreateR1<uint32_t>({5});
  Literal arg1 = LiteralUtil::CreateR1<uint32_t>({7});
  HloEvaluator evaluator;
  evaluator.set_custom_call_handler([](const HloInstruction* custom_call,
                                       absl::Span<const Literal*> operands) {
    EXPECT_EQ(HloOpcode::kCustomCall, custom_call->opcode());
    EXPECT_EQ("_my_custom_call", custom_call->custom_call_target());
    EXPECT_EQ(2, custom_call->operand_count());
    EXPECT_EQ(2, operands.size());
    auto output = Literal::CreateFromShape(custom_call->shape());
    auto operand0_data = operands[0]->data<uint32_t>();
    auto operand1_data = operands[1]->data<uint32_t>();
    auto output_data = output.data<uint32_t>();
    output_data[0] = operand0_data[0] + operand1_data[0];
    return output;
  });
  TF_ASSERT_OK_AND_ASSIGN(
      Literal actual_literal,
      evaluator.Evaluate(*m_->entry_computation(), {&arg0, &arg1}));
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<uint32_t>({12}),
                                     actual_literal));
}

TEST_F(HloEvaluatorTest, EvaluateWithSubstitutionsRecursive) {
  const char* hlo = R"(
  HloModule test

  ENTRY main {
    param = s32[] parameter(0)
    c1 = s32[] constant(1)
    c2 = s32[] constant(2)
    add.1 = s32[] add(c1, c2)
    ROOT add.2 = s32[] add(param, add.1)
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  Literal param_value = LiteralUtil::CreateR0(PrimitiveType::S32, 3);
  HloInstruction* param = module->entry_computation()->parameter_instruction(0);
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      evaluator_.EvaluateWithSubstitutions(
          /*instruction=*/module->entry_computation()->root_instruction(),
          /*substitutions=*/{{param, &param_value}},
          /*recursively_evaluate_nonconstant_operands=*/true));
  EXPECT_EQ(result, LiteralUtil::CreateR0(PrimitiveType::S32, 1 + 2 + 3));
}

TEST_F(HloEvaluatorTest,
       EvaluateWithSubstitutionsRecursiveWithDeepSubstitutions) {
  const char* hlo = R"(
  HloModule test
  ENTRY main {
    param = s32[] parameter(0)
    c1 = s32[] constant(1)
    c2 = s32[] constant(2)
    add.1 = s32[] add(param, c1)
    add.2 = s32[] add(add.1, c2)
    ROOT add.3 = s32[] add(add.2, c1)
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  Literal param_value = LiteralUtil::CreateR0(PrimitiveType::S32, 4);
  HloInstruction* param = module->entry_computation()->parameter_instruction(0);
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      evaluator_.EvaluateWithSubstitutions(
          /*instruction=*/module->entry_computation()->root_instruction(),
          /*substitutions=*/{{param, &param_value}},
          /*recursively_evaluate_nonconstant_operands=*/true));
  EXPECT_EQ(result, LiteralUtil::CreateR0(PrimitiveType::S32, 4 + 1 + 2 + 1));
}

TEST_F(HloEvaluatorTest, EvaluateWithSubstitutionsLiteralBase) {
  HloComputation::Builder b(TestName());
  Shape shape = ShapeUtil::MakeShape(S64, {3});

  HloInstruction* param0 =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "param0"));
  HloInstruction* square = b.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kMultiply, param0, param0));

  int64_t int64_values[] = {1, 2, 3};
  const Shape literal_shape = ShapeUtil::MakeShape(S64, {3});

  BorrowingLiteral literal(reinterpret_cast<const char*>(int64_values),
                           literal_shape);
  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(Literal result, evaluator.EvaluateWithSubstitutions(
                                              square, {{param0, &literal}}));
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int64_t>({1, 4, 9}),
                                     result));
}

TEST_F(HloEvaluatorTest, DotUpcast) {
  const std::string_view hlo_text = R"(
  HloModule test
  ENTRY DotUpcast {
    l = s16[4,3]{1,0} parameter(0)
    r = s8[3,2]{1,0} parameter(1)
    ROOT result = s32[4,2] dot(l, r), lhs_contracting_dims={1},
                                      rhs_contracting_dims={0}
  }
  )";
  // lhs:
  // s16[4,3] {
  //  { 1, 2, 3 },
  //  { 5, 6, 7 },
  //  { 9, 10, 11 },
  //  { 13, 14, 15 },
  // }
  auto lhs_array = std::make_unique<Array2D<int16_t>>(4, 3);
  lhs_array->FillUnique(1);
  auto lhs_literal = LiteralUtil::CreateR2FromArray2D<int16_t>(*lhs_array);

  // rhs:
  // s8[3,2] {
  //  { 1, 2 },
  //  { 3, 4 },
  //  { 5, 6 },
  // }
  auto rhs_array = std::make_unique<Array2D<int8_t>>(3, 2);
  rhs_array->FillUnique(1);
  auto rhs_literal = LiteralUtil::CreateR2FromArray2D<int8_t>(*rhs_array);
  TF_ASSERT_OK_AND_ASSIGN(m_, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Evaluate({&lhs_literal, &rhs_literal}));

  auto expected_array =
      Array2D<int32_t>({{22, 28}, {58, 76}, {94, 124}, {130, 172}});
  auto expected = LiteralUtil::CreateR2FromArray2D<int32_t>(expected_array);

  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

// Tests that HloEvaluator can evaluate an instruction even when its operands
// are not constant.
TEST_F(HloEvaluatorTest, RecursivelyEvaluateNonConstantOperands) {
  Literal c0_literal = LiteralUtil::CreateR2<int32_t>({{0, 2}, {2, 4}});
  Literal c1_literal = LiteralUtil::CreateR2<int32_t>({{0, 5}, {0, 4}});
  Literal c2_literal = LiteralUtil::CreateR2<int32_t>({{2, 4}, {4, 4}});

  Shape shape = c0_literal.shape();
  HloComputation::Builder b(TestName());
  HloInstruction* c0 =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(c0_literal)));
  HloInstruction* c1 =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(c1_literal)));
  HloInstruction* c2 =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(c2_literal)));

  HloInstruction* add0 = b.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, c0, c1));
  HloInstruction* add1 = b.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, c1, c2));
  HloInstruction* add2 = b.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, add0, add1));

  m_->AddEntryComputation(b.Build());
  Literal expected = LiteralUtil::CreateR2<int32_t>({{2, 16}, {6, 16}});
  TestRecursivelyEvaluateInstruction(add2, expected);
}

// Tests that HloEvaluator can evaluate a GetTupleElement even when its operand
// Tuple instruction cannot be fully evaluated. Note that this requires that the
//  tuple element at the given tuple index can be evaluated.
TEST_F(HloEvaluatorTest, GetTupleElementOnPartiallyKnownTupleSucceeds) {
  Literal c0_literal = LiteralUtil::CreateR2<int32_t>({{0, 2}, {2, 4}});

  Shape shape = c0_literal.shape();
  HloComputation::Builder b(TestName());
  HloInstruction* c0 =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(c0_literal)));
  HloInstruction* p0 =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "param.0"));
  HloInstruction* p1 =
      b.AddInstruction(HloInstruction::CreateParameter(1, shape, "param.1"));

  HloInstruction* tuple =
      b.AddInstruction(HloInstruction::CreateTuple({p0, p1, c0}));
  HloInstruction* gte =
      b.AddInstruction(HloInstruction::CreateGetTupleElement(tuple, 2));

  m_->AddEntryComputation(b.Build());
  Literal expected = LiteralUtil::CreateR2<int32_t>({{0, 2}, {2, 4}});
  TestRecursivelyEvaluateInstruction(gte, expected);
}

// Tests that Infeed cannot be evaluated.
TEST_F(HloEvaluatorTest, InfeedFailure) {
  HloComputation::Builder b(TestName());
  HloInstruction* token = b.AddInstruction(HloInstruction::CreateToken());
  HloInstruction* infeed = b.AddInstruction(HloInstruction::CreateInfeed(
      ShapeUtil::MakeShape(S32, {4, 4}), token, ""));

  m_->AddEntryComputation(b.Build());
  TestRecursiveEvaluationFailure(infeed);
}

// Tests that GetTupleElement cannot be evaluated if the corresponding tuple
// element cannot be evaluated.
TEST_F(HloEvaluatorTest, GetUnknownTupleElementFails) {
  Literal c0_literal = LiteralUtil::CreateR2<int32_t>({{0, 2}, {2, 4}});

  Shape shape = c0_literal.shape();
  HloComputation::Builder b(TestName());
  HloInstruction* c0 =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(c0_literal)));
  HloInstruction* p0 =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "param.0"));
  HloInstruction* p1 =
      b.AddInstruction(HloInstruction::CreateParameter(1, shape, "param.1"));

  HloInstruction* tuple =
      b.AddInstruction(HloInstruction::CreateTuple({p0, p1, c0}));
  HloInstruction* gte =
      b.AddInstruction(HloInstruction::CreateGetTupleElement(tuple, 0));

  m_->AddEntryComputation(b.Build());
  TestRecursiveEvaluationFailure(gte);
}

// Tests that partial evaluation works for nested tuples.
TEST_F(HloEvaluatorTest, GetTupleElementFromNestedTupleSucceeds) {
  Literal c0_literal = LiteralUtil::CreateR2<int32_t>({{0, 2}, {2, 4}});

  Shape shape = c0_literal.shape();
  HloComputation::Builder b(TestName());
  HloInstruction* c0 =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(c0_literal)));
  HloInstruction* p0 =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "param.0"));
  HloInstruction* p1 =
      b.AddInstruction(HloInstruction::CreateParameter(1, shape, "param.1"));

  HloInstruction* tuple0 =
      b.AddInstruction(HloInstruction::CreateTuple({p0, c0}));
  HloInstruction* tuple1 =
      b.AddInstruction(HloInstruction::CreateTuple({tuple0, p1}));
  HloInstruction* gte0 =
      b.AddInstruction(HloInstruction::CreateGetTupleElement(tuple1, 0));
  HloInstruction* gte1 =
      b.AddInstruction(HloInstruction::CreateGetTupleElement(gte0, 1));

  m_->AddEntryComputation(b.Build());
  Literal expected = LiteralUtil::CreateR2<int32_t>({{0, 2}, {2, 4}});
  TestRecursivelyEvaluateInstruction(gte1, expected);
}

// Tests that partial evaluation works when the GetTupleElement is interleaved
// with other Tuple instructions.
TEST_F(HloEvaluatorTest, GetTupleElementInterleavedWithTupleSucceeds) {
  Literal c0_literal = LiteralUtil::CreateR2<int32_t>({{0, 2}, {2, 4}});

  Shape shape = c0_literal.shape();
  HloComputation::Builder b(TestName());
  HloInstruction* c0 =
      b.AddInstruction(HloInstruction::CreateConstant(std::move(c0_literal)));
  HloInstruction* p0 =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "param.0"));
  HloInstruction* p1 =
      b.AddInstruction(HloInstruction::CreateParameter(1, shape, "param.1"));
  HloInstruction* p2 =
      b.AddInstruction(HloInstruction::CreateParameter(2, shape, "param.2"));

  HloInstruction* tuple0 =
      b.AddInstruction(HloInstruction::CreateTuple({p0, c0}));
  HloInstruction* tuple1 =
      b.AddInstruction(HloInstruction::CreateTuple({tuple0, p1}));
  HloInstruction* gte0 =
      b.AddInstruction(HloInstruction::CreateGetTupleElement(tuple1, 0));
  HloInstruction* tuple2 =
      b.AddInstruction(HloInstruction::CreateTuple({gte0, p2}));
  HloInstruction* gte1 =
      b.AddInstruction(HloInstruction::CreateGetTupleElement(tuple2, 0));
  HloInstruction* gte2 =
      b.AddInstruction(HloInstruction::CreateGetTupleElement(gte1, 1));

  m_->AddEntryComputation(b.Build());
  Literal expected = LiteralUtil::CreateR2<int32_t>({{0, 2}, {2, 4}});
  TestRecursivelyEvaluateInstruction(gte2, expected);
}

// Tests that we can evaluate a parameter instruction through the call graph.
TEST_F(HloEvaluatorTest, ParameterThroughCallSucceeds) {
  constexpr std::string_view kHloModule = R"(
    HloModule parameter_through_call

    %identity {
      ROOT %param = s32[] parameter(0)
    }

    ENTRY parameter_through_call {
      %constant = s32[] constant(42)
      ROOT %call = s32[] call(s32[] %constant), to_apply=%identity
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloModule));
  const HloInstruction* parameter_instruction = nullptr;
  for (const auto* computation : hlo_module->computations()) {
    for (const auto* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kParameter) {
        parameter_instruction = instruction;
      }
    }
  }
  ASSERT_NE(parameter_instruction, nullptr);

  Literal expected = LiteralUtil::CreateR0<int32_t>(42);
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      evaluator_.Evaluate(parameter_instruction, /*precomputed_analyses=*/{},
                          /*recursively_evaluate_nonconstant_operands=*/true));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

}  // namespace
}  // namespace zkx
