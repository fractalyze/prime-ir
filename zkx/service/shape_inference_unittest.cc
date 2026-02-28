/* Copyright 2017 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

#include "zkx/service/shape_inference.h"

#include <array>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/hlo/parser/hlo_parser.h"

namespace zkx {

using ::testing::ContainsRegex;
using ::testing::HasSubstr;

constexpr std::string_view kBroadcastDimensionMismatchErrorMessage =
    "Broadcast dimension 0 mismatch";
constexpr std::string_view kIncompatibleBinaryOpShapeErrorMessage =
    "Binary op with incompatible shapes";
std::array<const int64_t, 1> zero_array = {0};

class ShapeInferenceTest : public ::testing::Test {
 protected:
  // Some handy scalar shapes.
  const Shape s32_ = ShapeUtil::MakeShape(S32, {});
  const Shape u16_ = ShapeUtil::MakeShape(U16, {});
  const Shape u32_ = ShapeUtil::MakeShape(U32, {});
  const Shape u64_ = ShapeUtil::MakeShape(U64, {});
  const Shape pred_ = ShapeUtil::MakeShape(PRED, {});

  // Some handy vector and matrix shapes of U32 type.
  // Suffix: vector_length_, matrix_rows_cols_
  const Shape vector_32_ = ShapeUtil::MakeShape(U32, {32});
  const Shape vector_64_ = ShapeUtil::MakeShape(U32, {64});
  const Shape matrix_32_48_ = ShapeUtil::MakeShape(U32, {32, 48});
  const Shape matrix_32_64_ = ShapeUtil::MakeShape(U32, {32, 64});
  const Shape matrix_64_48_ = ShapeUtil::MakeShape(U32, {64, 48});

  // Some handy S32 arrays.
  const Shape s32matrix_64_64_ = ShapeUtil::MakeShape(S32, {64, 64});
};

// Subclass for testing InferReduceShape.
class ReduceShapeInferenceTest : public ShapeInferenceTest {
 protected:
  // Helper that runs reduce shape inference with the input 'arg' and given
  // dimensions to reduce, and checks the inferred shape is as expected. The
  // element type here is hard-coded to U32.
  void ExpectInferredReduceShape(
      const Shape& expected_inferred_shape, const Shape& arg,
      absl::Span<const int64_t> dimensions_to_reduce) {
    ProgramShape to_apply = ShapeUtil::MakeProgramShape({u32_, u32_}, u32_);
    const absl::StatusOr<Shape> inferred_shape =
        ShapeInference::InferReduceShape({&arg, &u32_}, dimensions_to_reduce,
                                         to_apply);
    EXPECT_TRUE(inferred_shape.status().ok());
    EXPECT_TRUE(ShapeUtil::Equal(expected_inferred_shape, *inferred_shape));
  }
};

struct BinaryOpTestCase {
  std::string lhs;
  std::string rhs;
  absl::Span<const int64_t> broadcast_dimensions;
  std::string expected;
  std::optional<std::string_view> error_message;
};

// Subclass for testing unbounded dynamic logical ops
class UnboundedLogicalOpShapeInferenceTest
    : public ::testing::TestWithParam<BinaryOpTestCase> {};

// Subclass for testing unbounded dynamic binary ops
class UnboundedBinaryOpShapeInferenceTest
    : public ::testing::TestWithParam<BinaryOpTestCase> {};

// Subclass for testing unbounded dynamic compare op
class UnboundedCompareOpShapeInferenceTest
    : public ::testing::TestWithParam<BinaryOpTestCase> {};

// Subclass for testing unbounded dynamic complex op
class UnboundedComplexOpShapeInferenceTest
    : public ::testing::TestWithParam<BinaryOpTestCase> {};

// Subclass for testing unbounded dynamic concatenate op
class UnboundedConcatenateOpShapeInferenceTest
    : public ::testing::TestWithParam<std::vector<std::string>> {};

struct UnaryOpTestCase {
  std::string operand;
  std::string expected;
  HloOpcode opcode;
};

// Subclass for testing unbounded dynamic unary ops
class UnboundedUnaryOpShapeInferenceTest
    : public ::testing::TestWithParam<UnaryOpTestCase> {};

// Subclass for testing unbounded dynamic clamp op
class UnboundedClampOpShapeInferenceTest
    : public ::testing::TestWithParam<std::vector<std::string>> {};

// Subclass for testing unbounded dynamic select op
class UnboundedSelectOpShapeInferenceTest
    : public ::testing::TestWithParam<std::vector<std::string>> {};

TEST_F(ShapeInferenceTest, UnaryNegateMatrix) {
  const Shape matrix_shape = ShapeUtil::MakeShape(U32, {128, 64});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferUnaryOpShape(HloOpcode::kNegate, matrix_shape);
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_shape, *inferred_shape));
}

TEST_F(ShapeInferenceTest, SelectScalarPredBetweenTuples) {
  const Shape tuple = ShapeUtil::MakeTupleShape({s32_, u32_});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kSelect, pred_, tuple,
                                          tuple);
  ASSERT_FALSE(inferred_shape.ok());
  ASSERT_THAT(inferred_shape.status().message(),
              HasSubstr("Expected array argument for select"));
}

TEST_F(ShapeInferenceTest, SelectScalarPredBetweenArrays) {
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kSelect, pred_,
                                          matrix_64_48_, matrix_64_48_);
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_64_48_, *inferred_shape));
}

TEST_F(ShapeInferenceTest, SelectArrayPredBetweenArrays) {
  const Shape predarray = ShapeUtil::MakeShape(PRED, {64, 48});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kSelect, predarray,
                                          matrix_64_48_, matrix_64_48_);
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_64_48_, *inferred_shape));
}

TEST_F(ShapeInferenceTest, SelectBadShapes) {
  const absl::StatusOr<Shape> inferred_shape_error1 =
      ShapeInference::InferTernaryOpShape(HloOpcode::kSelect, pred_,
                                          matrix_64_48_, matrix_32_64_);
  ASSERT_FALSE(inferred_shape_error1.ok());
  ASSERT_THAT(inferred_shape_error1.status().message(),
              HasSubstr("Operands to select must be the same shape"));

  const absl::StatusOr<Shape> inferred_shape_error2 =
      ShapeInference::InferTernaryOpShape(HloOpcode::kSelect, s32_,
                                          matrix_64_48_, matrix_64_48_);
  ASSERT_FALSE(inferred_shape_error2.ok());
  ASSERT_THAT(inferred_shape_error2.status().message(),
              HasSubstr("pred operand must have PRED"));

  const absl::StatusOr<Shape> inferred_shape_error3 =
      ShapeInference::InferTernaryOpShape(HloOpcode::kSelect,
                                          ShapeUtil::MakeShape(PRED, {64}),
                                          matrix_64_48_, matrix_64_48_);
  ASSERT_FALSE(inferred_shape_error3.ok());
  ASSERT_THAT(
      inferred_shape_error3.status().message(),
      HasSubstr("Operands to select and predicate must be the same shape"));

  // Tuples have a TUPLE element type and cannot be the pred of a select.
  const absl::StatusOr<Shape> inferred_shape_error4 =
      ShapeInference::InferTernaryOpShape(
          HloOpcode::kSelect, ShapeUtil::MakeTupleShape({pred_, pred_}),
          ShapeUtil::MakeTupleShape({u32_, u32_}),
          ShapeUtil::MakeTupleShape({u32_, u32_}));
  ASSERT_FALSE(inferred_shape_error4.ok());
  ASSERT_THAT(inferred_shape_error4.status().message(),
              HasSubstr("Expected array argument for select pred"));
}

TEST_F(ShapeInferenceTest, SelectPreservesElementSize) {
  Shape pred_shape = ShapeUtil::MakeShape(PRED, {10});
  Shape int4_shape = ShapeUtil::MakeShape(S4, {10});
  int4_shape.mutable_layout()->set_element_size_in_bits(4);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kSelect, pred_shape,
                                          int4_shape, int4_shape);
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape, int4_shape));
}

TEST_F(ShapeInferenceTest, ClampAllMatrix) {
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, matrix_64_48_,
                                          matrix_64_48_, matrix_64_48_);
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_64_48_, *inferred_shape));
}

TEST_F(ShapeInferenceTest, ClampAllScalar) {
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, u32_, u32_, u32_);
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(u32_, *inferred_shape));
}

TEST_F(ShapeInferenceTest, ClampMinScalar) {
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, u32_,
                                          matrix_64_48_, matrix_64_48_);
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_64_48_, *inferred_shape));
}

TEST_F(ShapeInferenceTest, ClampMaxScalar) {
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, matrix_64_48_,
                                          matrix_64_48_, u32_);
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_64_48_, *inferred_shape));
}

TEST_F(ShapeInferenceTest, ClampOperandScalar) {
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, matrix_64_48_,
                                          u32_, matrix_64_48_);
  ASSERT_FALSE(inferred_shape.ok());
  ASSERT_THAT(inferred_shape.status().message(),
              HasSubstr("Clamp with incompatible shapes"));
}

TEST_F(ShapeInferenceTest, ClampMinMatrix) {
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, matrix_64_48_,
                                          u32_, u32_);
  ASSERT_FALSE(inferred_shape.ok());
  ASSERT_THAT(inferred_shape.status().message(),
              HasSubstr("Clamp with incompatible shapes"));
}

TEST_F(ShapeInferenceTest, ClampMaxMatrix) {
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, u32_, u32_,
                                          matrix_64_48_);
  ASSERT_FALSE(inferred_shape.ok());
  ASSERT_THAT(inferred_shape.status().message(),
              HasSubstr("Clamp with incompatible shapes"));
}

TEST_F(ShapeInferenceTest, ClampOperandMatrix) {
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, u32_,
                                          matrix_64_48_, u32_);
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_64_48_, *inferred_shape));
}

TEST_F(ShapeInferenceTest, ClampBadShapes) {
  // Type mismatch
  ASSERT_FALSE(
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, s32_, u32_, u32_)
          .ok());
  ASSERT_FALSE(
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, u32_, s32_, u32_)
          .ok());
  ASSERT_FALSE(
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, u32_, u32_, s32_)
          .ok());
  // Dimension mismatch
  ASSERT_FALSE(ShapeInference::InferTernaryOpShape(
                   HloOpcode::kClamp, vector_64_, vector_32_, vector_32_)
                   .ok());
  ASSERT_FALSE(ShapeInference::InferTernaryOpShape(
                   HloOpcode::kClamp, vector_32_, vector_64_, vector_32_)
                   .ok());
  ASSERT_FALSE(ShapeInference::InferTernaryOpShape(
                   HloOpcode::kClamp, vector_32_, vector_32_, vector_64_)
                   .ok());
  // Dimension mismatch, where one operand is a scalar
  ASSERT_FALSE(ShapeInference::InferTernaryOpShape(HloOpcode::kClamp,
                                                   vector_64_, vector_32_, u32_)
                   .ok());
  ASSERT_FALSE(ShapeInference::InferTernaryOpShape(HloOpcode::kClamp,
                                                   vector_64_, u32_, vector_32_)
                   .ok());
  ASSERT_FALSE(ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, u32_,
                                                   vector_64_, vector_32_)
                   .ok());
}

TEST_F(ShapeInferenceTest, VariadicOpTuplify) {
  const absl::StatusOr<Shape> result =
      ShapeInference::InferVariadicOpShape(HloOpcode::kTuple, {&s32_, &u32_});
  ASSERT_TRUE(result.status().ok());
  ASSERT_TRUE(
      ShapeUtil::Equal(*result, ShapeUtil::MakeTupleShape({s32_, u32_})));
}

TEST_F(ShapeInferenceTest, AllGatherStart) {
  const Shape operand = ShapeUtil::MakeShape(U32, {1, 8, 4});
  const Shape expected_shape = ShapeUtil::MakeTupleShape(
      {operand, ShapeUtil::MakeShape(U32, {8, 8, 4})});

  const absl::StatusOr<Shape> inferred_ag_shape =
      ShapeInference::InferAllGatherStartShape(
          {&operand}, /*all_gather_dimension=*/0, /*shard_count=*/8);
  EXPECT_TRUE(inferred_ag_shape.ok());
  EXPECT_TRUE(ShapeUtil::Equal(*inferred_ag_shape, expected_shape));
}

TEST_F(ShapeInferenceTest, AllGatherStartMultiOperand) {
  const Shape operand0 = ShapeUtil::MakeShape(U32, {1, 8, 4});
  const Shape operand1 = ShapeUtil::MakeShape(U16, {1, 5});
  const Shape expected_output0_shape = ShapeUtil::MakeShape(U32, {8, 8, 4});
  const Shape expected_output1_shape = ShapeUtil::MakeShape(U16, {8, 5});
  const Shape expected_shape = ShapeUtil::MakeTupleShape(
      {/* tuple of all input shapes*/
       ShapeUtil::MakeTupleShape({operand0, operand1}),
       /* tuple of all output shapes*/
       ShapeUtil::MakeTupleShape(
           {expected_output0_shape, expected_output1_shape})});

  const absl::StatusOr<Shape> inferred_ag_shape =
      ShapeInference::InferAllGatherStartShape({&operand0, &operand1},
                                               /*all_gather_dimension=*/0,
                                               /*shard_count=*/8);
  EXPECT_TRUE(inferred_ag_shape.ok());
  EXPECT_TRUE(ShapeUtil::Equal(*inferred_ag_shape, expected_shape));
}

TEST_F(ShapeInferenceTest, AllGatherDone) {
  const Shape input_shape =
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(U32, {1, 8, 4}),
                                 ShapeUtil::MakeShape(U32, {8, 8, 4})});
  const Shape expected_shape = ShapeUtil::MakeShape(U32, {8, 8, 4});

  const absl::StatusOr<Shape> inferred_ag_done_shape =
      ShapeInference::InferAllGatherDoneShape(input_shape);
  EXPECT_TRUE(inferred_ag_done_shape.ok());
  EXPECT_TRUE(ShapeUtil::Equal(*inferred_ag_done_shape, expected_shape));
}

TEST_F(ShapeInferenceTest, AllGatherDoneMultiOperand) {
  const Shape operand0 = ShapeUtil::MakeShape(U32, {1, 8, 4});
  const Shape operand1 = ShapeUtil::MakeShape(U16, {1, 5});
  const Shape expected_output0_shape = ShapeUtil::MakeShape(U32, {8, 8, 4});
  const Shape expected_output1_shape = ShapeUtil::MakeShape(U16, {8, 5});
  const Shape input_shape = ShapeUtil::MakeTupleShape(
      {/* tuple of all input shapes*/
       ShapeUtil::MakeTupleShape({operand0, operand1}),
       /* tuple of all output shapes*/
       ShapeUtil::MakeTupleShape(
           {expected_output0_shape, expected_output1_shape})});

  const Shape expected_shape = ShapeUtil::MakeTupleShape(
      {expected_output0_shape, expected_output1_shape});

  const absl::StatusOr<Shape> inferred_ag_done_shape =
      ShapeInference::InferAllGatherDoneShape(input_shape);
  EXPECT_TRUE(inferred_ag_done_shape.ok());
  EXPECT_TRUE(ShapeUtil::Equal(*inferred_ag_done_shape, expected_shape));
}

TEST_F(ShapeInferenceTest, MapThatChangesElementType) {
  const Shape arg = ShapeUtil::MakeShape(U32, {20});
  ProgramShape to_apply = ShapeUtil::MakeProgramShape({u32_}, s32_);
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferMapShape({&arg}, to_apply, {0});
  EXPECT_TRUE(inferred_shape.status().ok());
  const Shape expected = ShapeUtil::MakeShape(S32, {20});
  EXPECT_TRUE(ShapeUtil::Equal(expected, *inferred_shape));
}

TEST_F(ShapeInferenceTest, Map) {
  const absl::StatusOr<Shape> inferred_shape_r1u32 =
      ShapeInference::InferMapShape(
          {&vector_32_, &vector_32_},
          ShapeUtil::MakeProgramShape({u32_, u32_}, u32_), {0});
  EXPECT_TRUE(inferred_shape_r1u32.status().ok());
  EXPECT_TRUE(ShapeUtil::Equal(vector_32_, *inferred_shape_r1u32));

  // It's OK to provide a single argument, as long as the applied arity matches
  // (this degenerates to a Map).
  const absl::StatusOr<Shape> inferred_shape_r1u32_one =
      ShapeInference::InferMapShape(
          {&vector_32_}, ShapeUtil::MakeProgramShape({u32_}, u32_), {0});
  EXPECT_TRUE(inferred_shape_r1u32_one.status().ok());
  EXPECT_TRUE(ShapeUtil::Equal(vector_32_, *inferred_shape_r1u32_one));

  const absl::StatusOr<Shape> inferred_shape_r2s32 =
      ShapeInference::InferMapShape(
          {&s32matrix_64_64_, &s32matrix_64_64_, &s32matrix_64_64_},
          ShapeUtil::MakeProgramShape({s32_, s32_, s32_}, s32_), {0, 1});
  EXPECT_TRUE(inferred_shape_r2s32.status().ok());
  EXPECT_TRUE(ShapeUtil::Equal(s32matrix_64_64_, *inferred_shape_r2s32));

  const auto no_args_error = ShapeInference::InferMapShape(
      {}, ShapeUtil::MakeProgramShape({u32_, u32_}, u32_), {});
  ASSERT_FALSE(no_args_error.ok());
  ASSERT_THAT(no_args_error.status().message(),
              HasSubstr("expects at least one argument"));

  const auto args_diff_shapes_error = ShapeInference::InferMapShape(
      {&vector_32_, &vector_64_},
      ShapeUtil::MakeProgramShape({u32_, u32_}, u32_), {0});
  ASSERT_FALSE(args_diff_shapes_error.ok());
  ASSERT_THAT(args_diff_shapes_error.status().message(),
              HasSubstr("requires all operands to have the same shape"));

  const auto arity_error = ShapeInference::InferMapShape(
      {&vector_32_, &vector_32_}, ShapeUtil::MakeProgramShape({u32_}, u32_),
      {0});
  ASSERT_FALSE(arity_error.ok());
  ASSERT_THAT(arity_error.status().message(),
              HasSubstr("function arity must match"));

  const auto output_shape_error = ShapeInference::InferMapShape(
      {&vector_32_, &vector_32_},
      ShapeUtil::MakeProgramShape({u32_, u32_}, vector_32_), {0});
  ASSERT_FALSE(output_shape_error.ok());
  ASSERT_THAT(output_shape_error.status().message(),
              HasSubstr("result has to be a scalar"));

  const auto param_shape_error = ShapeInference::InferMapShape(
      {&vector_32_, &vector_32_},
      ShapeUtil::MakeProgramShape({vector_32_, u32_}, u32_), {0});
  ASSERT_FALSE(param_shape_error.ok());
  ASSERT_THAT(param_shape_error.status().message(),
              HasSubstr("parameter has to be a scalar"));

  const auto param_element_type_error = ShapeInference::InferMapShape(
      {&vector_32_, &vector_32_},
      ShapeUtil::MakeProgramShape({u32_, s32_}, u32_), {0});
  ASSERT_FALSE(param_element_type_error.ok());
  ASSERT_THAT(param_element_type_error.status().message(),
              HasSubstr("parameter type has to match argument"));

  const Shape arg = ShapeUtil::MakeShape(U32, {20});
  ProgramShape to_apply = ShapeUtil::MakeProgramShape({u32_}, u32_);
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferMapShape({&arg}, to_apply, {0});
  EXPECT_TRUE(inferred_shape.status().ok());
  EXPECT_TRUE(ShapeUtil::Equal(arg, *inferred_shape));

  const absl::StatusOr<Shape> inferred_shape_error1 =
      ShapeInference::InferMapShape(
          {&arg}, ShapeUtil::MakeProgramShape({u32_, u32_}, u32_), {0});
  ASSERT_FALSE(inferred_shape_error1.ok());
  ASSERT_THAT(inferred_shape_error1.status().message(),
              HasSubstr("arity must match number of arguments"));

  const absl::StatusOr<Shape> inferred_shape_error2 =
      ShapeInference::InferMapShape(
          {&arg}, ShapeUtil::MakeProgramShape({vector_32_}, u32_), {0});
  ASSERT_FALSE(inferred_shape_error2.ok());
  ASSERT_THAT(inferred_shape_error2.status().message(),
              HasSubstr("has to be a scalar"));

  const absl::StatusOr<Shape> inferred_shape_error3 =
      ShapeInference::InferMapShape(
          {&arg}, ShapeUtil::MakeProgramShape({u32_}, vector_32_), {0});
  ASSERT_FALSE(inferred_shape_error3.ok());
  ASSERT_THAT(inferred_shape_error3.status().message(),
              HasSubstr("has to be a scalar"));

  const absl::StatusOr<Shape> inferred_shape_error5 =
      ShapeInference::InferMapShape(
          {&arg}, ShapeUtil::MakeProgramShape({s32_}, s32_), {0});
  ASSERT_FALSE(inferred_shape_error5.ok());
  ASSERT_THAT(inferred_shape_error5.status().message(),
              HasSubstr("parameter type has to match argument"));
}

TEST_F(ShapeInferenceTest, MapWithDifferentInputTypes) {
  const Shape arg0 = ShapeUtil::MakeShape(U32, {20});
  const Shape arg1 = ShapeUtil::MakeShape(S32, {20});
  ProgramShape to_apply = ShapeUtil::MakeProgramShape({u32_, s32_}, s32_);
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferMapShape({&arg0, &arg1}, to_apply, {0});
  EXPECT_TRUE(inferred_shape.status().ok());
  const Shape expected = ShapeUtil::MakeShape(S32, {20});
  EXPECT_TRUE(ShapeUtil::Equal(expected, *inferred_shape));
}

TEST_F(ReduceShapeInferenceTest, ReduceVectorToScalar) {
  ExpectInferredReduceShape(u32_, ShapeUtil::MakeShape(U32, {128}),
                            /*dimensions_to_reduce=*/{0});
}

TEST_F(ReduceShapeInferenceTest, ReduceCubeAmongFirstDimension) {
  ExpectInferredReduceShape(ShapeUtil::MakeShape(U32, {3, 4}),
                            ShapeUtil::MakeShape(U32, {2, 3, 4}),
                            /*dimensions_to_reduce=*/{0});
}

TEST_F(ReduceShapeInferenceTest, ReduceCubeAmongMiddleDimension) {
  ExpectInferredReduceShape(ShapeUtil::MakeShape(U32, {2, 4}),
                            ShapeUtil::MakeShape(U32, {2, 3, 4}),
                            /*dimensions_to_reduce=*/{1});
}

TEST_F(ReduceShapeInferenceTest, ReduceCubeAmongFirstTwoDimensions) {
  ExpectInferredReduceShape(ShapeUtil::MakeShape(U32, {4}),
                            ShapeUtil::MakeShape(U32, {2, 3, 4}),
                            /*dimensions_to_reduce=*/{0, 1});
}

TEST_F(ReduceShapeInferenceTest, ReduceCubeAmongLastTwoDimensions) {
  ExpectInferredReduceShape(ShapeUtil::MakeShape(U32, {2}),
                            ShapeUtil::MakeShape(U32, {2, 3, 4}),
                            /*dimensions_to_reduce=*/{1, 2});
}

TEST_F(ReduceShapeInferenceTest, ReduceCubeAmongFirstAndLastDimensions) {
  ExpectInferredReduceShape(ShapeUtil::MakeShape(U32, {3}),
                            ShapeUtil::MakeShape(U32, {2, 3, 4}),
                            /*dimensions_to_reduce=*/{0, 2});

  // Check that the order of dimensions_to_reduce doesn't matter.
  ExpectInferredReduceShape(ShapeUtil::MakeShape(U32, {3}),
                            ShapeUtil::MakeShape(U32, {2, 3, 4}),
                            /*dimensions_to_reduce=*/{2, 0});
}

TEST_F(ReduceShapeInferenceTest, ReduceCubeAmongAllDimensions) {
  ExpectInferredReduceShape(u32_, ShapeUtil::MakeShape(U32, {2, 3, 4}),
                            /*dimensions_to_reduce=*/{0, 1, 2});
}

TEST_F(ReduceShapeInferenceTest, ReduceMultiOutput) {
  const Shape u32_arg_shape = ShapeUtil::MakeShape(U32, {5, 3});
  const Shape s32_arg_shape = ShapeUtil::MakeShape(S32, {5, 3});
  ProgramShape to_apply = ShapeUtil::MakeProgramShape(
      {u32_, s32_, u32_, s32_}, ShapeUtil::MakeTupleShape({u32_, s32_}));
  const absl::StatusOr<Shape> inferred_shape = ShapeInference::InferReduceShape(
      {&u32_arg_shape, &s32_arg_shape, &u32_, &s32_}, {0, 1}, to_apply);
  EXPECT_TRUE(inferred_shape.status().ok());
  EXPECT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeTupleShape({u32_, s32_}),
                               *inferred_shape));
}

TEST_F(ShapeInferenceTest, InferSliceShapeRank2) {
  const Shape matrix_shape = ShapeUtil::MakeShape(U32, {128, 64});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferSliceShape(matrix_shape, {32, 0}, {64, 64}, {1, 1});
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(U32, {32, 64}), *inferred_shape));
}

TEST_F(ShapeInferenceTest, InferSliceWithDynamicDimensions) {
  const Shape matrix_shape = ShapeUtil::MakeShape(U32, {128, 64}, {true, true});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferSliceShape(matrix_shape, {32, 0}, {33, 64}, {1, 1});
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(
      ShapeUtil::MakeShape(U32, {1, 64}, {false, true}), *inferred_shape));
}

TEST_F(ShapeInferenceTest, InferSliceShapeRank2WithStrides) {
  const Shape matrix_shape = ShapeUtil::MakeShape(U32, {128, 64});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferSliceShape(matrix_shape, {32, 0}, {64, 64}, {2, 4});
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(U32, {16, 16}), *inferred_shape));
}

TEST_F(ShapeInferenceTest, InferSliceShapeRank2WithStridesNotIntegral) {
  const Shape matrix_shape = ShapeUtil::MakeShape(U32, {128, 64});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferSliceShape(matrix_shape, {15, 0}, {20, 13}, {2, 4});
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(U32, {3, 4}), *inferred_shape));
}

TEST_F(ShapeInferenceTest, InferInvalidStride) {
  const Shape matrix_shape = ShapeUtil::MakeShape(U32, {128, 64});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferSliceShape(matrix_shape, {127, 0}, {129, 2}, {0, 1});
  ASSERT_FALSE(inferred_shape.ok());
  ASSERT_EQ(absl::StatusCode::kInvalidArgument, inferred_shape.status().code());
}

TEST_F(ShapeInferenceTest, InferOobSliceShapeRank2) {
  const Shape matrix_shape = ShapeUtil::MakeShape(U32, {128, 64});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferSliceShape(matrix_shape, {127, 0}, {129, 2}, {1, 1});
  ASSERT_FALSE(inferred_shape.ok());
  ASSERT_EQ(absl::StatusCode::kInvalidArgument, inferred_shape.status().code());
}

TEST_F(ShapeInferenceTest, InferSliceShapeRank1) {
  const Shape vector_shape = ShapeUtil::MakeShape(U32, {17});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferSliceShape(vector_shape, {2}, {4}, {1});
  ASSERT_TRUE(inferred_shape.ok());
  ASSERT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(U32, {2}), *inferred_shape));
}

TEST_F(ShapeInferenceTest, InferConstIndexShape) {
  const Shape tuple_shape = ShapeUtil::MakeTupleShape({u32_, s32_});
  const absl::StatusOr<Shape> inferred0_status =
      ShapeInference::InferGetTupleElementShape(tuple_shape, 0);
  const absl::StatusOr<Shape> inferred1_status =
      ShapeInference::InferGetTupleElementShape(tuple_shape, 1);
  ASSERT_TRUE(inferred0_status.status().ok());
  ASSERT_TRUE(inferred1_status.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(u32_, *inferred0_status));
  ASSERT_TRUE(ShapeUtil::Equal(s32_, *inferred1_status));
}

TEST_F(ShapeInferenceTest, InferTupleElementShapeOutOfBound) {
  const Shape tuple_shape = ShapeUtil::MakeTupleShape({u32_, s32_});
  const absl::StatusOr<Shape> inferredNegative_status =
      ShapeInference::InferGetTupleElementShape(tuple_shape, -1);
  const absl::StatusOr<Shape> inferred2_status =
      ShapeInference::InferGetTupleElementShape(tuple_shape, 2);
  ASSERT_FALSE(inferredNegative_status.ok());
  ASSERT_FALSE(inferred2_status.ok());
  EXPECT_THAT(inferredNegative_status.status().message(),
              HasSubstr("attempt to index out of tuple bounds"));
  EXPECT_THAT(inferred2_status.status().message(),
              HasSubstr("attempt to index out of tuple bounds"));
}

TEST_F(ShapeInferenceTest, InferPowShape) {
  const Shape ten_u32s = ShapeUtil::MakeShape(U32, {10});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kPower, ten_u32s, u32_, {});
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(ten_u32s, *inferred_shape));
}

TEST_F(ShapeInferenceTest, InferCompareShape) {
  const Shape ten_u32s = ShapeUtil::MakeShape(U32, {10});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kCompare, ten_u32s, u32_,
                                         {});
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(PRED, {10}), *inferred_shape));
}

TEST_F(ShapeInferenceTest, InferReshapeDegenerateCombine) {
  // [1, <=1]
  //   | reshape
  // [<=1]
  //
  // Both output dimension can be dynamic, use inferred_dimension to tie-break.
  const Shape operand = ShapeUtil::MakeShape(U32, {1, 1}, {false, true});
  const auto status =
      ShapeInference::InferReshapeShape(operand, {1, 0}, {1},
                                        /*inferred_dimension=*/-1);
  ASSERT_EQ(ShapeUtil::MakeShape(U32, {1}, {true}), *status);
}

TEST_F(ShapeInferenceTest, InferReshapeSplit) {
  // [<=10]
  //   | reshape
  // [1, 10]
  //
  // Both output dimension can be dynamic, use inferred_dimension to tie-break.
  const Shape operand = ShapeUtil::MakeShape(U32, {10}, {true});
  const auto status =
      ShapeInference::InferReshapeShape(operand, {0}, {1, 10},
                                        /*inferred_dimension=*/0);
  ASSERT_EQ(ShapeUtil::MakeShape(U32, {1, 10}, {true, false}), *status);
}

TEST_F(ShapeInferenceTest, InferReshapeCombine) {
  // [6, <=10]
  //   | reshape
  // [<=60]
  const Shape operand = ShapeUtil::MakeShape(U32, {6, 10}, {false, true});
  const auto status =
      ShapeInference::InferReshapeShape(operand, {1, 0}, {60},
                                        /*inferred_dimension=*/-11);
  ASSERT_EQ(ShapeUtil::MakeShape(U32, {60}, {true}), *status);
}

TEST_F(ShapeInferenceTest, UnchangedDimension) {
  // [6, <=10]
  //   | reshape
  // [2, 3, <=10]
  const Shape operand = ShapeUtil::MakeShape(U32, {6, 10}, {false, true});
  const auto status =
      ShapeInference::InferReshapeShape(operand, {1, 0}, {2, 3, 10},
                                        /*inferred_dimension=*/-11);
  ASSERT_EQ(ShapeUtil::MakeShape(U32, {2, 3, 10}, {false, false, true}),
            *status);
}

TEST_F(ShapeInferenceTest, InferDynamicBroadcast) {
  // CHECK:
  // %broadcast = s32[15,<=15]{1,0} broadcast(s32[<=15]{0}), dimensions={1}

  const Shape operand_shape = ShapeUtil::MakeShape(U32, {15}, {true});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBroadcastShape(operand_shape, {15});
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_EQ(ShapeUtil::MakeShape(U32, {15, 15}, {false, true}),
            *inferred_shape);
}

TEST_F(ShapeInferenceTest, BroadcastScalar) {
  for (auto element_type : {U32, U32, S8}) {
    const Shape scalar_shape = ShapeUtil::MakeShape(element_type, {});
    {  // no-op scalar broadcast
      const auto status = ShapeInference::InferBroadcastShape(scalar_shape, {});
      ASSERT_TRUE(status.status().ok());
      ASSERT_TRUE(ShapeUtil::Equal(scalar_shape, *status));
    }
    const Shape oned_shape = ShapeUtil::MakeShape(element_type, {3});
    {  // scalar -> 1d broadcast
      const auto status =
          ShapeInference::InferBroadcastShape(scalar_shape, {3});
      ASSERT_TRUE(status.status().ok());
      ASSERT_TRUE(ShapeUtil::Equal(oned_shape, *status));
    }
    {  // no-op 1d broadcast
      const auto status = ShapeInference::InferBroadcastShape(oned_shape, {});
      ASSERT_TRUE(status.status().ok());
      ASSERT_TRUE(ShapeUtil::Equal(oned_shape, *status));
    }
    const Shape twod_shape = ShapeUtil::MakeShape(element_type, {2, 3});
    {  // scalar -> 2d broadcast
      const auto status =
          ShapeInference::InferBroadcastShape(scalar_shape, {2, 3});
      ASSERT_TRUE(status.status().ok());
      ASSERT_TRUE(ShapeUtil::Equal(twod_shape, *status));
    }
    {  // 1d -> 2d broadcast
      const auto status = ShapeInference::InferBroadcastShape(oned_shape, {2});
      ASSERT_TRUE(status.status().ok());
      ASSERT_TRUE(ShapeUtil::Equal(twod_shape, *status));
    }
  }
}

// scalar <dot> vector: ok
TEST_F(ShapeInferenceTest, ScalarDotVector) {
  DotDimensionNumbers dot_dnums;
  const absl::StatusOr<Shape> inferred_shape = ShapeInference::InferDotOpShape(
      u32_, vector_32_, dot_dnums, /*preferred_element_type=*/std::nullopt);
  EXPECT_TRUE(inferred_shape.ok());
  EXPECT_EQ(*inferred_shape, vector_32_);
}

// 3D <dot> 2D: error
TEST_F(ShapeInferenceTest, DotWithRankHigherThanTwo) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  const absl::StatusOr<Shape> inferred_shape = ShapeInference::InferDotOpShape(
      ShapeUtil::MakeShape(U32, {32, 32, 32}), matrix_32_64_, dot_dnums,
      /*preferred_element_type=*/std::nullopt);
  EXPECT_TRUE(inferred_shape.ok());
  EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape,
                               ShapeUtil::MakeShape(U32, {32, 32, 64})));
}

// vector <dot> vector -> scalar
TEST_F(ShapeInferenceTest, VectorDotVector) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(0);
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferDotOpShape(vector_64_, vector_64_, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(u32_, *inferred_shape));
  const absl::StatusOr<Shape> inferred_shape_mismatch =
      ShapeInference::InferDotOpShape(vector_64_, vector_32_, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_FALSE(inferred_shape_mismatch.ok());
}

// matrix <dot> vector -> vector
TEST_F(ShapeInferenceTest, MatrixDotVector) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferDotOpShape(matrix_32_64_, vector_64_, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape, vector_32_));
  const absl::StatusOr<Shape> inferred_shape_mismatch =
      ShapeInference::InferDotOpShape(matrix_32_64_, vector_32_, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_FALSE(inferred_shape_mismatch.ok());
}

// vector <dot> matrix -> vector
TEST_F(ShapeInferenceTest, VectorDotMatrix) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(0);
  dot_dnums.add_rhs_contracting_dimensions(0);
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferDotOpShape(vector_32_, matrix_32_64_, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape, vector_64_));
  const absl::StatusOr<Shape> inferred_shape_mismatch =
      ShapeInference::InferDotOpShape(vector_64_, matrix_32_64_, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_FALSE(inferred_shape_mismatch.ok());
}

// matrix <dot> matrix -> matrix
TEST_F(ShapeInferenceTest, MatrixDotMatrix) {
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  const absl::StatusOr<Shape> inferred_shape_match =
      ShapeInference::InferDotOpShape(matrix_32_64_, matrix_64_48_, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_TRUE(inferred_shape_match.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, matrix_32_48_))
      << "inferred: " << ShapeUtil::HumanString(*inferred_shape_match)
      << " expected: " << ShapeUtil::HumanString(matrix_64_48_);
  const absl::StatusOr<Shape> inferred_shape_mismatch =
      ShapeInference::InferDotOpShape(matrix_32_64_, matrix_32_64_, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_FALSE(inferred_shape_mismatch.ok());
}

// BatchMatMul with two batch dimensions and one contracting dimension.
TEST_F(ShapeInferenceTest, DotGeneral) {
  const Shape lhs_shape = ShapeUtil::MakeShape(U32, {5, 2, 11, 3});
  const Shape rhs_shape = ShapeUtil::MakeShape(U32, {5, 2, 3, 14});
  const Shape output_shape = ShapeUtil::MakeShape(U32, {5, 2, 11, 14});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(3);
  dot_dnums.add_lhs_batch_dimensions(0);
  dot_dnums.add_lhs_batch_dimensions(1);

  dot_dnums.add_rhs_contracting_dimensions(2);
  dot_dnums.add_rhs_batch_dimensions(0);
  dot_dnums.add_rhs_batch_dimensions(1);

  const absl::StatusOr<Shape> inferred_shape_match =
      ShapeInference::InferDotOpShape(lhs_shape, rhs_shape, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_TRUE(inferred_shape_match.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, output_shape))
      << "inferred: " << ShapeUtil::HumanString(*inferred_shape_match)
      << " expected: " << ShapeUtil::HumanString(output_shape);
}

// BatchMatMul with two contracting dimensions fails.
TEST_F(ShapeInferenceTest, DotWithTwoContractingDimsFails) {
  const Shape lhs_shape = ShapeUtil::MakeShape(U32, {2, 11, 3, 2});
  const Shape rhs_shape = ShapeUtil::MakeShape(U32, {2, 3, 14});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(2);
  dot_dnums.add_lhs_contracting_dimensions(3);
  dot_dnums.add_lhs_batch_dimensions(0);

  dot_dnums.add_rhs_contracting_dimensions(1);
  dot_dnums.add_rhs_batch_dimensions(0);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferDotOpShape(lhs_shape, rhs_shape, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  ASSERT_FALSE(inferred_shape.ok());
  ASSERT_THAT(inferred_shape.status().message(),
              HasSubstr("Must specify the same number of contracting "
                        "dimensions for lhs and rhs."));
}

TEST_F(ShapeInferenceTest, DotWithTwoContractingDimsPasses) {
  const Shape lhs_shape = ShapeUtil::MakeShape(U32, {2, 11, 3, 2});
  const Shape rhs_shape = ShapeUtil::MakeShape(U32, {2, 3, 2, 14});
  const Shape output_shape = ShapeUtil::MakeShape(U32, {2, 11, 14});

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(2);
  dot_dnums.add_lhs_contracting_dimensions(3);
  dot_dnums.add_lhs_batch_dimensions(0);

  dot_dnums.add_rhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(2);
  dot_dnums.add_rhs_batch_dimensions(0);

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferDotOpShape(lhs_shape, rhs_shape, dot_dnums,
                                      /*preferred_element_type=*/std::nullopt);
  EXPECT_TRUE(inferred_shape.ok());
  EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, output_shape));
}

TEST_F(ShapeInferenceTest, ErrorSetDimensionSize) {
  const Shape arg_shape = ShapeUtil::MakeShape(U32, {5, 3});
  const Shape val_shape = ShapeUtil::MakeShape(S32, {1});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferSetDimensionSizeShape(arg_shape, val_shape,
                                                 /*dimension=*/0);

  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("value has to be S32 scalar"));
}

TEST_F(ShapeInferenceTest, ErrorSetDimensionSizeWrongType) {
  const Shape arg_shape = ShapeUtil::MakeShape(U32, {5, 3});
  const Shape val_shape = ShapeUtil::MakeShape(U32, {});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferSetDimensionSizeShape(arg_shape, val_shape,
                                                 /*dimension=*/0);

  EXPECT_FALSE(inferred_shape.ok());
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("value has to be S32 scalar"));
}

TEST_F(ShapeInferenceTest, BinOpBroadcastMatrixVector) {
  // Test variations of broadcasting a vector for a binary add with a
  // matrix.
  const Shape mat = ShapeUtil::MakeShape(U32, {16, 8});
  const Shape vec8 = ShapeUtil::MakeShape(U32, {8});
  const Shape vec16 = ShapeUtil::MakeShape(U32, {16});

  absl::StatusOr<Shape> inferred_shape_match =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, mat, vec8, {1});
  ASSERT_TRUE(inferred_shape_match.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, mat));

  absl::StatusOr<Shape> inferred_shape_mismatch =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, mat, vec8, {0});
  ASSERT_FALSE(inferred_shape_mismatch.ok());

  inferred_shape_match =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, mat, vec16, {0});
  ASSERT_TRUE(inferred_shape_match.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, mat));

  inferred_shape_mismatch =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, mat, vec16, {1});
  ASSERT_FALSE(inferred_shape_mismatch.ok());
}

TEST_F(ShapeInferenceTest, BinOpBroadcastCubeMatrix) {
  // Test variations of broadcasting a matrix for a binary add with a cube.
  const Shape cube = ShapeUtil::MakeShape(U32, {16, 8, 4});
  const Shape matrix8_4 = ShapeUtil::MakeShape(U32, {8, 4});
  const Shape matrix16_4 = ShapeUtil::MakeShape(U32, {16, 4});
  const Shape matrix16_8 = ShapeUtil::MakeShape(U32, {16, 8});

  absl::StatusOr<Shape> inferred_shape_match =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, cube, matrix8_4,
                                         {1, 2});
  ASSERT_TRUE(inferred_shape_match.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, cube));

  inferred_shape_match = ShapeInference::InferBinaryOpShape(
      HloOpcode::kAdd, cube, matrix16_4, {0, 2});
  ASSERT_TRUE(inferred_shape_match.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, cube));

  inferred_shape_match = ShapeInference::InferBinaryOpShape(
      HloOpcode::kAdd, cube, matrix16_8, {0, 1});
  ASSERT_TRUE(inferred_shape_match.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, cube));
}

TEST_F(ShapeInferenceTest, BinOpBroadcastBadDimension) {
  // Test various errors with the broadcast argument.
  const Shape tensor = ShapeUtil::MakeShape(U32, {16, 8, 4});
  const Shape tensor8_8_8 = ShapeUtil::MakeShape(U32, {8, 8, 8});
  const Shape vec8 = ShapeUtil::MakeShape(U32, {8});
  const Shape matrix8_4 = ShapeUtil::MakeShape(U32, {8, 4});
  const Shape matrix8_8 = ShapeUtil::MakeShape(U32, {8, 8});

  // "magical" broadcast rejected
  const absl::StatusOr<Shape> inferred_shape_error1 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor, vec8, {});
  ASSERT_FALSE(inferred_shape_error1.ok());
  ASSERT_THAT(inferred_shape_error1.status().message(),
              HasSubstr("Shapes must be equal rank"));

  // broadcast_dimension out of bounds for tensor's rank
  const absl::StatusOr<Shape> inferred_shape_error2 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor, vec8, {3});
  ASSERT_FALSE(inferred_shape_error2.ok());
  ASSERT_THAT(inferred_shape_error2.status().message(),
              ContainsRegex("Broadcast dimension number .* too large"));

  // broadcast_dimension doesn't match corresponding dimension
  const absl::StatusOr<Shape> inferred_shape_error3 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor, vec8, {0});
  ASSERT_FALSE(inferred_shape_error3.ok());
  ASSERT_THAT(inferred_shape_error3.status().message(),
              HasSubstr("Broadcast dimension 0 mismatch"));

  // broadcast_dimensions list too long
  const absl::StatusOr<Shape> inferred_shape_error4 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor, matrix8_4,
                                         {0, 1, 2});
  ASSERT_FALSE(inferred_shape_error4.ok());
  ASSERT_THAT(inferred_shape_error4.status().message(),
              HasSubstr("broadcast_dimensions has to match"));

  // there's a dimension above the rank of the tensor
  const absl::StatusOr<Shape> inferred_shape_error5 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor, matrix8_4,
                                         {3, 0});
  ASSERT_FALSE(inferred_shape_error5.ok());
  ASSERT_THAT(inferred_shape_error5.status().message(),
              ContainsRegex("dimension number .* too large"));

  // broadcasting dimensions don't match in this order
  const absl::StatusOr<Shape> inferred_shape_error6 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor, matrix8_4,
                                         {2, 1});
  ASSERT_FALSE(inferred_shape_error6.ok());
  ASSERT_THAT(inferred_shape_error6.status().message(),
              HasSubstr("dimension 0 mismatch"));

  // The following two tests make sure that broadcasting dimensions are listed
  // in a proper (strictly increasing) order, even if the lower-rank array
  // matches the higher-rank array in many different ways.
  const absl::StatusOr<Shape> inferred_shape_error7 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor8_8_8,
                                         matrix8_8, {0, 0});
  ASSERT_FALSE(inferred_shape_error7.ok());
  ASSERT_THAT(inferred_shape_error7.status().message(),
              HasSubstr("dimensions order is wrong"));

  const absl::StatusOr<Shape> inferred_shape_error8 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor8_8_8,
                                         matrix8_8, {1, 0});
  ASSERT_FALSE(inferred_shape_error8.ok());
  ASSERT_THAT(inferred_shape_error8.status().message(),
              HasSubstr("dimensions order is wrong"));
}

// Tests for the while instruction with proper shapes.
TEST_F(ShapeInferenceTest, WhileWithCorrectShapes) {
  const Shape result_shape = ShapeUtil::MakeTupleShape({s32_, vector_32_});
  ProgramShape cond = ShapeUtil::MakeProgramShape({result_shape}, pred_);
  ProgramShape body = ShapeUtil::MakeProgramShape({result_shape}, result_shape);
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferWhileShape(cond, body, result_shape);
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(result_shape, *inferred_shape));
}

// Tests for the while instruction with wrong shapes.
TEST_F(ShapeInferenceTest, WhileWithBadShapes) {
  const Shape inferred_shape = ShapeUtil::MakeTupleShape({s32_, vector_32_});
  ProgramShape cond = ShapeUtil::MakeProgramShape({inferred_shape}, pred_);
  ProgramShape body =
      ShapeUtil::MakeProgramShape({inferred_shape}, inferred_shape);

  const auto bad_shape_1 =
      ShapeUtil::MakeProgramShape({s32_, inferred_shape}, pred_);
  const absl::StatusOr<Shape> inferred_shape_error1 =
      ShapeInference::InferWhileShape(bad_shape_1, body, inferred_shape);
  ASSERT_FALSE(inferred_shape_error1.ok());
  ASSERT_THAT(inferred_shape_error1.status().message(),
              HasSubstr("Condition must take 1 arguments"));

  const auto bad_shape_2 =
      ShapeUtil::MakeProgramShape({s32_, inferred_shape}, inferred_shape);
  const absl::StatusOr<Shape> inferred_shape_error2 =
      ShapeInference::InferWhileShape(cond, bad_shape_2, inferred_shape);
  ASSERT_FALSE(inferred_shape_error2.ok());
  ASSERT_THAT(inferred_shape_error2.status().message(),
              HasSubstr("Body must take 1 arguments"));

  const auto bad_shape_3 = ShapeUtil::MakeProgramShape({inferred_shape}, s32_);
  const absl::StatusOr<Shape> inferred_shape_error3 =
      ShapeInference::InferWhileShape(bad_shape_3, body, inferred_shape);
  ASSERT_FALSE(inferred_shape_error3.ok());
  ASSERT_THAT(inferred_shape_error3.status().message(),
              HasSubstr("Condition must return a boolean"));

  const auto bad_shape_4 =
      ShapeUtil::MakeProgramShape({inferred_shape}, vector_32_);
  const absl::StatusOr<Shape> inferred_shape_error4 =
      ShapeInference::InferWhileShape(cond, bad_shape_4, inferred_shape);
  ASSERT_FALSE(inferred_shape_error4.ok());
  ASSERT_THAT(inferred_shape_error4.status().message(),
              HasSubstr("parameter of condition and body"));
}

// Tests for the concatenate instruction with dynamic shapes.
TEST_F(ShapeInferenceTest, ConcatenateWithDynamicShapes) {
  const auto dynamic_shape_1 =
      ShapeUtil::MakeShape(U32, {32, 160, 10}, {true, false, false});
  const auto dynamic_shape_2 =
      ShapeUtil::MakeShape(U32, {32, 160, 10}, {false, true, false});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferConcatOpShape({&dynamic_shape_1, &dynamic_shape_2},
                                         /*dimension=*/0);
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(
      ShapeUtil::MakeShape(U32, {64, 160, 10}, {true, true, false}),
      *inferred_shape));
}

// Tests for the concatenate instruction with proper shapes.
TEST_F(ShapeInferenceTest, ConcatenateWithCorrectShapes) {
  const absl::StatusOr<Shape> inferred_shape_1 =
      ShapeInference::InferConcatOpShape({&vector_32_, &vector_64_},
                                         /*dimension=*/0);
  ASSERT_TRUE(inferred_shape_1.status().ok());
  ASSERT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(U32, {96}), *inferred_shape_1));

  const absl::StatusOr<Shape> inferred_shape_2 =
      ShapeInference::InferConcatOpShape(
          {&vector_32_, &vector_64_, &vector_32_}, /*dimension=*/0);
  ASSERT_TRUE(inferred_shape_2.status().ok());
  ASSERT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(U32, {128}), *inferred_shape_2));

  const absl::StatusOr<Shape> inferred_shape_3 =
      ShapeInference::InferConcatOpShape(
          {&matrix_32_48_, &matrix_32_64_, &matrix_32_48_}, /*dimension=*/1);
  ASSERT_TRUE(inferred_shape_3.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeShape(U32, {32, 160}),
                               *inferred_shape_3));
}

// Tests for the concatenate instruction with wrong shapes.
TEST_F(ShapeInferenceTest, ConcatenateWithBadShapes) {
  const absl::StatusOr<Shape> inferred_shape_error1 =
      ShapeInference::InferConcatOpShape({}, /*dimension=*/0);
  ASSERT_FALSE(inferred_shape_error1.ok());
  ASSERT_THAT(inferred_shape_error1.status().message(),
              HasSubstr("Concatenate expects at least one argument"));

  const absl::StatusOr<Shape> inferred_shape_error2 =
      ShapeInference::InferConcatOpShape({&vector_32_}, /*dimension=*/-1);
  ASSERT_FALSE(inferred_shape_error2.ok());
  ASSERT_THAT(inferred_shape_error2.status().message(),
              HasSubstr("dimension out of bounds: -1"));

  const absl::StatusOr<Shape> inferred_shape_error3 =
      ShapeInference::InferConcatOpShape({&vector_32_}, /*dimension=*/1);
  ASSERT_FALSE(inferred_shape_error3.ok());
  ASSERT_THAT(inferred_shape_error3.status().message(),
              HasSubstr("dimension out of bounds: 1"));

  const Shape tuple = ShapeUtil::MakeTupleShape({vector_32_});
  const absl::StatusOr<Shape> inferred_shape_error4 =
      ShapeInference::InferConcatOpShape({&vector_32_, &tuple},
                                         /*dimension=*/0);
  ASSERT_FALSE(inferred_shape_error4.ok());
  ASSERT_THAT(
      inferred_shape_error4.status().message(),
      HasSubstr("Expected array argument for operand of concatenation"));

  const Shape vector_s32 = ShapeUtil::MakeShape(S32, {32});
  const absl::StatusOr<Shape> inferred_shape_error5 =
      ShapeInference::InferConcatOpShape({&vector_32_, &vector_s32},
                                         /*dimension=*/0);
  ASSERT_FALSE(inferred_shape_error5.ok());
  ASSERT_THAT(inferred_shape_error5.status().message(),
              HasSubstr("concatenate arrays with different element types"));

  const absl::StatusOr<Shape> inferred_shape_error6 =
      ShapeInference::InferConcatOpShape({&matrix_32_48_, &matrix_32_64_},
                                         /*dimension=*/0);
  ASSERT_FALSE(inferred_shape_error6.ok());
  ASSERT_THAT(inferred_shape_error6.status().message(),
              HasSubstr("concatenate arrays that differ in "
                        "dimensions other than the one being "
                        "concatenated"));
}

TEST_F(ShapeInferenceTest, Pad) {
  const Shape input_shape = ShapeUtil::MakeShape(U32, {10, 25});
  const Shape padding_value_shape = ShapeUtil::MakeShape(U32, {});
  PaddingConfig padding_config;
  const auto dimension0 = padding_config.add_dimensions();
  dimension0->set_edge_padding_low(0);
  dimension0->set_edge_padding_high(2);
  // TODO(chokobole): Do we need this? Dependency: interior_padding
  // dimension0->set_interior_padding(3);
  const auto dimension1 = padding_config.add_dimensions();
  dimension1->set_edge_padding_low(1);
  dimension1->set_edge_padding_high(5);
  // TODO(chokobole): Do we need this? Dependency: interior_padding
  // dimension1->set_interior_padding(0);

  const absl::StatusOr<Shape> inferred_shape = ShapeInference::InferPadShape(
      input_shape, padding_value_shape, padding_config);
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(U32, {12, 31}), *inferred_shape));

  dimension1->set_edge_padding_low(-20);
  dimension1->set_edge_padding_high(-10);
  const auto negative_dimension_size = ShapeInference::InferPadShape(
      input_shape, padding_value_shape, padding_config);
  ASSERT_FALSE(negative_dimension_size.ok());
  ASSERT_THAT(negative_dimension_size.status().message(),
              HasSubstr("negative size for dimension 1"));
}

TEST_F(ShapeInferenceTest, Reverse) {
  const Shape input_shape = ShapeUtil::MakeShape(U32, {10, 25});

  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferReverseShape(input_shape, {0, 1});
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(input_shape, *inferred_shape));
}

TEST_F(ShapeInferenceTest, ReverseInvalidDimension) {
  const Shape input_shape = ShapeUtil::MakeShape(U32, {10, 25});

  const absl::StatusOr<Shape> inferred_shape_error0 =
      ShapeInference::InferReverseShape(input_shape, {0, 2});
  ASSERT_FALSE(inferred_shape_error0.ok());
  ASSERT_THAT(inferred_shape_error0.status().message(),
              HasSubstr("out-of-bounds"));

  const absl::StatusOr<Shape> inferred_shape_error1 =
      ShapeInference::InferReverseShape(input_shape, {0, -1});
  ASSERT_FALSE(inferred_shape_error1.ok());
  ASSERT_THAT(inferred_shape_error1.status().message(),
              HasSubstr("out-of-bounds"));

  const absl::StatusOr<Shape> inferred_shape_error2 =
      ShapeInference::InferReverseShape(input_shape, {0, 0});
  ASSERT_FALSE(inferred_shape_error2.ok());
  ASSERT_THAT(inferred_shape_error2.status().message(),
              HasSubstr("duplicated"));

  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({input_shape, input_shape});
  const absl::StatusOr<Shape> inferred_shape_error3 =
      ShapeInference::InferReverseShape(tuple_shape, {0});
  ASSERT_FALSE(inferred_shape_error3.ok());
  ASSERT_THAT(inferred_shape_error3.status().message(),
              HasSubstr("Expected array argument"));
}

TEST_F(ShapeInferenceTest, Call) {
  const absl::StatusOr<Shape> inferred_shape0 =
      ShapeInference::InferCallShape({}, ShapeUtil::MakeProgramShape({}, u32_));
  EXPECT_TRUE(inferred_shape0.status().ok());
  EXPECT_TRUE(ShapeUtil::Equal(u32_, *inferred_shape0));

  const absl::StatusOr<Shape> inferred_shape1 = ShapeInference::InferCallShape(
      {&u32_, &s32_, &pred_, &vector_32_, &matrix_32_48_},
      ShapeUtil::MakeProgramShape(
          {u32_, s32_, pred_, vector_32_, matrix_32_48_}, s32matrix_64_64_));
  EXPECT_TRUE(inferred_shape1.status().ok());
  EXPECT_TRUE(ShapeUtil::Equal(s32matrix_64_64_, *inferred_shape1));

  const absl::StatusOr<Shape> inferred_shape_error0 =
      ShapeInference::InferCallShape({},
                                     ShapeUtil::MakeProgramShape({u32_}, u32_));
  EXPECT_FALSE(inferred_shape_error0.ok());
  EXPECT_THAT(inferred_shape_error0.status().message(),
              HasSubstr("arity must match"));

  const absl::StatusOr<Shape> inferred_shape_error1 =
      ShapeInference::InferCallShape({&u32_},
                                     ShapeUtil::MakeProgramShape({}, u32_));
  EXPECT_FALSE(inferred_shape_error1.ok());
  EXPECT_THAT(inferred_shape_error1.status().message(),
              HasSubstr("arity must match"));

  const absl::StatusOr<Shape> inferred_shape_error2 =
      ShapeInference::InferCallShape({&u32_},
                                     ShapeUtil::MakeProgramShape({s32_}, u32_));
  EXPECT_FALSE(inferred_shape_error2.ok());
  EXPECT_THAT(inferred_shape_error2.status().message(),
              HasSubstr("parameter must match argument"));
}

TEST_F(ShapeInferenceTest, Transpose) {
  const Shape a_shape = ShapeUtil::MakeShape(U32, {2, 3, 4, 5});
  const absl::StatusOr<Shape> inferred_shape_and_status =
      ShapeInference::InferTransposeShape(a_shape, {1, 2, 3, 0});
  EXPECT_TRUE(inferred_shape_and_status.ok());
  EXPECT_TRUE(ShapeUtil::Compatible(ShapeUtil::MakeShape(U32, {3, 4, 5, 2}),
                                    *inferred_shape_and_status));
}

TEST_F(ShapeInferenceTest, Rank1Transpose) {
  const Shape a_shape = ShapeUtil::MakeShape(U32, {5});
  const absl::StatusOr<Shape> inferred_shape_and_status =
      ShapeInference::InferTransposeShape(a_shape, {0});
  EXPECT_TRUE(inferred_shape_and_status.ok());
  EXPECT_TRUE(ShapeUtil::Compatible(ShapeUtil::MakeShape(U32, {5}),
                                    *inferred_shape_and_status));
}

TEST_F(ShapeInferenceTest, ConditionalPred) {
  const absl::StatusOr<Shape> inferred_shape0 =
      ShapeInference::InferConditionalShape(
          pred_,
          {ShapeUtil::MakeProgramShape({vector_32_}, u32_),
           ShapeUtil::MakeProgramShape({vector_64_}, u32_)},
          {vector_32_, vector_64_});
  EXPECT_TRUE(inferred_shape0.status().ok());
  EXPECT_TRUE(ShapeUtil::Equal(u32_, *inferred_shape0));

  const absl::StatusOr<Shape> inferred_shape1 =
      ShapeInference::InferConditionalShape(
          pred_,
          {ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_64_),
           ShapeUtil::MakeProgramShape({vector_32_}, vector_64_)},
          {matrix_32_48_, vector_32_});
  EXPECT_TRUE(inferred_shape1.status().ok());
  EXPECT_TRUE(ShapeUtil::Equal(vector_64_, *inferred_shape1));

  const auto tuple_u32_v32 = ShapeUtil::MakeTupleShape({u32_, vector_32_});
  const absl::StatusOr<Shape> inferred_shape2 =
      ShapeInference::InferConditionalShape(
          pred_,
          {ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_32_),
           ShapeUtil::MakeProgramShape({tuple_u32_v32}, vector_32_)},
          {matrix_32_48_, tuple_u32_v32});
  EXPECT_TRUE(inferred_shape2.status().ok());
  EXPECT_TRUE(ShapeUtil::Equal(vector_32_, *inferred_shape2));

  const absl::StatusOr<Shape> inferred_shape_error0 =
      ShapeInference::InferConditionalShape(
          u32_,
          {ShapeUtil::MakeProgramShape({vector_32_}, u32_),
           ShapeUtil::MakeProgramShape({vector_64_}, u32_)},
          {vector_32_, vector_64_});
  EXPECT_FALSE(inferred_shape_error0.ok());
  EXPECT_THAT(inferred_shape_error0.status().message(),
              HasSubstr("must be bool or int32_t"));

  const absl::StatusOr<Shape> inferred_shape_error1 =
      ShapeInference::InferConditionalShape(
          pred_,
          {ShapeUtil::MakeProgramShape({u32_, vector_32_}, vector_32_),
           ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_32_)},
          {ShapeUtil::MakeTupleShape({u32_, vector_32_}), matrix_32_48_});
  EXPECT_FALSE(inferred_shape_error1.ok());
  EXPECT_THAT(inferred_shape_error1.status().message(),
              HasSubstr("branch computation 0 must take 1 argument"));

  const absl::StatusOr<Shape> inferred_shape_error2 =
      ShapeInference::InferConditionalShape(
          pred_,
          {ShapeUtil::MakeProgramShape({vector_64_}, u32_),
           ShapeUtil::MakeProgramShape({vector_64_}, u32_)},
          {vector_32_, vector_64_});
  EXPECT_FALSE(inferred_shape_error2.ok());
  EXPECT_THAT(inferred_shape_error2.status().message(),
              HasSubstr("branch operand 0 must match the shape of the only "
                        "parameter of branch computation 0"));

  const absl::StatusOr<Shape> inferred_shape_error3 =
      ShapeInference::InferConditionalShape(
          pred_,
          {ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_32_),
           ShapeUtil::MakeProgramShape({u32_, vector_32_}, vector_32_)},
          {matrix_32_48_, ShapeUtil::MakeTupleShape({u32_, vector_32_})});
  EXPECT_FALSE(inferred_shape_error3.ok());
  EXPECT_THAT(inferred_shape_error3.status().message(),
              HasSubstr("branch computation 1 must take 1 argument"));

  const absl::StatusOr<Shape> inferred_shape_error4 =
      ShapeInference::InferConditionalShape(
          pred_,
          {ShapeUtil::MakeProgramShape({vector_32_}, u32_),
           ShapeUtil::MakeProgramShape({vector_32_}, u32_)},
          {vector_32_, vector_64_});
  EXPECT_FALSE(inferred_shape_error4.ok());
  EXPECT_THAT(inferred_shape_error4.status().message(),
              HasSubstr("branch operand 1 must match the shape of the only "
                        "parameter of branch computation 1"));

  const absl::StatusOr<Shape> inferred_shape_error5 =
      ShapeInference::InferConditionalShape(
          pred_,
          {ShapeUtil::MakeProgramShape({vector_32_}, u32_),
           ShapeUtil::MakeProgramShape({vector_64_}, vector_32_)},
          {vector_32_, vector_64_});
  EXPECT_FALSE(inferred_shape_error5.ok());
  EXPECT_THAT(inferred_shape_error5.status().message(),
              HasSubstr("the result of branch 0 computation and branch 1 "
                        "computation must have the same shape"));
}

TEST_F(ShapeInferenceTest, ConditionalIndexed) {
  const Shape r0s32 = ShapeUtil::MakeShape(S32, {});
  const absl::StatusOr<Shape> inferred_shape0 =
      ShapeInference::InferConditionalShape(
          r0s32,
          {ShapeUtil::MakeProgramShape({vector_32_}, u32_),
           ShapeUtil::MakeProgramShape({vector_64_}, u32_),
           ShapeUtil::MakeProgramShape({vector_64_}, u32_)},
          {vector_32_, vector_64_, vector_64_});
  EXPECT_TRUE(inferred_shape0.status().ok());
  EXPECT_TRUE(ShapeUtil::Equal(u32_, *inferred_shape0));

  const absl::StatusOr<Shape> inferred_shape1 =
      ShapeInference::InferConditionalShape(
          r0s32,
          {ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_64_),
           ShapeUtil::MakeProgramShape({vector_32_}, vector_64_),
           ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_64_)},
          {matrix_32_48_, vector_32_, matrix_32_48_});
  EXPECT_TRUE(inferred_shape1.status().ok());
  EXPECT_TRUE(ShapeUtil::Equal(vector_64_, *inferred_shape1));

  const auto tuple_u32_v32 = ShapeUtil::MakeTupleShape({u32_, vector_32_});
  const absl::StatusOr<Shape> inferred_shape2 =
      ShapeInference::InferConditionalShape(
          r0s32, {ShapeUtil::MakeProgramShape({tuple_u32_v32}, vector_32_)},
          {tuple_u32_v32});
  EXPECT_TRUE(inferred_shape2.status().ok());
  EXPECT_TRUE(ShapeUtil::Equal(vector_32_, *inferred_shape2));

  const absl::StatusOr<Shape> inferred_shape_error0 =
      ShapeInference::InferConditionalShape(
          pred_,
          {ShapeUtil::MakeProgramShape({vector_32_}, u32_),
           ShapeUtil::MakeProgramShape({vector_32_}, u32_),
           ShapeUtil::MakeProgramShape({vector_64_}, u32_)},
          {vector_32_, vector_32_, vector_64_});
  EXPECT_FALSE(inferred_shape_error0.ok());
  EXPECT_THAT(inferred_shape_error0.status().message(),
              HasSubstr("2 == branch_computations.size()"));

  const absl::StatusOr<Shape> inferred_shape_error1 =
      ShapeInference::InferConditionalShape(
          r0s32,
          {ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_32_),
           ShapeUtil::MakeProgramShape({u32_, vector_32_}, vector_32_),
           ShapeUtil::MakeProgramShape({matrix_32_48_}, vector_32_)},
          {matrix_32_48_, ShapeUtil::MakeTupleShape({u32_, vector_32_}),
           matrix_32_48_});
  EXPECT_FALSE(inferred_shape_error1.ok());
  EXPECT_THAT(inferred_shape_error1.status().message(),
              HasSubstr("branch computation 1 must take 1 argument"));

  const absl::StatusOr<Shape> inferred_shape_error2 =
      ShapeInference::InferConditionalShape(
          r0s32,
          {ShapeUtil::MakeProgramShape({r0s32}, u32_),
           ShapeUtil::MakeProgramShape({vector_32_}, u32_),
           ShapeUtil::MakeProgramShape({vector_32_}, u32_)},
          {r0s32, vector_32_, vector_64_});
  EXPECT_FALSE(inferred_shape_error2.ok());
  EXPECT_THAT(inferred_shape_error2.status().message(),
              HasSubstr("branch operand 2 must match the shape of the only "
                        "parameter of branch computation 2"));

  const absl::StatusOr<Shape> inferred_shape_error3 =
      ShapeInference::InferConditionalShape(
          r0s32,
          {ShapeUtil::MakeProgramShape({vector_32_}, u32_),
           ShapeUtil::MakeProgramShape({vector_32_}, u32_),
           ShapeUtil::MakeProgramShape({vector_32_}, u32_),
           ShapeUtil::MakeProgramShape({vector_64_}, vector_32_)},
          {vector_32_, vector_32_, vector_32_, vector_64_});
  EXPECT_FALSE(inferred_shape_error3.ok());
  EXPECT_THAT(inferred_shape_error3.status().message(),
              HasSubstr("the result of branch 0 computation and branch 3 "
                        "computation must have the same shape"));

  const absl::StatusOr<Shape> inferred_shape_error4 =
      ShapeInference::InferConditionalShape(r0s32, {}, {});
  EXPECT_FALSE(inferred_shape_error4.ok());
  EXPECT_THAT(inferred_shape_error4.status().message(),
              HasSubstr("!branch_computations.empty()"));
}

TEST_F(ShapeInferenceTest, ConditionalDynamic) {
  const Shape r0s32 = ShapeUtil::MakeShape(S32, {});
  const Shape static_shape = ShapeUtil::MakeShape(S32, {4}, {false});
  const Shape dynamic_shape = ShapeUtil::MakeShape(S32, {4}, {true});
  const absl::StatusOr<Shape> inferred_shape0 =
      ShapeInference::InferConditionalShape(
          r0s32,
          {ShapeUtil::MakeProgramShape({vector_32_}, static_shape),
           ShapeUtil::MakeProgramShape({vector_64_}, dynamic_shape),
           ShapeUtil::MakeProgramShape({vector_64_}, dynamic_shape)},
          {vector_32_, vector_64_, vector_64_});
  EXPECT_TRUE(inferred_shape0.status().ok());
  EXPECT_TRUE(ShapeUtil::Equal(dynamic_shape, *inferred_shape0));

  const absl::StatusOr<Shape> inferred_shape1 =
      ShapeInference::InferConditionalShape(
          r0s32,
          {ShapeUtil::MakeProgramShape({vector_32_}, dynamic_shape),
           ShapeUtil::MakeProgramShape({vector_64_}, static_shape),
           ShapeUtil::MakeProgramShape({vector_64_}, dynamic_shape)},
          {vector_32_, vector_64_, vector_64_});
  EXPECT_TRUE(inferred_shape1.status().ok());
  EXPECT_TRUE(ShapeUtil::Equal(dynamic_shape, *inferred_shape1));
}

TEST_F(ShapeInferenceTest, BadSlice) {
  const Shape arg = ShapeUtil::MakeShape(U32, {4});
  const absl::StatusOr<Shape> statusor =
      ShapeInference::InferSliceShape(arg, {0}, {5}, {1});
  ASSERT_FALSE(statusor.ok());

  LOG(INFO) << statusor.status();

  EXPECT_THAT(statusor.status().message(),
              HasSubstr("less than or equal to dimension size"))
      << statusor.status();
  EXPECT_THAT(statusor.status().message(), HasSubstr("argument shape"))
      << statusor.status();
}

TEST_F(ShapeInferenceTest, BadSort) {
  const Shape keys = ShapeUtil::MakeShape(U32, {4});
  const Shape values = ShapeUtil::MakeShape(U32, {5});
  const absl::StatusOr<Shape> statusor =
      ShapeInference::InferVariadicOpShape(HloOpcode::kSort, {&keys, &values});
  EXPECT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(), HasSubstr("dimensions must match"))
      << statusor.status();
}

TEST_F(ShapeInferenceTest, BadSortValuesMismatch) {
  const Shape keys = ShapeUtil::MakeShape(U32, {4});
  const Shape values_good = ShapeUtil::MakeShape(U32, {4});
  const Shape values_bad = ShapeUtil::MakeShape(U32, {5});
  const absl::StatusOr<Shape> statusor = ShapeInference::InferVariadicOpShape(
      HloOpcode::kSort, {&keys, &values_good, &values_bad});
  EXPECT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(), HasSubstr("dimensions must match"))
      << statusor.status();
}

TEST_F(ShapeInferenceTest, SortManyValues) {
  const Shape keys = ShapeUtil::MakeShape(U32, {4});
  const Shape values_s32 = ShapeUtil::MakeShape(S32, {4});
  const Shape values_u32 = ShapeUtil::MakeShape(U32, {4});
  const absl::StatusOr<Shape> statusor = ShapeInference::InferVariadicOpShape(
      HloOpcode::kSort, {&keys, &values_s32, &values_u32});
  EXPECT_TRUE(statusor.status().ok());
  const Shape inferred_shape = *statusor;
  EXPECT_TRUE(ShapeUtil::Compatible(
      inferred_shape,
      ShapeUtil::MakeTupleShape({keys, values_s32, values_u32})));
}

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferGatherShape
// clang-format on

class ScatterShapeInferenceTest
    : public ShapeInferenceTest,
      public ::testing::WithParamInterface<std::vector<PrimitiveType>> {
 protected:
  struct ScatterShapes {
    void Add(Shape shape) {
      shapes.push_back(std::move(shape));
      ptrs.push_back(&shapes.back());
    }
    std::vector<Shape> shapes;
    std::vector<const Shape*> ptrs;
  };
  static ScatterShapes CreateShapes(absl::Span<const int64_t> operand_dims,
                                    const Shape& scatter_indices_shape,
                                    absl::Span<const int64_t> update_dims,
                                    absl::Span<const PrimitiveType> types) {
    CHECK(!types.empty());
    size_t size = types.size() * 2 + 1;
    ScatterShapes shapes;
    shapes.shapes.reserve(size);
    shapes.ptrs.reserve(size);
    for (PrimitiveType type : types) {
      shapes.Add(ShapeUtil::MakeShape(type, operand_dims));
    }
    shapes.Add(scatter_indices_shape);
    for (PrimitiveType type : types) {
      shapes.Add(ShapeUtil::MakeShape(type, update_dims));
    }
    return shapes;
  }
  static Shape Collate(absl::Span<const int64_t> dims,
                       absl::Span<const PrimitiveType> types) {
    CHECK(!types.empty());
    if (types.size() == 1) {
      return ShapeUtil::MakeShape(types[0], dims);
    }
    std::vector<Shape> shapes;
    for (PrimitiveType type : types) {
      shapes.push_back(ShapeUtil::MakeShape(type, dims));
    }
    return ShapeUtil::MakeTupleShape(shapes);
  }
  static Shape scalar(PrimitiveType type) {
    return ShapeUtil::MakeShape(type, {});
  }
  static Shape s64_vector(int dim) { return ShapeUtil::MakeShape(S64, {dim}); }
  static Shape s64_tensor(absl::Span<const int64_t> dims) {
    return ShapeUtil::MakeShape(S64, dims);
  }
  static ProgramShape to_apply(absl::Span<const PrimitiveType> types) {
    CHECK(!types.empty());
    ProgramShape program_shape;
    Shape& result = *program_shape.mutable_result();
    result = ShapeUtil::MakeNil();
    result.mutable_tuple_shapes()->reserve(types.size());
    program_shape.mutable_parameters()->reserve(types.size() * 2);
    for (PrimitiveType type : types) {
      *program_shape.add_parameters() = scalar(type);
      *result.add_tuple_shapes() = scalar(type);
    }
    for (PrimitiveType type : types) {
      *program_shape.add_parameters() = scalar(type);
    }
    return program_shape;
  }
  std::vector<PrimitiveType> types() const { return GetParam(); }
};

TEST_P(ScatterShapeInferenceTest, TfScatterWithFullUpdates) {
  const auto shapes = CreateShapes({64, 48}, s64_vector(32), {64, 32}, types());
  TF_ASSERT_OK_AND_ASSIGN(const Shape scatter_shape,
                          ShapeInference::InferScatterShape(
                              shapes.ptrs, to_apply(types()),
                              HloScatterInstruction::MakeScatterDimNumbers(
                                  /*update_window_dims=*/{0},
                                  /*inserted_window_dims=*/{1},
                                  /*scatter_dims_to_operand_dims=*/{1},
                                  /*index_vector_dim=*/1)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, Collate({64, 48}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, TfScatterWithFullUpdatesV2) {
  const auto shapes = CreateShapes({64, 48}, s64_vector(32), {32, 48}, types());
  TF_ASSERT_OK_AND_ASSIGN(const Shape scatter_shape,
                          ShapeInference::InferScatterShape(
                              shapes.ptrs, to_apply(types()),
                              HloScatterInstruction::MakeScatterDimNumbers(
                                  /*update_window_dims=*/{1},
                                  /*inserted_window_dims=*/{0},
                                  /*scatter_dims_to_operand_dims=*/{0},
                                  /*index_vector_dim=*/1)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, Collate({64, 48}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, TfScatterWithPartialUpdates) {
  const auto shapes = CreateShapes({64, 48}, s64_vector(32), {10, 32}, types());
  TF_ASSERT_OK_AND_ASSIGN(const Shape scatter_shape,
                          ShapeInference::InferScatterShape(
                              shapes.ptrs, to_apply(types()),
                              HloScatterInstruction::MakeScatterDimNumbers(
                                  /*update_window_dims=*/{0},
                                  /*inserted_window_dims=*/{1},
                                  /*scatter_dims_to_operand_dims=*/{1},
                                  /*index_vector_dim=*/1)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, Collate({64, 48}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, TfScatterWithPartialUpdatesV2) {
  const auto shapes = CreateShapes({64, 48}, s64_vector(32), {32, 8}, types());
  TF_ASSERT_OK_AND_ASSIGN(const Shape scatter_shape,
                          ShapeInference::InferScatterShape(
                              shapes.ptrs, to_apply(types()),
                              HloScatterInstruction::MakeScatterDimNumbers(
                                  /*update_window_dims=*/{1},
                                  /*inserted_window_dims=*/{0},
                                  /*scatter_dims_to_operand_dims=*/{0},
                                  /*index_vector_dim=*/1)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, Collate({64, 48}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, TfScatterWithUpdatesBiggerThanInput) {
  const auto shapes = CreateShapes({64, 48}, s64_vector(32), {65, 32}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{0},
          /*inserted_window_dims=*/{1},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/1));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr("Bounds of the window dimensions of updates must not exceed "
                "the bounds of the corresponding dimensions of operand."))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest, TfScatterWithUpdatesBiggerThanInputV2) {
  const auto shapes = CreateShapes({64, 48}, s64_vector(32), {32, 49}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{1},
          /*inserted_window_dims=*/{0},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/1));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr("Bounds of the window dimensions of updates must not exceed "
                "the bounds of the corresponding dimensions of operand."))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest,
       TfScatterBatchingDimsWithUpdatesBiggerThanInput) {
  const auto shapes = CreateShapes({100, 64, 48}, s64_tensor({100, 32}),
                                   {100, 65, 32}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{1},
          /*inserted_window_dims=*/{2},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/2,
          /*input_batching_dims=*/{0},
          /*scatter_indices_batching_dims=*/{0}));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr("Bounds of the window dimensions of updates must not exceed "
                "the bounds of the corresponding dimensions of operand."))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest, TfScatterWithUpdatesNotMatchingIndices) {
  const auto shapes = CreateShapes({64, 48}, s64_vector(32), {64, 31}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{0},
          /*inserted_window_dims=*/{1},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/1));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr(
          "Bounds of the scatter dimensions of updates must be same as the "
          "bounds of the corresponding dimensions of scatter indices."))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest, TfScatterWithUpdatesNotMatchingIndicesV2) {
  const auto shapes = CreateShapes({64, 48}, s64_vector(32), {31, 48}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{1},
          /*inserted_window_dims=*/{0},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/1));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr(
          "Bounds of the scatter dimensions of updates must be same as the "
          "bounds of the corresponding dimensions of scatter indices."))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest, TfScatterNdWithFullUpdates) {
  const auto shapes = CreateShapes({64, 48}, s64_tensor({10, 9, 8, 7, 1}),
                                   {10, 9, 8, 7, 48}, types());
  TF_ASSERT_OK_AND_ASSIGN(const Shape scatter_shape,
                          ShapeInference::InferScatterShape(
                              shapes.ptrs, to_apply(types()),
                              HloScatterInstruction::MakeScatterDimNumbers(
                                  /*update_window_dims=*/{4},
                                  /*inserted_window_dims=*/{0},
                                  /*scatter_dims_to_operand_dims=*/{0},
                                  /*index_vector_dim=*/4)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, Collate({64, 48}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, TfScatterNdWithFullUpdatesV2) {
  const auto shapes = CreateShapes({64, 48}, s64_tensor({10, 9, 8, 7, 1}),
                                   {10, 9, 8, 7, 64}, types());
  TF_ASSERT_OK_AND_ASSIGN(const Shape scatter_shape,
                          ShapeInference::InferScatterShape(
                              shapes.ptrs, to_apply(types()),
                              HloScatterInstruction::MakeScatterDimNumbers(
                                  /*update_window_dims=*/{4},
                                  /*inserted_window_dims=*/{1},
                                  /*scatter_dims_to_operand_dims=*/{0},
                                  /*index_vector_dim=*/4)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, Collate({64, 48}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, TfScatterNdWithPartialUpdates) {
  const auto shapes = CreateShapes({64, 48}, s64_tensor({10, 9, 8, 7, 1}),
                                   {10, 9, 8, 7, 10}, types());
  TF_ASSERT_OK_AND_ASSIGN(const Shape scatter_shape,
                          ShapeInference::InferScatterShape(
                              shapes.ptrs, to_apply(types()),
                              HloScatterInstruction::MakeScatterDimNumbers(
                                  /*update_window_dims=*/{4},
                                  /*inserted_window_dims=*/{0},
                                  /*scatter_dims_to_operand_dims=*/{0},
                                  /*index_vector_dim=*/4)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, Collate({64, 48}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, TfScatterNdWithPartialUpdatesV2) {
  const auto shapes = CreateShapes({64, 48}, s64_tensor({10, 9, 8, 7, 1}),
                                   {10, 9, 8, 7, 12}, types());
  TF_ASSERT_OK_AND_ASSIGN(const Shape scatter_shape,
                          ShapeInference::InferScatterShape(
                              shapes.ptrs, to_apply(types()),
                              HloScatterInstruction::MakeScatterDimNumbers(
                                  /*update_window_dims=*/{4},
                                  /*inserted_window_dims=*/{1},
                                  /*scatter_dims_to_operand_dims=*/{0},
                                  /*index_vector_dim=*/4)));
  EXPECT_TRUE(ShapeUtil::Equal(scatter_shape, Collate({64, 48}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, TfScatterNdWithUpdatesBiggerThanInput) {
  const auto shapes = CreateShapes({64, 48}, s64_tensor({10, 9, 8, 7, 1}),
                                   {10, 9, 8, 7, 65}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4},
          /*inserted_window_dims=*/{1},
          /*scatter_dims_to_operand_dims=*/{0},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr("Bounds of the window dimensions of updates must not exceed "
                "the bounds of the corresponding dimensions of operand."))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest, TfScatterNdWithUpdatesNotMatchingIndices) {
  const auto shapes = CreateShapes({64, 48}, s64_tensor({10, 9, 8, 7, 1}),
                                   {9, 9, 8, 7, 64}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4},
          /*inserted_window_dims=*/{1},
          /*scatter_dims_to_operand_dims=*/{0},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr(
          "Bounds of the scatter dimensions of updates must be same as the "
          "bounds of the corresponding dimensions of scatter indices."))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest, TfBatchDynamicUpdateSlice) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28, 27, 26}, types());
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape scatter_shape,
      ShapeInference::InferScatterShape(
          shapes.ptrs, to_apply(types()),
          HloScatterInstruction::MakeScatterDimNumbers(
              /*update_window_dims=*/{4, 5, 6, 7, 8},
              /*inserted_window_dims=*/{},
              /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
              /*index_vector_dim=*/4)));
  EXPECT_TRUE(
      ShapeUtil::Equal(scatter_shape, Collate({50, 49, 48, 47, 46}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, NonDefaultScatterIndicesLeafDim) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 5, 7, 6}),
                   {10, 9, 7, 6, 30, 29, 28, 27, 26}, types());
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape scatter_shape,
      ShapeInference::InferScatterShape(
          shapes.ptrs, to_apply(types()),
          HloScatterInstruction::MakeScatterDimNumbers(
              /*update_window_dims=*/{4, 5, 6, 7, 8},
              /*inserted_window_dims=*/{},
              /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
              /*index_vector_dim=*/2)));

  EXPECT_TRUE(
      ShapeUtil::Equal(scatter_shape, Collate({50, 49, 48, 47, 46}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, NonDefaultScatterIndicesLeafDimV2) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({5, 10, 9, 7, 6}),
                   {10, 9, 7, 6, 30, 29, 28, 27, 26}, types());
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape scatter_shape,
      ShapeInference::InferScatterShape(
          shapes.ptrs, to_apply(types()),
          HloScatterInstruction::MakeScatterDimNumbers(
              /*update_window_dims=*/{4, 5, 6, 7, 8},
              /*inserted_window_dims=*/{},
              /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
              /*index_vector_dim=*/0)));

  EXPECT_TRUE(
      ShapeUtil::Equal(scatter_shape, Collate({50, 49, 48, 47, 46}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, NoUpdateScatterDims) {
  const auto shapes = CreateShapes({50, 49, 48, 47, 46}, s64_vector(5),
                                   {30, 29, 28, 27, 26}, types());
  // This is equivalent to a dynamic update slice.
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape scatter_shape,
      ShapeInference::InferScatterShape(
          shapes.ptrs, to_apply(types()),
          HloScatterInstruction::MakeScatterDimNumbers(
              /*update_window_dims=*/{0, 1, 2, 3, 4},
              /*inserted_window_dims=*/{},
              /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
              /*index_vector_dim=*/0)));

  EXPECT_TRUE(
      ShapeUtil::Equal(scatter_shape, Collate({50, 49, 48, 47, 46}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, ScalarScatterIndices) {
  const auto shapes = CreateShapes({50, 49, 48, 47, 46}, scalar(S64),
                                   {30, 29, 28, 27}, types());
  // The scalar indices "tensor" is a scalar S here that's used to update a
  // [30,29,28,27] shaped tensor within the operand at position S.
  TF_ASSERT_OK_AND_ASSIGN(const Shape scatter_shape,
                          ShapeInference::InferScatterShape(
                              shapes.ptrs, to_apply(types()),
                              HloScatterInstruction::MakeScatterDimNumbers(
                                  /*update_window_dims=*/{0, 1, 2, 3},
                                  /*inserted_window_dims=*/{0},
                                  /*scatter_dims_to_operand_dims=*/{0},
                                  /*index_vector_dim=*/0)));

  EXPECT_TRUE(
      ShapeUtil::Equal(scatter_shape, Collate({50, 49, 48, 47, 46}, types())))
      << ShapeUtil::HumanString(scatter_shape);
}

TEST_P(ScatterShapeInferenceTest, ScatterWithTupleShapedTensorInput) {
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(S64, {10, 9, 8, 7, 1}),
                                 ShapeUtil::MakeShape(S64, {10, 9, 8, 7, 1})});
  const Shape s64_vector_32 = s64_vector(32);
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      {&tuple_shape, &s64_vector_32, &s64_vector_32}, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{0},
          /*inserted_window_dims=*/{1},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/1));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Expected array argument for operand"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest, ScatterWithTupleShapedScatterIndicesInput) {
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(S64, {10, 9, 8, 7, 1}),
                                 ShapeUtil::MakeShape(S64, {10, 9, 8, 7, 1})});
  const Shape s64_vector_32 = s64_vector(32);
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      {&s64_vector_32, &tuple_shape, &s64_vector_32}, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{0},
          /*inserted_window_dims=*/{1},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/0));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Expected array argument for scatter indices"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest, ScatterWithTupleShapedUpdatesInput) {
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(S64, {10, 9, 8, 7, 1}),
                                 ShapeUtil::MakeShape(S64, {10, 9, 8, 7, 1})});
  const Shape s64_vector_32 = s64_vector(32);
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      {&s64_vector_32, &s64_vector_32, &tuple_shape}, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{0},
          /*inserted_window_dims=*/{1},
          /*scatter_dims_to_operand_dims=*/{1},
          /*index_vector_dim=*/0));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Expected array argument for updates"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest, OutOfBoundsScatterIndicesLeafDim) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 2},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/10));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Scatter index leaf dimension must be within [0, "
                        "rank(scatter_indices) + 1)"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest, InvalidUpdates) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28, 50}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 2},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Updates tensor must be of rank 7; got 8."))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest, InvalidUpdateComputation) {
  const ProgramShape invalid_update_computation =
      ShapeUtil::MakeProgramShape({u32_}, u32_);
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, invalid_update_computation,
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 2},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr(absl::Substitute(
                  "Reduction function must take $0 parameters, but takes 1",
                  2 * types().size())))
      << statusor.status();
}
TEST_P(ScatterShapeInferenceTest,
       InvalidScatterDimNumbers_NonAscendingUpdateWindowDims) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28, 27, 26}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6, 8, 7},
          /*inserted_window_dims=*/{},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("update_window_dims in scatter op must be sorted"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest,
       InvalidScatterDimNumbers_RepeatedUpdateWindowDims) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28, 27, 26}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6, 7, 7},
          /*inserted_window_dims=*/{},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("update_window_dims in scatter op must not repeat"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest,
       InvalidScatterDimNumbers_OutOfBoundsUpdateWindowDims) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28, 27, 26}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6, 7, 9},
          /*inserted_window_dims=*/{},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Invalid update_window_dims set in scatter op; valid "
                        "range is [0, 9)"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest,
       InvalidScatterDimNumbers_NonAscendingInsertedWindowDims) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{2, 1},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("inserted_window_dims in scatter op must be sorted"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest,
       InvalidScatterDimNumbers_RepeatedInsertedWindowDims) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 1},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("inserted_window_dims in scatter op must not repeat"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest,
       InvalidScatterDimNumbers_OutOfBoundsInsertedWindowDims) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 5},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Invalid inserted_window_dims set in scatter op; valid "
                        "range is [0, 5)"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest,
       InvalidScatterDimNumbers_MismatchingScatterDimsToOperandDims) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 2},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr("Scatter op has 4 elements in scatter_dims_to_operand_dims and "
                "the bound of dimension index_vector_dim=4 of scatter_indices "
                "is 5. These two numbers must be equal"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest,
       InvalidScatterDimNumbers_OutOfBoundsScatterDimsToOperandDims) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 2},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 10},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Invalid scatter_dims_to_operand_dims mapping; domain "
                        "is [0, 5), got: 4->10"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest,
       InvalidScatterDimNumbers_RepeatedValuesInScatterDimsToOperandDims) {
  const auto shapes =
      CreateShapes({50, 49, 48, 47, 46}, s64_tensor({10, 9, 8, 7, 5}),
                   {10, 9, 8, 7, 30, 29, 28}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{4, 5, 6},
          /*inserted_window_dims=*/{1, 2},
          /*scatter_dims_to_operand_dims=*/{0, 1, 2, 2, 3},
          /*index_vector_dim=*/4));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr(
          "Repeated dimensions not allowed in scatter_dims_to_operand_dims"))
      << statusor.status();
}

TEST_P(ScatterShapeInferenceTest,
       InvalidScatterDimNumbers_InsufficientWindowDims) {
  const auto shapes = CreateShapes({50, 49, 48, 47, 46}, scalar(S64),
                                   {30, 29, 28, 27}, types());
  const absl::StatusOr<Shape> statusor = ShapeInference::InferScatterShape(
      shapes.ptrs, to_apply(types()),
      HloScatterInstruction::MakeScatterDimNumbers(
          /*update_window_dims=*/{0, 1, 2, 3},
          /*inserted_window_dims=*/{},
          /*scatter_dims_to_operand_dims=*/{0},
          /*index_vector_dim=*/0));
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr(
          "Scatter op has window of size 4; doesn't match operand of rank 5."))
      << statusor.status();
}

struct ScatterTestName {
  std::string operator()(
      const ::testing::TestParamInfo<std::vector<PrimitiveType>>& info) const {
    return absl::StrJoin(info.param, "_", absl::StreamFormatter());
  }
};

INSTANTIATE_TEST_SUITE_P(All, ScatterShapeInferenceTest,
                         ::testing::Values(std::vector<PrimitiveType>{U32},
                                           std::vector<PrimitiveType>{U32,
                                                                      U16}),
                         ScatterTestName());

TEST_P(UnboundedUnaryOpShapeInferenceTest, UnboundedUnaryOps) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape(GetParam().operand));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape(GetParam().expected));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred,
      ShapeInference::InferUnaryOpShape(GetParam().opcode, operand));
  EXPECT_TRUE(ShapeUtil::Equal(inferred, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedAdd) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_F(ShapeInferenceTest, UnboundedAllGather) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferAllGatherShape(
          {&operand}, /*all_gather_dimension=*/0, /*shard_count=*/2));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedAllReduce) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferAllReduceShape({&operand}));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedAllToAll) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferAllToAllShape(/*shape=*/operand,
                                         /*split_dimension=*/0,
                                         /*concat_dimension=*/0,
                                         /*split_count=*/3));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedAllToAllTupleUnsupported) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("(u32[?, 10], u32[?, 10])"));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferAllToAllTupleShape(
          /*operand_shapes=*/{&operand, &operand});
  EXPECT_THAT(
      inferred_shape.status().message(),
      HasSubstr("AllToAllTuple does not support unbounded dynamic shapes"));
}

TEST_P(UnboundedLogicalOpShapeInferenceTest, UnboundedAnd) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAnd, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_F(ShapeInferenceTest, UnboundedBitcastConvert) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u64[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferBitcastConvertShape(operand, PrimitiveType::U32));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10, 2]"));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedBroadcastUnsupportedOperand) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[<=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[1, <=2, ?]"));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBroadcastShape(operand, /*broadcast_sizes=*/{1});
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("is_unbounded_dynamic"));
}

TEST_F(ShapeInferenceTest, UnboundedBroadcastUnsupportedBroadcastSize) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[<=2, 4]"));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBroadcastShape(
          operand, /*broadcast_sizes=*/{Shape::kUnboundedSize});
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("Non-broadcast dimensions must not be dynamic."));
}

TEST_F(ShapeInferenceTest, UnboundedBroadcastInDim) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[<=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[<=2, 3, 4]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferBroadcastShape(operand, expected,
                                          /*broadcast_dimensions=*/{0, 2}));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedBroadcastInDimToBounded) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[<=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[<=2, 3, <=4]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferBroadcastShape(operand, expected,
                                          /*broadcast_dimensions=*/{0, 2}));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedBroadcastInDimUnsupportedOutput) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[<=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[<=2, 3, ?]"));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBroadcastShape(operand, expected,
                                          /*broadcast_dimensions=*/{0, 2});
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("is_unbounded_dynamic"));
}

TEST_F(ShapeInferenceTest, UnboundedBroadcastInDimUnsupported) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[<=2, 4]"));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBroadcastShape(
          operand, /*broadcast_sizes=*/{2, Shape::kUnboundedSize, 4});
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("Non-broadcast dimensions must not be dynamic."));
}

TEST_F(ShapeInferenceTest, UnboundedCall) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand0, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand1, ParseShape("u32[10, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferCallShape(
                              /*arg_shapes=*/{&operand0, &operand1},
                              /*to_apply=*/ShapeUtil::MakeProgramShape(
                                  {operand1, operand0}, operand0)));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_P(UnboundedClampOpShapeInferenceTest, UnboundedClamp) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam()[0]));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam()[1]));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape(GetParam()[2]));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, lhs, rhs, ehs);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape(GetParam()[3]));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    EXPECT_EQ(inferred_shape.status().message(), GetParam()[4]);
  }
}

TEST_F(ShapeInferenceTest, UnboundedClampWithTuple) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("(u32[2], u32[?])"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("(u32[?], u32[2])"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("(u32[2], u32[?])"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("(u32[?], u32[2])"));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kClamp, lhs, rhs, ehs);
  EXPECT_THAT(
      inferred_shape.status().message(),
      HasSubstr(
          "Expected array argument for clamp min, but got (u32[2], u32[?])."));
}

TEST_F(ShapeInferenceTest, UnboundedCollectiveBroadcast) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferCollectiveBroadcastShape(
                              /*operand_shapes=*/{&operand}));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, CollectivePermute) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[8, 8]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[8, 8]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferCollectivePermuteShape(
          /*operand_shapes=*/{&operand}, /*inplace=*/false));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, CollectivePermuteStart) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[8, 8]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("(u32[8, 8], u32[8, 8])"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferCollectivePermuteStartShape(
          /*operand_shapes=*/{&operand}, {}, /*inplace=*/false));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, CombinedCollectivePermute) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand_0, ParseShape("u32[8, 8]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand_1, ParseShape("u32[16, 16]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("(u32[8, 8], u32[16, 16])"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferCollectivePermuteShape(
          /*operand_shapes=*/{&operand_0, &operand_1}, /*inplace=*/false));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, CombinedCollectivePermuteStart) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand_0, ParseShape("u32[8, 8]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand_1, ParseShape("u32[16, 16]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape expected,
      ParseShape("((u32[8, 8], u32[16, 16]), (u32[8, 8], u32[16, 16]))"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferCollectivePermuteStartShape(
                              /*operand_shapes=*/{&operand_0, &operand_1}, {},
                              /*inplace=*/false));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, CombinedInplaceCollectivePermute) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand,
                          ParseShape("(u32[2,3], u32[2,3], u32[], u32[])"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[2,3]"));
  std::vector<const Shape*> operand_shapes;
  absl::c_transform(operand.tuple_shapes(), std::back_inserter(operand_shapes),
                    [](const Shape& shape) { return &shape; });
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferCollectivePermuteShape(
          /*operand_shapes=*/operand_shapes, /*inplace=*/true));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedCollectivePermute) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferCollectivePermuteShape(
          /*operand_shapes=*/{&operand}, /*inplace=*/false));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_P(UnboundedCompareOpShapeInferenceTest, UnboundedCompare) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kCompare, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_P(UnboundedConcatenateOpShapeInferenceTest, UnboundedConcatenate) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand1, ParseShape(GetParam()[0]));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand2, ParseShape(GetParam()[1]));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferConcatOpShape({&operand1, &operand2},
                                         /*dimension=*/0);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape(GetParam()[2]));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    EXPECT_EQ(inferred_shape.status().message(), GetParam()[3]);
  }
}

TEST_F(UnboundedConcatenateOpShapeInferenceTest,
       UnboundedConcatenateMismatchedDimensions) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand1, ParseShape("u32[2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand2, ParseShape("u32[2, 3]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand3, ParseShape("u32[2, 4]"));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferConcatOpShape({&operand1, &operand2, &operand3},
                                         /*dimension=*/0);
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("Mismatched dimension sizes 3 and 4 in dimension 1"));
}

TEST_F(UnboundedConcatenateOpShapeInferenceTest,
       UnboundedConcatenateMismatchedBoundSizes) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand1, ParseShape("u32[2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand2, ParseShape("u32[2, <=3]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand3, ParseShape("u32[2, <=4]"));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferConcatOpShape({&operand1, &operand2, &operand3},
                                         /*dimension=*/0);
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("Mismatched bound sizes 3 and 4 in dimension 1"));
}

TEST_F(ShapeInferenceTest, UnboundedConvert) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u64[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape result, ShapeInference::InferConvertShape(
                                                  operand, PrimitiveType::U64));
  EXPECT_TRUE(ShapeUtil::Equal(result, expected))
      << "inferred: " << ShapeUtil::HumanString(result)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedDiv) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kDivide, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_F(ShapeInferenceTest, UnboundedDot) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));

  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(1);
  dnums.add_rhs_contracting_dimensions(0);

  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferDotOpShape(lhs, rhs, dnums,
                                      /*preferred_element_type=*/std::nullopt));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedDotGeneral) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("u32[?, <=3, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("u32[2, 4, 5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, <=3, 5]"));

  DotDimensionNumbers dnums;
  dnums.add_lhs_batch_dimensions(0);
  dnums.add_rhs_batch_dimensions(0);
  dnums.add_lhs_contracting_dimensions(2);
  dnums.add_rhs_contracting_dimensions(1);

  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferDotOpShape(lhs, rhs, dnums,
                                      /*preferred_element_type=*/std::nullopt));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedDynamicSlice) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape start_index, ParseShape("s32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[2, 2]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferDynamicSliceShape(
          operand, /*start_index_shapes=*/{start_index, start_index},
          /*slice_sizes=*/{2, 2}, /*allow_scalar_indices=*/true));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedDynamicUpdateSlice) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape update, ParseShape("u32[?, 5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape start_index, ParseShape("s32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferDynamicUpdateSliceShape(
          operand, update, /*start_index_shapes=*/{start_index, start_index},
          /*allow_scalar_indices=*/true));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

// clang-format off
// TODO(chokobole): Uncomment comment. Dependency: ShapeInference::InferGatherShape
// clang-format on
// TEST_F(ShapeInferenceTest, UnboundedGather) {

TEST_F(ShapeInferenceTest, UnboundedGetTupleElement) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferGetTupleElementShape(
          ShapeUtil::MakeTupleShape({operand}), /*index=*/0));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedMap) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand0, ParseShape("u32[2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand1, ParseShape("u32[?, 3, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[2, ?, ?]"));

  const ProgramShape to_apply = ShapeUtil::MakeProgramShape({u32_, u32_}, u32_);

  TF_ASSERT_OK_AND_ASSIGN(
      const Shape result_shape,
      ShapeInference::InferMapShape(/*arg_shapes=*/{&operand0, &operand1},
                                    to_apply, /*dimensions=*/{0, 1, 2}));
  EXPECT_TRUE(ShapeUtil::Equal(result_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(result_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedMax) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kMaximum, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedMin) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_status =
      ShapeInference::InferBinaryOpShape(HloOpcode::kMinimum, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_status.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_status, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_status)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_status.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedMul) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kMultiply, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_P(UnboundedLogicalOpShapeInferenceTest, UnboundedOr) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kOr, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_F(ShapeInferenceTest, UnboundedPad) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape padding_value, ParseShape("u32[]"));
  // TODO(chokobole): Do we need this? Dependency: interior_padding
  // TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 21]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 12]"));

  PaddingConfig padding_config;
  for (int i = 0; i < 2; i++) {
    const auto dimension = padding_config.add_dimensions();
    dimension->set_edge_padding_low(1);
    dimension->set_edge_padding_high(1);
    // TODO(chokobole): Do we need this? Dependency: interior_padding
    // dimension->set_interior_padding(1);
  }

  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferPadShape(operand, padding_value, padding_config));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedPow) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kPower, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_F(ShapeInferenceTest, UnboundedReduce) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape input0, ParseShape("u32[7, 5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape input1, ParseShape("u32[?, 5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape input2, ParseShape("u32[7, ?]"));

  ProgramShape to_apply = ShapeUtil::MakeProgramShape(
      {u32_, u32_, u32_, u32_, u32_, u32_},
      ShapeUtil::MakeTupleShape({u32_, u32_, u32_}));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferReduceShape(
          {&input0, &input1, &input2, &u32_, &u32_, &u32_}, {1}, to_apply));
  const Shape shape = ShapeUtil::MakeShape(U32, {7});
  const Shape expected = ShapeUtil::MakeTupleShape({shape, shape, shape});
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedReduceInvalidReduceDimension) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape input0, ParseShape("u32[7, 5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape input1, ParseShape("u32[?, 5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape input2, ParseShape("u32[5, ?]"));

  ProgramShape to_apply = ShapeUtil::MakeProgramShape(
      {u32_, u32_, u32_, u32_, u32_, u32_},
      ShapeUtil::MakeTupleShape({u32_, u32_, u32_}));
  const absl::StatusOr<Shape> inferred_shape = ShapeInference::InferReduceShape(
      {&input0, &input1, &input2, &u32_, &u32_, &u32_}, {1}, to_apply);
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("All reduced tensors must have compatible dimension"));
}

TEST_F(ShapeInferenceTest, UnboundedReduceScatter) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferReduceScatterShape(
                              /*operand_shapes=*/{&operand},
                              /*scatter_dimension=*/0, /*shard_count=*/2));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedRemainder) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_status =
      ShapeInference::InferBinaryOpShape(HloOpcode::kRemainder, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_status.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_status, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_status)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_status.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_F(ShapeInferenceTest, UnboundedReshape) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[2,3]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred,
      ShapeInference::InferReshapeShape(operand, /*dimensions=*/{0},
                                        /*new_sizes=*/{2, 3}, -1));
  ASSERT_TRUE(ShapeUtil::Equal(inferred, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedReshapeUnsupportedOutputShape) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[6]"));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferReshapeShape(
          operand, /*dimensions=*/{0},
          /*new_sizes=*/{Shape::kUnboundedSize, Shape::kUnboundedSize}, -1);
  EXPECT_THAT(
      inferred_shape.status().message(),
      HasSubstr("Reshaping with unbounded result shape is not supported."));
}

TEST_F(ShapeInferenceTest, UnboundedReshapeUnsupportedMixOfDynamism) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, <=3]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[<=3]"));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferReshapeShape(operand, /*dimensions=*/{0},
                                        /*new_sizes=*/{3}, -1);
  ASSERT_THAT(inferred_shape.status().message(),
              HasSubstr("Reshape operand with bounded and unbounded dynamism "
                        "not supported."));
}

TEST_F(ShapeInferenceTest, UnboundedReverse) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferReverseShape(operand, /*dimensions=*/{0, 1}));
  ASSERT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedScatter) {
  TF_ASSERT_OK_AND_ASSIGN(Shape input, ParseShape("u32[?, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(Shape scatter_indices, ParseShape("s32[?, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(Shape updates, ParseShape("u32[?, ?, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(Shape expected, ParseShape("u32[?, ?, ?]"));

  const ProgramShape to_apply = ShapeUtil::MakeProgramShape({u32_, u32_}, u32_);

  ScatterDimensionNumbers dimension_numbers;
  dimension_numbers.add_update_window_dims(2);
  dimension_numbers.add_update_window_dims(3);
  dimension_numbers.add_inserted_window_dims(0);
  dimension_numbers.add_scatter_dims_to_operand_dims(1);
  dimension_numbers.add_scatter_dims_to_operand_dims(0);
  dimension_numbers.set_index_vector_dim(2);

  TF_ASSERT_OK_AND_ASSIGN(
      Shape result,
      ShapeInference::InferScatterShape({&input, &scatter_indices, &updates},
                                        to_apply, dimension_numbers));
  EXPECT_TRUE(ShapeUtil::Equal(result, expected))
      << "inferred: " << ShapeUtil::HumanString(result)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_P(UnboundedSelectOpShapeInferenceTest, UnboundedSelect) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam()[0]));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam()[1]));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape(GetParam()[2]));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kSelect, lhs, rhs, ehs);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape(GetParam()[3]));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    EXPECT_EQ(inferred_shape.status().message(), GetParam()[4]);
  }
}

TEST_F(ShapeInferenceTest, UnboundedSelectWithTupleUnsupported) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("(pred[2], pred[?])"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("(u32[?], u32[2])"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("(u32[2], u32[?])"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("(u32[?], u32[2])"));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferTernaryOpShape(HloOpcode::kSelect, lhs, rhs, ehs);
  EXPECT_THAT(inferred_shape.status().message(),
              HasSubstr("Expected array argument for select pred, but got "
                        "(pred[2], pred[?])."));
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedShiftLeft) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_status =
      ShapeInference::InferBinaryOpShape(HloOpcode::kShiftLeft, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_status.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_status, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_status)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_status.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedShiftRightArithmetic) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_status =
      ShapeInference::InferBinaryOpShape(HloOpcode::kShiftRightArithmetic, lhs,
                                         rhs, GetParam().broadcast_dimensions);
  if (inferred_status.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_status, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_status)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_status.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedShiftRightLogical) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_status =
      ShapeInference::InferBinaryOpShape(HloOpcode::kShiftRightLogical, lhs,
                                         rhs, GetParam().broadcast_dimensions);
  if (inferred_status.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_status, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_status)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_status.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_F(ShapeInferenceTest, UnboundedSlice) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[1, <=3, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[1, <=2, 3]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferSliceShape(operand, /*starts=*/{0, 1, 2},
                                      /*limits=*/{1, 3, 5},
                                      /*strides=*/{1, 1, 1}));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedSort) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferVariadicOpShape(HloOpcode::kSort, {&operand}));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedSub) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kSubtract, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_F(ShapeInferenceTest, UnboundedTranspose) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand,
                          ParseShape("u32[1, ?, 2, ?, <=2]{4,3,2,1,0}"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("u32[<=2, 1, ?, 2, ?]{0,2,3,4,1}"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferTransposeShape(
                              operand, /*dimensions=*/{4, 0, 3, 2, 1}));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedTransposeRank1) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferTransposeShape(operand, /*dimensions=*/{0}));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedTuple) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  const Shape expected = ShapeUtil::MakeTupleShape({operand});
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape result_shape,
      ShapeInference::InferVariadicOpShape(
          HloOpcode::kTuple, std::vector<const Shape*>({&operand})));
  EXPECT_TRUE(ShapeUtil::Equal(result_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(result_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedWhile) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape init, ParseShape("u32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape result_shape, ParseShape("u32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferWhileShape(
          /*condition=*/ShapeUtil::MakeProgramShape({result_shape}, pred_),
          /*body=*/ShapeUtil::MakeProgramShape({result_shape}, result_shape),
          /*init=*/init));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_P(UnboundedLogicalOpShapeInferenceTest, UnboundedXor) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_status =
      ShapeInference::InferBinaryOpShape(HloOpcode::kXor, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_status.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_status, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_status)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_status.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

INSTANTIATE_TEST_SUITE_P(UnboundedDynamism,
                         UnboundedLogicalOpShapeInferenceTest,
                         ::testing::ValuesIn<BinaryOpTestCase>(
                             {// LHS | RHS | bdims | Res
                              // 1   | ?   | []    | ?
                              {"s32[1]", "s32[?]", {}, "s32[?]"},
                              // ?   | 1   | []    | ?
                              {"s32[?]", "s32[1]", {}, "s32[?]"},
                              // 2   | ?   | []    | 2
                              {"s32[2]", "s32[?]", {}, "s32[2]"},
                              // ?   | 2   | []    | 2
                              {"s32[?]", "s32[2]", {}, "s32[2]"},
                              // <=2 | ?   | []    | <=2
                              {"s32[<=2]", "s32[?]", {}, "s32[<=2]"},
                              // ?   | <=2 | []    | <=2
                              {"s32[?]", "s32[<=2]", {}, "s32[<=2]"},
                              // ?   | ?   | []    | ?
                              {"s32[?]", "s32[?]", {}, "s32[?]"},
                              // 1   | ?,3 | [0]   | ?,3
                              {"s32[1]", "s32[?,3]", zero_array, "s32[?,3]"},
                              // 2   | ?,3 | [0]   | err
                              {"s32[2]", "s32[?,3]", zero_array, "",
                               kBroadcastDimensionMismatchErrorMessage},
                              // ?,2 | ?,3 | []    | err
                              {"s32[?,2]",
                               "s32[?,3]",
                               {},
                               "",
                               kIncompatibleBinaryOpShapeErrorMessage}}));

INSTANTIATE_TEST_SUITE_P(UnboundedDynamism, UnboundedBinaryOpShapeInferenceTest,
                         ::testing::ValuesIn<BinaryOpTestCase>(
                             {// LHS | RHS | bdims | Res
                              // 1   | ?   | []    | ?
                              {"u32[1]", "u32[?]", {}, "u32[?]"},
                              // ?   | 1   | []    | ?
                              {"u32[?]", "u32[1]", {}, "u32[?]"},
                              // 2   | ?   | []    | 2
                              {"u32[2]", "u32[?]", {}, "u32[2]"},
                              // ?   | 2   | []    | 2
                              {"u32[?]", "u32[2]", {}, "u32[2]"},
                              // <=2 | ?   | []    | <=2
                              {"u32[<=2]", "u32[?]", {}, "u32[<=2]"},
                              // ?   | <=2 | []    | <=2
                              {"u32[?]", "u32[<=2]", {}, "u32[<=2]"},
                              // ?   | ?   | []    | ?
                              {"u32[?]", "u32[?]", {}, "u32[?]"},
                              // 1   | ?,3 | [0]   | ?,3
                              {"u32[1]", "u32[?,3]", zero_array, "u32[?,3]"},
                              // 2   | ?,3 | [0]   | err
                              {"u32[2]", "u32[?,3]", zero_array, "",
                               kBroadcastDimensionMismatchErrorMessage},
                              // ?,2 | ?,3 | []    | err
                              {"u32[?,2]",
                               "u32[?,3]",
                               {},
                               "",
                               kIncompatibleBinaryOpShapeErrorMessage}}));

INSTANTIATE_TEST_SUITE_P(UnboundedDynamism,
                         UnboundedCompareOpShapeInferenceTest,
                         ::testing::ValuesIn<BinaryOpTestCase>(
                             {// LHS | RHS | bdims | Res
                              // 1   | ?   | []    | ?
                              {"u32[1]", "u32[?]", {}, "pred[?]"},
                              // ?   | 1   | []    | ?
                              {"u32[?]", "u32[1]", {}, "pred[?]"},
                              // 2   | ?   | []    | 2
                              {"u32[2]", "u32[?]", {}, "pred[2]"},
                              // ?   | 2   | []    | 2
                              {"u32[?]", "u32[2]", {}, "pred[2]"},
                              // <=2 | ?   | []    | <=2
                              {"u32[<=2]", "u32[?]", {}, "pred[<=2]"},
                              // ?   | <=2 | []    | <=2
                              {"u32[?]", "u32[<=2]", {}, "pred[<=2]"},
                              // ?   | ?   | []    | ?
                              {"u32[?]", "u32[?]", {}, "pred[?]"},
                              // 1   | ?,3 | [0]   | ?,3
                              {"u32[1]", "u32[?,3]", zero_array, "pred[?,3]"},
                              // 2   | ?,3 | [0]   | err
                              {"u32[2]", "u32[?,3]", zero_array, "",
                               kBroadcastDimensionMismatchErrorMessage},
                              // ?,2 | ?,3 | []    | err
                              {"u32[?,2]",
                               "u32[?,3]",
                               {},
                               "",
                               kIncompatibleBinaryOpShapeErrorMessage}}));

INSTANTIATE_TEST_SUITE_P(
    UnboundedDynamism, UnboundedConcatenateOpShapeInferenceTest,
    ::testing::Values(
        // LHS shape | RHS shape   | Result shape (Concat dim is 0)
        // [X1, Y]   | [X2, Y]     | [X1+X2, Y]
        std::vector<std::string>({"u32[2, 3]", "u32[4, 3]", "u32[6, 3]", ""}),
        // [X, Y]    | [?, ?]      | [?, Y]
        std::vector<std::string>({"u32[2, 3]", "u32[?, ?]", "u32[?, 3]", ""}),
        // [X1, Y]   | [<=X2, <=Y] | [<=X1+X2, <=Y]
        std::vector<std::string>({"u32[4, 3]", "u32[<=2, <=3]", "u32[<=6, <=3]",
                                  ""}),
        // [?, ?]    | [?, ?]      | [?, ?]
        std::vector<std::string>({"u32[?, ?]", "u32[?, ?]", "u32[?, ?]", ""}),
        // [?, ?]    | [<=B1, <=B2]| [?, <=B2]
        std::vector<std::string>({"u32[?, ?]", "u32[<=2, <=3]", "u32[?, <=3]",
                                  ""}),
        // [<=B1, ?] | [<=B2, X]   | [<=B1+B2, X]
        std::vector<std::string>({"u32[<=2, ?]", "u32[<=4, 3]", "u32[<=6, 3]",
                                  ""}),
        // [X, <=B1] | [X, <=B2]   | Error, mismatched
        // bound sizes
        std::vector<std::string>(
            {"u32[2, <=3]", "u32[2, <=4]", "",
             "Cannot concatenate arrays that differ in dimensions other than "
             "the one being concatenated. Dimension 1 in both shapes must be "
             "equal (or compatible): u32[2,<=3] vs u32[2,<=4]."}),
        // [X, Y1]   | [X, Y2]     | Error, mismatched
        // dimension sizes
        std::vector<std::string>(
            {"u32[2, 3]", "u32[2, 4]", "",
             "Cannot concatenate arrays that differ in dimensions other than "
             "the one being concatenated. Dimension 1 in both shapes must be "
             "equal (or compatible): u32[2,3] vs u32[2,4]."})));

INSTANTIATE_TEST_SUITE_P(
    UnboundedDynamism, UnboundedClampOpShapeInferenceTest,
    ::testing::Values(
        // MIN shape | OPERAND shape | MAX shape  | Result
        // []        | [?]           | []         | [?]
        std::vector<std::string>({"u32[]", "u32[?]", "u32[]", "u32[?]", ""}),
        // []        | [?]           | [X]        | [?]
        std::vector<std::string>({"u32[]", "u32[?]", "u32[2]", "u32[?]", ""}),
        // []        | [?]           | [<=B]      | [?]
        std::vector<std::string>({"u32[]", "u32[?]", "u32[<=2]", "u32[?]", ""}),
        // [X]       | [?]           | [X]        | [?]
        std::vector<std::string>({"u32[2]", "u32[?]", "u32[2]", "u32[?]", ""}),
        // [?]       | [X]           | [X]        | [X]
        std::vector<std::string>({"u32[?]", "u32[2]", "u32[2]", "u32[2]", ""}),
        // [?]       | [<=B]         | [?]        | [<=B]
        std::vector<std::string>({"u32[?]", "u32[<=2]", "u32[?]", "u32[<=2]",
                                  ""}),
        // [<=B]     | [?]           | [<=B]      | [?]
        std::vector<std::string>({"u32[<=2]", "u32[?]", "u32[<=2]", "u32[?]",
                                  ""}),
        // [?]       | [?]           | [?]        | [?]
        std::vector<std::string>({"u32[?]", "u32[?]", "u32[?]", "u32[?]", ""}),
        // [?]       | []            | [?]        | error
        std::vector<std::string>(
            {"u32[?]", "u32[]", "u32[?]", "",
             "Clamp with incompatible shapes: u32[?], u32[], u32[?]."}),
        // A[]       | B[?]          | B[?]       | error
        std::vector<std::string>(
            {"s32[]", "u32[?]", "u32[?]", "",
             "Clamp with incompatible element types: s32[], u32[?], u32[?]."}),
        // [X]       | [<=B]         | [X]        | error
        std::vector<std::string>(
            {"u32[3]", "u32[<=2]", "u32[3]", "",
             "Clamp with incompatible shapes: u32[3], u32[<=2], u32[3]."}),
        // [X]       | [?]           | [Y]        | error
        std::vector<std::string>(
            {"u32[2]", "u32[?]", "u32[3]", "",
             "Clamp with incompatible shapes: u32[2], u32[?], u32[3]."})));

INSTANTIATE_TEST_SUITE_P(
    UnboundedDynamism, UnboundedSelectOpShapeInferenceTest,
    ::testing::Values(
        // PRED shape | ON_TRUE shape | ON_FALSE shape  | Result
        // []         | [?]           | [X]             | [X]
        std::vector<std::string>({"pred[]", "u32[?]", "u32[2]", "u32[2]", ""}),
        // []         | [?]           | [<=B]           | [<=B]
        std::vector<std::string>({"pred[]", "u32[?]", "u32[<=2]", "u32[<=2]",
                                  ""}),
        // [X]        | [?]           | [X]             | [X]
        std::vector<std::string>({"pred[2]", "u32[?]", "u32[2]", "u32[2]", ""}),
        // [?]        | [X]           | [X]             | [X]
        std::vector<std::string>({"pred[?]", "u32[2]", "u32[?]", "u32[2]", ""}),
        // [?]        | [<=B]         | [?]             | [<=B]
        std::vector<std::string>({"pred[?]", "u32[<=2]", "u32[?]", "u32[<=2]",
                                  ""}),
        // [<=B]      | [?]           | [<=B]           | [<=B]
        std::vector<std::string>({"pred[<=2]", "u32[?]", "u32[<=2]", "u32[<=2]",
                                  ""}),
        // [?]        | [?]           | [?]             | [?]
        std::vector<std::string>({"pred[?]", "u32[?]", "u32[?]", "u32[?]", ""}),
        // [X]        | A[X]          | B[X]            | error
        std::vector<std::string>({"pred[3]", "s32[3]", "u32[3]", "",
                                  "Operands to select must be the same shape; "
                                  "got s32[3] and u32[3]."}),
        // [X]        | [?]           | [<=B]           | error
        std::vector<std::string>(
            {"pred[3]", "u32[?]", "u32[<=2]", "",
             "Operands to select and predicate must be the same shape; got "
             "u32[?] and u32[<=2] and pred[3]."}),
        // [X]        | [<=B]         | [X]             | error
        std::vector<std::string>({"pred[3]", "u32[<=2]", "u32[3]", "",
                                  "Operands to select must be the same shape; "
                                  "got u32[<=2] and u32[3]."}),
        // [X]        | [?]           | [Y]             | error
        std::vector<std::string>(
            {"pred[2]", "u32[?]", "u32[3]", "u32[3]",
             "Operands to select and predicate must be the same shape; got "
             "u32[?] and u32[3] and pred[2]."}),
        // [?]        | []            | []              | error
        std::vector<std::string>(
            {"pred[?]", "u32[]", "u32[]", "",
             "Operands to select and predicate must be the same shape; got "
             "u32[] and u32[] and pred[?]."}),
        // []         | [?]           | []              | error
        std::vector<std::string>({"pred[]", "u32[?]", "u32[]", "",
                                  "Operands to select must be the same shape; "
                                  "got u32[?] and u32[]."})));

// =============================================================================
// EC Scalar Multiplication — field × EC point shape inference
// =============================================================================

TEST_F(ShapeInferenceTest, EcScalarMulFieldTimesAffine) {
  // affine → jacobian: ScalarMulOp changes representation
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("bn254_sf[4]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("bn254_g1_affine[4]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred,
      ShapeInference::InferBinaryOpShape(HloOpcode::kMultiply, lhs, rhs, {}));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("bn254_g1_jacobian[4]"));
  EXPECT_TRUE(ShapeUtil::Equal(inferred, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, EcScalarMulAffineTimesField) {
  // affine → jacobian: ScalarMulOp changes representation
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("bn254_g1_affine[4]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("bn254_sf[4]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred,
      ShapeInference::InferBinaryOpShape(HloOpcode::kMultiply, lhs, rhs, {}));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("bn254_g1_jacobian[4]"));
  EXPECT_TRUE(ShapeUtil::Equal(inferred, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, EcScalarMulFieldTimesXyzz) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("bn254_sf[8]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("bn254_g1_xyzz[8]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred,
      ShapeInference::InferBinaryOpShape(HloOpcode::kMultiply, lhs, rhs, {}));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("bn254_g1_xyzz[8]"));
  EXPECT_TRUE(ShapeUtil::Equal(inferred, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, EcScalarMulXyzzTimesField) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("bn254_g1_xyzz[8]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("bn254_sf[8]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred,
      ShapeInference::InferBinaryOpShape(HloOpcode::kMultiply, lhs, rhs, {}));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("bn254_g1_xyzz[8]"));
  EXPECT_TRUE(ShapeUtil::Equal(inferred, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, EcScalarMulFieldTimesJacobian) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("bn254_sf[2]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("bn254_g1_jacobian[2]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred,
      ShapeInference::InferBinaryOpShape(HloOpcode::kMultiply, lhs, rhs, {}));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("bn254_g1_jacobian[2]"));
  EXPECT_TRUE(ShapeUtil::Equal(inferred, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, EcScalarMulJacobianTimesField) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("bn254_g1_jacobian[2]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("bn254_sf[2]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred,
      ShapeInference::InferBinaryOpShape(HloOpcode::kMultiply, lhs, rhs, {}));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("bn254_g1_jacobian[2]"));
  EXPECT_TRUE(ShapeUtil::Equal(inferred, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, EcScalarMulRejectsDimensionMismatch) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("bn254_sf[4]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("bn254_g1_xyzz[8]"));
  const absl::StatusOr<Shape> inferred =
      ShapeInference::InferBinaryOpShape(HloOpcode::kMultiply, lhs, rhs, {});
  EXPECT_FALSE(inferred.ok());
}

TEST_F(ShapeInferenceTest, EcAddRejectsMixedTypes) {
  // add(field, ec) should still be rejected
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("bn254_sf[4]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("bn254_g1_xyzz[4]"));
  const absl::StatusOr<Shape> inferred =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, lhs, rhs, {});
  EXPECT_FALSE(inferred.ok());
  EXPECT_THAT(inferred.status().message(),
              HasSubstr("Binary op add with different element types"));
}

INSTANTIATE_TEST_SUITE_P(UnboundedDynamism, UnboundedUnaryOpShapeInferenceTest,
                         ::testing::ValuesIn<UnaryOpTestCase>({
                             {"s32[?]", "s32[?]", HloOpcode::kAbs},
                             {"u32[?]", "u32[?]", HloOpcode::kClz},
                             {"u32[?]", "u32[?]", HloOpcode::kNegate},
                             {"s32[?]", "s32[?]", HloOpcode::kNot},
                             {"u32[?]", "u32[?]", HloOpcode::kPopulationCount},
                             {"s32[?]", "s32[?]", HloOpcode::kSign},
                         }));

}  // namespace zkx
