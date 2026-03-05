/* Copyright 2018 The OpenXLA Authors.
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

#include "zkx/hlo/builder/zkx_builder.h"

#include <array>
#include <functional>
#include <optional>
#include <string_view>

#include "absl/status/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/debug_options_flags.h"
#include "zkx/hlo/builder/zkx_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/parser/hlo_parser.h"
#include "zkx/hlo/testlib/pattern_matcher_gmock.h"
#include "zkx/service/pattern_matcher.h"
#include "zkx/window_util.h"

namespace zkx {

namespace {

namespace m = ::zkx::match;

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Test;

HloInstruction* GetRoot(HloModule& module) {
  return module.entry_computation()->root_instruction();
}

// TODO(b/74197823): Move the tests to service/.
absl::StatusOr<std::unique_ptr<HloModule>> BuildHloModule(ZkxBuilder& b) {
  TF_ASSIGN_OR_RETURN(ZkxComputation computation,
                      b.Build(/*remove_dynamic_dimensions=*/false));
  const HloModuleProto& proto = computation.proto();
  TF_ASSIGN_OR_RETURN(const auto& config,
                      HloModule::CreateModuleConfigFromProto(
                          proto, GetDebugOptionsFromFlags()));
  return HloModule::CreateFromProto(proto, config);
}

// Overload which explicitly specifies the root instruction.
absl::StatusOr<std::unique_ptr<HloModule>> BuildHloModule(ZkxBuilder& b,
                                                          ZkxOp root) {
  TF_ASSIGN_OR_RETURN(ZkxComputation computation,
                      b.Build(root, /*remove_dynamic_dimensions=*/false));
  const HloModuleProto& proto = computation.proto();
  TF_ASSIGN_OR_RETURN(const auto& config,
                      HloModule::CreateModuleConfigFromProto(
                          proto, GetDebugOptionsFromFlags()));
  return HloModule::CreateFromProto(proto, config);
}

// Returns the name of the test currently being run.
std::string TestName() {
  return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

TEST(ZkxBuilderTest, OnePlusTwo) {
  ZkxBuilder b(TestName());
  Add(ConstantR0<uint32_t>(&b, 1), ConstantR0<uint32_t>(&b, 2));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Add(m::Constant(), m::Constant())));
}

TEST(ZkxBuilderTest, UnaryOperatorsBuildExpectedHLO) {
  auto test_unary_operator = [&](std::function<ZkxOp(ZkxOp)> op,
                                 auto matches_pattern) {
    ZkxBuilder b(TestName());
    op(ConstantR0<int32_t>(&b, 1));
    TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
    EXPECT_THAT(GetRoot(*module), matches_pattern);
  };
  test_unary_operator([](ZkxOp x) { return -x; },
                      GmockMatch(m::Negate(m::Constant())));
  test_unary_operator([](ZkxOp x) { return ~x; },
                      GmockMatch(m::Not(m::Constant())));
}

TEST(ZkxBuilderTest, BinaryOperatorsBuildExpectedHLO) {
  auto test_binary_operator = [&](std::function<ZkxOp(ZkxOp, ZkxOp)> op,
                                  auto matches_pattern) {
    ZkxBuilder b(TestName());
    op(ConstantR0<int32_t>(&b, 1), ConstantR0<int32_t>(&b, 2));
    TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
    EXPECT_THAT(GetRoot(*module), matches_pattern);
  };

  test_binary_operator([](ZkxOp x, ZkxOp y) { return x + y; },
                       GmockMatch(m::Add(m::Constant(), m::Constant())));
  test_binary_operator([](ZkxOp x, ZkxOp y) { return x - y; },
                       GmockMatch(m::Subtract(m::Constant(), m::Constant())));
  test_binary_operator([](ZkxOp x, ZkxOp y) { return x * y; },
                       GmockMatch(m::Multiply(m::Constant(), m::Constant())));
  test_binary_operator([](ZkxOp x, ZkxOp y) { return x / y; },
                       GmockMatch(m::Divide(m::Constant(), m::Constant())));

  test_binary_operator([](ZkxOp x, ZkxOp y) { return x & y; },
                       GmockMatch(m::And(m::Constant(), m::Constant())));
  test_binary_operator([](ZkxOp x, ZkxOp y) { return x | y; },
                       GmockMatch(m::Or(m::Constant(), m::Constant())));
  test_binary_operator([](ZkxOp x, ZkxOp y) { return x ^ y; },
                       GmockMatch(m::Xor(m::Constant(), m::Constant())));
  test_binary_operator([](ZkxOp x, ZkxOp y) { return x << y; },
                       GmockMatch(m::ShiftLeft(m::Constant(), m::Constant())));
  test_binary_operator(
      [](ZkxOp x, ZkxOp y) { return x >> y; },
      GmockMatch(m::ShiftRightArithmetic(m::Constant(), m::Constant())));

  auto test_unsigned_binary_operator =
      [&](std::function<ZkxOp(ZkxOp, ZkxOp)> op, auto matches_pattern) {
        ZkxBuilder b(TestName());
        op(ConstantR0<uint32_t>(&b, 1), ConstantR0<uint32_t>(&b, 2));
        TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
        EXPECT_THAT(GetRoot(*module), matches_pattern);
      };
  test_unsigned_binary_operator(
      [](ZkxOp x, ZkxOp y) { return x >> y; },
      GmockMatch(m::ShiftRightLogical(m::Constant(), m::Constant())));
}

TEST(ZkxBuilderTest, VariadicAnd) {
  ZkxBuilder b(TestName());
  const Shape s = ShapeUtil::MakeShape(PRED, {});
  And(Parameter(&b, 0, s, "p0"), Parameter(&b, 1, s, "p1"),
      Parameter(&b, 2, s, "p2"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  // Don't specify in the test whether And(x, y, z) is right- or
  // left-associative; accept either one.
  EXPECT_THAT(GetRoot(*module),
              ::testing::AnyOf(
                  GmockMatch(m::And(m::Parameter(0),
                                    m::And(m::Parameter(1), m::Parameter(2)))),
                  GmockMatch(m::And(m::And(m::Parameter(0), m::Parameter(1)),
                                    m::Parameter(2)))));
}

TEST(ZkxBuilderTest, VariadicOr) {
  ZkxBuilder b(TestName());
  const Shape s = ShapeUtil::MakeShape(PRED, {});
  Or(Parameter(&b, 0, s, "p0"), Parameter(&b, 1, s, "p1"),
     Parameter(&b, 2, s, "p2"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  // Don't specify in the test whether Or(x, y, z) is right- or
  // left-associative; accept either one.
  EXPECT_THAT(GetRoot(*module),
              ::testing::AnyOf(
                  GmockMatch(m::Or(m::Parameter(0),
                                   m::Or(m::Parameter(1), m::Parameter(2)))),
                  GmockMatch(m::Or(m::Or(m::Parameter(0), m::Parameter(1)),
                                   m::Parameter(2)))));
}

TEST(ZkxBuilderTest, ParamPlusConstantHasScalarBroadcast) {
  ZkxBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {3, 5}), "x");
  Add(x, ConstantR0<uint32_t>(&b, 1));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Add(m::Parameter(), m::Broadcast(m::Constant()))));
}

TEST(ZkxBuilderTest, ParamPlusConstantHasScalarBroadcastReversed) {
  ZkxBuilder b(TestName());
  const ZkxOp x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {3, 5}), "x");
  Add(ConstantR0<uint32_t>(&b, 1), x);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Add(m::Broadcast(m::Constant()), m::Parameter())));
}

TEST(ZkxBuilderTest, ParamPlusParamHasBroadcast) {
  ZkxBuilder b(TestName());
  const auto& x_shape = ShapeUtil::MakeShape(S32, {2, 4, 6});
  const auto& y_shape = ShapeUtil::MakeShape(S32, {2, 4});
  auto x = Parameter(&b, 0, x_shape, "x");
  auto y = Parameter(&b, 1, y_shape, "y");
  auto add = Add(x, y, /*broadcast_dimensions=*/{0, 1});

  TF_ASSERT_OK_AND_ASSIGN(const auto add_shape, b.GetShape(add));
  EXPECT_TRUE(ShapeUtil::Equal(add_shape, x_shape));

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(
      GetRoot(*module),
      GmockMatch(m::Add(m::Parameter(0), m::Broadcast(m::Parameter(1)))));
}

TEST(ZkxBuilderTest, XPlusX) {
  ZkxBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(S32, {1, 3, 5, 7}), "x");
  Add(x, x);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Add(m::Parameter(0), m::Parameter(0))));
}

TEST(ZkxBuilderTest, TestBinaryOpImplicitBroadcast) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("u32[1]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("u32[2, 2]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[2,2]"));
  Add(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
      /*broadcast_dimensions=*/{1});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, TestBinaryOpImplicitBroadcastBounded) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("u32[1]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("u32[<=2, <=2]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[<=2, <=2]"));
  Add(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
      /*broadcast_dimensions=*/{1});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, ShapeInferenceError) {
  ZkxBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {2, 4, 6}), "x");
  auto y = Parameter(&b, 1, ShapeUtil::MakeShape(U32, {2, 4}), "y");
  Add(x, y);
  auto statusor = BuildHloModule(b);
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Shapes must be equal rank"));
}

TEST(ZkxBuilderTest, DynamicDimensionReshapeToR0) {
  ZkxBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {1}), "x");
  auto y = Parameter(&b, 1, ShapeUtil::MakeShape(S32, {}), "dyn_dim");
  auto dx = SetDimensionSize(x, y, 0);
  Reshape(dx, {});
  auto statusor = BuildHloModule(b);
  ASSERT_TRUE(statusor.ok());
}

TEST(ZkxBuilderTest, ParameterAlreadyRegistered) {
  ZkxBuilder b_call("add");
  Parameter(&b_call, 0, ShapeUtil::MakeShape(PRED, {}), "x");

  ZkxBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(PRED, {}), "x");
  auto y = Parameter(&b, 0, ShapeUtil::MakeShape(PRED, {}), "y");
  Add(x, y);
  auto statusor = BuildHloModule(b);
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("parameter 0 already registered"));
}

TEST(ZkxBuilderTest, Call) {
  ZkxBuilder b_call("the_only_to_apply");
  auto p0 = Parameter(&b_call, 0, ShapeUtil::MakeShape(U32, {}), "p0");
  auto p1 = Parameter(&b_call, 1, ShapeUtil::MakeShape(U32, {}), "p1");
  Add(p0, p1);
  TF_ASSERT_OK_AND_ASSIGN(const auto call, b_call.Build());
  ZkxBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {}), "x");
  auto y = Parameter(&b, 1, ShapeUtil::MakeShape(U32, {}), "y");
  auto one = ConstantR0<uint32_t>(&b, 1);
  auto two = ConstantR0<uint32_t>(&b, 2);
  Add(Call(&b, call, {x, y}), Call(&b, call, {one, two}));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Add(m::Call(m::Parameter(), m::Parameter()),
                                m::Call(m::Constant(), m::Constant()))));
}

TEST(ZkxBuilderTest, CompositeCall) {
  ZkxBuilder b(TestName());
  const Shape shape = ShapeUtil::MakeShape(U32, {});

  ZkxBuilder bsum(TestName());
  Add(Parameter(&bsum, 0, shape, "arg0"), Parameter(&bsum, 1, shape, "arg1"));
  TF_ASSERT_OK_AND_ASSIGN(const ZkxComputation computation, bsum.Build());

  std::vector<ZkxOp> operands = {Parameter(&b, 0, shape, "arg0"),
                                 Parameter(&b, 1, shape, "arg1")};
  CompositeCall(&b, computation, absl::MakeSpan(operands),
                /*name=*/"foo.bar",
                /*attributes=*/"{n = 1 : i32, tensor = dense<1> : tensor<i32>}",
                /*version=*/1);

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Call(m::Parameter(), m::Parameter())));
}

TEST(ZkxBuilderTest, CompositeCallFrontendAttributesStayLocal) {
  ZkxBuilder b(TestName());
  const Shape shape = ShapeUtil::MakeShape(U32, {});

  ZkxBuilder bsum(TestName());
  Add(Parameter(&bsum, 0, shape, "arg0"), Parameter(&bsum, 1, shape, "arg1"));
  TF_ASSERT_OK_AND_ASSIGN(const ZkxComputation computation, bsum.Build());

  std::vector<ZkxOp> operands = {Parameter(&b, 0, shape, "arg0"),
                                 Parameter(&b, 1, shape, "arg1")};
  CompositeCall(&b, computation, absl::MakeSpan(operands),
                /*name=*/"foo.bar",
                /*attributes=*/"{n = 1 : i32, tensor = dense<1> : tensor<i32>}",
                /*version=*/1);
  Add(operands[0], operands[1]);

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_TRUE(GetRoot(*module)->frontend_attributes().map().empty());
}

TEST(ZkxBuilderTest, CompositeCallMissingName) {
  ZkxBuilder b(TestName());
  const Shape shape = ShapeUtil::MakeShape(U32, {});

  ZkxBuilder bsum(TestName());
  Add(Parameter(&bsum, 0, shape, "arg0"), Parameter(&bsum, 1, shape, "arg1"));
  TF_ASSERT_OK_AND_ASSIGN(const ZkxComputation computation, bsum.Build());

  std::vector<ZkxOp> operands = {Parameter(&b, 0, shape, "arg0"),
                                 Parameter(&b, 1, shape, "arg1")};
  CompositeCall(&b, computation, absl::MakeSpan(operands), /*name=*/"",
                /*attributes=*/"{n = 1 : i32, tensor = dense<1> : tensor<i32>}",
                /*version=*/1);

  auto statusor = BuildHloModule(b);
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("A composite call op must have frontend attributes "
                        "with key composite.name whose value is non-empty"));
}

TEST(ZkxBuilderTest, CompositeCallMissingAttribute) {
  ZkxBuilder b(TestName());
  const Shape shape = ShapeUtil::MakeShape(U32, {});

  ZkxBuilder bsum(TestName());
  Add(Parameter(&bsum, 0, shape, "arg0"), Parameter(&bsum, 1, shape, "arg1"));
  TF_ASSERT_OK_AND_ASSIGN(const ZkxComputation computation, bsum.Build());

  std::vector<ZkxOp> operands = {Parameter(&b, 0, shape, "arg0"),
                                 Parameter(&b, 1, shape, "arg1")};
  CompositeCall(&b, computation, absl::MakeSpan(operands), /*name=*/"foo.bar",
                /*attributes=*/"", /*version=*/1);

  auto statusor = BuildHloModule(b);
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr(
          "A composite call op must have frontend attributes with key "
          "composite.attributes whose value is default: {} or non-empty"));
}

TEST(ZkxBuilderTest, CompositeCallNonNegativeVersion) {
  ZkxBuilder b(TestName());

  FrontendAttributes frontend_attributes = b.frontend_attributes();
  frontend_attributes.mutable_map()->insert({"foo", "bar"});
  b.SetFrontendAttributes(frontend_attributes);

  const Shape shape = ShapeUtil::MakeShape(U32, {});

  ZkxBuilder bsum(TestName());
  Add(Parameter(&bsum, 0, shape, "arg0"), Parameter(&bsum, 1, shape, "arg1"));
  TF_ASSERT_OK_AND_ASSIGN(const ZkxComputation computation, bsum.Build());

  std::vector<ZkxOp> operands = {Parameter(&b, 0, shape, "arg0"),
                                 Parameter(&b, 1, shape, "arg1")};
  CompositeCall(&b, computation, absl::MakeSpan(operands),
                /*name=*/"foo.bar",
                /*attributes=*/"{n = 1 : i32, tensor = dense<1> : tensor<i32>}",
                /*version=*/-1);

  auto statusor = BuildHloModule(b);
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("A composite call op must have frontend attributes "
                        "with a composite.version whose value is a "
                        "non-negative integer but got: -1"));
}

TEST(ZkxBuilderTest, CompositeCallOptionalVersionAndAttribute) {
  ZkxBuilder b(TestName());
  const Shape shape = ShapeUtil::MakeShape(U32, {});

  ZkxBuilder bsum(TestName());
  Add(Parameter(&bsum, 0, shape, "arg0"), Parameter(&bsum, 1, shape, "arg1"));
  TF_ASSERT_OK_AND_ASSIGN(const ZkxComputation computation, bsum.Build());

  std::vector<ZkxOp> operands = {Parameter(&b, 0, shape, "arg0"),
                                 Parameter(&b, 1, shape, "arg1")};
  CompositeCall(&b, computation, absl::MakeSpan(operands), /*name=*/"foo.bar");

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  ASSERT_THAT(GetRoot(*module),
              GmockMatch(m::Call(m::Parameter(), m::Parameter())));
  ASSERT_TRUE(GetRoot(*module)->frontend_attributes().map().contains(
      "composite.attributes"));
  EXPECT_EQ(
      GetRoot(*module)->frontend_attributes().map().at("composite.attributes"),
      "{}");
  EXPECT_EQ(
      GetRoot(*module)->frontend_attributes().map().at("composite.version"),
      "0");
}

TEST(ZkxBuilderTest, CompositeCallWithExtraFrontendAttributes) {
  ZkxBuilder b(TestName());

  FrontendAttributes frontend_attributes = b.frontend_attributes();
  frontend_attributes.mutable_map()->insert({"foo", "bar"});
  b.SetFrontendAttributes(frontend_attributes);

  const Shape shape = ShapeUtil::MakeShape(U32, {});

  ZkxBuilder bsum(TestName());
  Add(Parameter(&bsum, 0, shape, "arg0"), Parameter(&bsum, 1, shape, "arg1"));
  TF_ASSERT_OK_AND_ASSIGN(const ZkxComputation computation, bsum.Build());

  std::vector<ZkxOp> operands = {Parameter(&b, 0, shape, "arg0"),
                                 Parameter(&b, 1, shape, "arg1")};
  CompositeCall(&b, computation, absl::MakeSpan(operands),
                /*name=*/"foo.bar",
                /*attributes=*/"{n = 1 : i32, tensor = dense<1> : tensor<i32>}",
                /*version=*/1);

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Call(m::Parameter(), m::Parameter())));
  ASSERT_TRUE(GetRoot(*module)->frontend_attributes().map().contains("foo"));
  EXPECT_EQ(GetRoot(*module)->frontend_attributes().map().at("foo"), "bar");
}

TEST(ZkxBuilderTest, BinopHasDegenerateBroadcast) {
  ZkxBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {1, 2, 3}), "x");
  auto y = Parameter(&b, 1, ShapeUtil::MakeShape(U32, {1, 2, 1}), "y");
  Add(x, y);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));

  // Expected:
  //
  //  x: u32[1,2,3]  y: u32[1,2,1]
  //      |               |
  //      |          reshape: u32[1,2]
  //      |               |
  //      |          broadcast: u32[1,2,3]
  //       \             /
  //            add
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Add(m::Parameter(0),
                                m::Broadcast(m::Reshape(m::Parameter(1))))));
}

TEST(ZkxBuilderTest, BinopHasInDimAndDegenerateBroadcast) {
  ZkxBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {2, 3}), "x");
  auto y = Parameter(&b, 1, ShapeUtil::MakeShape(U32, {2, 1, 4}), "y");
  Add(x, y, /*broadcast_dimensions=*/{0, 1});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));

  // The binary operation has in-dim broadcast and degenerate broadcast, should
  // first do the in-dim broadcast then convert the degenerate broadcast into a
  // reshape and a broadcast.
  //
  // Expected:
  //
  //  x: u32[2,3]            y: u32[2,1,4]
  //      |                        |
  //  broadcast: u32[2,3,4]  reshape: u32[2,4]
  //      |                        |
  //      |                  broadcast: u32[2,3,4]
  //       \                      /
  //                 add
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Add(m::Broadcast(m::Parameter(0)),
                                m::Broadcast(m::Reshape(m::Parameter(1))))));
}

TEST(ZkxBuilderTest, BroadcastInDim) {
  ZkxBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {2, 3}), "x");
  BroadcastInDim(x, {2, 4, 3},
                 /*broadcast_dimensions=*/{0, 2});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module), GmockMatch(m::Broadcast()));
}

TEST(ZkxBuilderTest, BroadcastInDimWithDegeneratedDim) {
  ZkxBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {2, 1, 4}), "x");
  BroadcastInDim(x, {2, 3, 4},
                 /*broadcast_dimensions=*/{0, 1, 2});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Broadcast(m::Reshape(m::Broadcast()))));
}

TEST(ZkxBuilderTest, BroadcastInDimWithBoundedDim) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape shape, ParseShape("u32[2, <=3]"));
  auto x = Parameter(&b, 0, shape, "x");
  BroadcastInDim(x, {1, 2, 3},
                 /*broadcast_dimensions=*/{1, 2});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module), GmockMatch(m::Broadcast()));
}

TEST(ZkxBuilderTest, BroadcastInDimWithNegativeSize) {
  ZkxBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {2, 1, 4}), "x");
  BroadcastInDim(x, {-3, 3, 4},
                 /*broadcast_dimensions=*/{0, 1, 2});
  auto statusor = BuildHloModule(b);
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(), HasSubstr("invalid shape"));
}

TEST(ZkxBuilderTest, OperandFromWrongBuilder) {
  ZkxBuilder b1("b1");
  auto p0 = Parameter(&b1, 0, ShapeUtil::MakeShape(U32, {}), "p0");
  ZkxBuilder builder("main");
  auto p = Parameter(&builder, 0, ShapeUtil::MakeShape(U32, {}), "p");
  Add(p, p0);
  auto statusor = builder.Build();
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr(
          "built by builder 'b1', but is trying to use it in builder 'main'"));
}

TEST(ZkxBuilderTest, ReshapeDefaultOrder) {
  ZkxBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {2, 3, 5, 7}), "x");
  Reshape(x, /*new_sizes=*/{6, 35});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module), GmockMatch(m::Reshape(m::Parameter())));
}

TEST(ZkxBuilderTest, ReshapeHasTranspose) {
  ZkxBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {2, 3, 5, 7}), "x");
  Reshape(x, /*dimensions=*/{3, 2, 1, 0}, /*new_sizes=*/{6, 35});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Reshape(m::Transpose(m::Parameter()))));
}

TEST(ZkxBuilderTest, Transpose) {
  ZkxBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {5, 7}), "x");
  Transpose(x, /*permutation=*/{1, 0});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module), GmockMatch(m::Transpose(m::Parameter())));
}

// TODO(chokobole): Add test. Dependency: ZkxBuilder::AllGather
// TEST(ZkxBuilderTest, AllGatherR1) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::AllGather
// TEST(ZkxBuilderTest, AllGatherR2) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::AllGather
// TEST(ZkxBuilderTest, AllGatherWithTuple) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::AllGatherTuple
// TEST(ZkxBuilderTest, AllGatherTuple) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::ReduceScatter
// TEST(ZkxBuilderTest, ReduceScatter) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::ReduceScatter
// TEST(ZkxBuilderTest, ReduceScatterWithTuple) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::AlltoAll
// TEST(ZkxBuilderTest, AllToAll) {

// Test the special case where split_dimension is the same as concat_dimension.
// TODO(chokobole): Add test. Dependency: ZkxBuilder::AlltoAll
// TEST(ZkxBuilderTest, AllToAllSpecial) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::AllToAllTuple
// TEST(ZkxBuilderTest, AllToAllTuple) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::AllReduceTuple
// TEST(ZkxBuilderTest, AllReduceTuple) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::CollectiveBroadcast
// TEST(ZkxBuilderTest, CollectiveBroadcast) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::CollectivePermute
// TEST(ZkxBuilderTest, CollectivePermute) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::MultiCollectivePermute
// TEST(ZkxBuilderTest, CombinedCollectivePermute) {

TEST(ZkxBuilderTest, GetDimensionSize) {
  ZkxBuilder b(TestName());
  auto x =
      Parameter(&b, 0, ShapeUtil::MakeShape(U32, {5, 7}, {false, true}), "x");
  GetDimensionSize(x, 1);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_EQ(GetRoot(*module)->opcode(), HloOpcode::kGetDimensionSize);
}

TEST(ZkxBuilderTest, GetDimensionSizeConstant) {
  ZkxBuilder b(TestName());
  auto x =
      Parameter(&b, 0, ShapeUtil::MakeShape(U32, {5, 7}, {false, true}), "x");
  // Get dimension size from a constant dimension gives us a constant.
  GetDimensionSize(x, 0);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_EQ(GetRoot(*module)->opcode(), HloOpcode::kConstant);
}

TEST(ZkxBuilderTest, ReportError) {
  ZkxBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {5, 7}), "x");
  Add(b.ReportError(absl::InvalidArgumentError("a test error")), x);
  auto statusor = b.Build();
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(), HasSubstr("a test error"));
}

TEST(ZkxBuilderTest, ReportErrorOrReturnHandlesNonErrors) {
  ZkxBuilder b(TestName());
  absl::StatusOr<ZkxOp> op(ConstantR0<uint32_t>(&b, 1));
  Add(b.ReportErrorOrReturn(op), ConstantR0<uint32_t>(&b, 2));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Add(m::Constant(), m::Constant())));
}

TEST(ZkxBuilderTest, ReportErrorOrReturnHandlesErrors) {
  ZkxBuilder b(TestName());
  absl::StatusOr<ZkxOp> op(absl::InvalidArgumentError("a test error"));
  Add(b.ReportErrorOrReturn(op), ConstantR0<uint32_t>(&b, 2));
  auto statusor = b.Build();
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(), HasSubstr("a test error"));
}

TEST(ZkxBuilderTest, BuildWithSpecificRoot) {
  ZkxBuilder b(TestName());
  const ZkxOp constant = ConstantR0<uint32_t>(&b, 1);
  Add(constant, ConstantR0<uint32_t>(&b, 2));
  TF_ASSERT_OK_AND_ASSIGN(const auto module,
                          BuildHloModule(b, /*root=*/constant));
  EXPECT_THAT(GetRoot(*module), GmockMatch(m::Constant()));
}

TEST(ZkxBuilderTest, BuildWithSpecificRootAndMultipleParameters) {
  // Specifying a particular root in Build should still include all entry
  // parameters.
  ZkxBuilder b(TestName());
  const Shape shape = ShapeUtil::MakeShape(U32, {42, 123});
  const ZkxOp x = Parameter(&b, 0, shape, "x");
  const ZkxOp y = Parameter(&b, 1, shape, "y");
  const ZkxOp z = Parameter(&b, 2, shape, "z");
  Add(x, Sub(y, z));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b, /*root=*/x));
  EXPECT_THAT(GetRoot(*module), GmockMatch(m::Parameter()));
  EXPECT_EQ(module->entry_computation()->num_parameters(), 3);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 5);
}

TEST(ZkxBuilderTest, BuildWithSpecificRootWithWrongBuilder) {
  ZkxBuilder b(TestName());
  ZkxBuilder other_b(TestName());
  const Shape shape = ShapeUtil::MakeShape(U32, {42, 123});

  Parameter(&b, 0, shape, "param");
  const ZkxOp other_param = Parameter(&other_b, 0, shape, "other_param");

  absl::Status status = b.Build(other_param).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("root operation is not in this computation"));
}

TEST(ZkxBuilderTest, ProtoMatches) {
  std::vector<ZkxComputation> computations;
  const int n = 2;
  computations.reserve(n);
  for (int i = 0; i < n; ++i) {
    ZkxBuilder b_call("the_only_to_apply");
    auto p0 = Parameter(&b_call, 0, ShapeUtil::MakeShape(U32, {}), "p0");
    auto p1 = Parameter(&b_call, 1, ShapeUtil::MakeShape(U32, {}), "p1");
    Add(p0, Add(p1, p0));
    TF_ASSERT_OK_AND_ASSIGN(const auto call, b_call.Build());
    ZkxBuilder b(TestName());
    auto x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {}), "x");
    auto y = Parameter(&b, 1, ShapeUtil::MakeShape(U32, {}), "y");
    auto one = ConstantR0<uint32_t>(&b, 1);
    auto two = ConstantR0<uint32_t>(&b, 2);
    Add(Call(&b, call, {x, y}), Call(&b, call, {one, two}));
    computations.push_back(b.Build().value());
  }
  auto c0_string = computations[0].proto().SerializeAsString();
  auto c1_string = computations[1].proto().SerializeAsString();
  EXPECT_EQ(c0_string, c1_string);
}

TEST(ZkxBuilderTest, DynamicParameter) {
  ZkxBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {5}), ShapeUtil::MakeShape(U32, {6}, {true})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  Parameter(&b, 1, ShapeUtil::MakeShape(U32, {}), "p1");
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b, /*root=*/p0));
  const Shape& param_shape = module->entry_computation()
                                 ->parameter_instruction(0)
                                 ->shape()
                                 .tuple_shapes(1);
  EXPECT_TRUE(param_shape.is_dynamic_dimension(0));
}

TEST(ZkxBuilderTest, SetDimensionSize) {
  ZkxBuilder b(TestName());
  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {10}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(S32, {}), "p1");
  auto set_dim_size = SetDimensionSize(p0, p1, 0);
  TF_ASSERT_OK_AND_ASSIGN(const auto module,
                          BuildHloModule(b, /*root=*/set_dim_size));
  const Shape& root_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(root_shape.is_dynamic_dimension(0));
}

TEST(ZkxBuilderTest, RemoveDynamicDimension) {
  ZkxBuilder b(TestName());
  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {10}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(S32, {}), "p1");
  auto set_dim_size = SetDimensionSize(p0, p1, 0);
  auto remove_dim_size = RemoveDynamicDimension(set_dim_size, 0);
  TF_ASSERT_OK_AND_ASSIGN(const auto module,
                          BuildHloModule(b, /*root=*/remove_dim_size));
  const Shape& root_shape =
      module->entry_computation()->root_instruction()->shape();
  // Dynamic dimension has been removed.
  EXPECT_FALSE(root_shape.is_dynamic_dimension(0));
}

TEST(ZkxBuilderTest, RemoveDynamicDimensionMultiDims) {
  ZkxBuilder b(TestName());
  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {10, 10}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(S32, {}), "p1");
  auto set_dim_size = SetDimensionSize(p0, p1, 0);
  set_dim_size = SetDimensionSize(set_dim_size, p1, 1);
  auto remove_dim_size = RemoveDynamicDimension(set_dim_size, 0);
  remove_dim_size = RemoveDynamicDimension(remove_dim_size, 1);
  TF_ASSERT_OK_AND_ASSIGN(const auto module,
                          BuildHloModule(b, /*root=*/remove_dim_size));
  const Shape& root_shape =
      module->entry_computation()->root_instruction()->shape();
  // Dynamic dimensions are removed.
  EXPECT_FALSE(root_shape.is_dynamic_dimension(0));
  EXPECT_FALSE(root_shape.is_dynamic_dimension(1));
}

TEST(ZkxBuilderTest, DynamicUnary) {
  ZkxBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {5}, {true}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto gte = GetTupleElement(p0, 0);
  Neg(gte);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(result_shape.is_dynamic_dimension(0));
}

TEST(ZkxBuilderTest, DynamicBinary) {
  ZkxBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {5}, {true}),
       ShapeUtil::MakeShape(U32, {5}, {true}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto gte0 = GetTupleElement(p0, 0);
  auto gte1 = GetTupleElement(p0, 1);
  Add(gte0, gte1);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(result_shape.is_dynamic_dimension(0));
}

TEST(ZkxBuilderTest, DynamicBinaryHasBroadcast) {
  ZkxBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {5, 4}, {true, false}),
       ShapeUtil::MakeShape(U32, {5}, {true}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto gte0 = GetTupleElement(p0, 0);
  auto gte1 = GetTupleElement(p0, 1);
  Add(gte0, gte1, {0});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_THAT(result_shape.dynamic_dimensions(), ElementsAre(true, false))
      << result_shape;
}

TEST(ZkxBuilderTest, DynamicBroadcast) {
  ZkxBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {5, 4}, {true, false}),
       ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto gte = GetTupleElement(p0, 0);
  BroadcastInDim(gte, /*out_dim_size=*/{3, 5, 4},
                 /*broadcast_dimensions=*/{1, 2});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_THAT(result_shape.dynamic_dimensions(),
              ElementsAre(false, true, false))
      << result_shape;
}

TEST(ZkxBuilderTest, DynamicBinaryHasDegenerateBroadcast) {
  ZkxBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {10}, {true}),
       ShapeUtil::MakeShape(U32, {1, 15}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto gte0 = GetTupleElement(p0, 0);
  auto gte1 = GetTupleElement(p0, 1);
  Add(gte0, gte1, /*broadcast_dimensions=*/{0});  // u32[<=10, 15]
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_THAT(result_shape.dynamic_dimensions(), ElementsAre(true, false))
      << result_shape;
}

TEST(ZkxBuilderTest, DynamicSelectOnlyPredDynamic) {
  ZkxBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(PRED, {10}, {true}),
       ShapeUtil::MakeShape(U32, {10}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto gte0 = GetTupleElement(p0, 0);
  auto gte1 = GetTupleElement(p0, 1);

  Select(gte0, gte1, gte1);

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_THAT(result_shape.dynamic_dimensions(), ElementsAre(true))
      << result_shape;
}

TEST(ZkxBuilderTest, SelectIntoConditional) {
  ZkxBuilder b(TestName());
  const Shape selector_shape = ShapeUtil::MakeShape(PRED, {});
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(S32, {}), ShapeUtil::MakeShape(U32, {})});
  const ZkxOp p0 = Parameter(&b, 0, selector_shape, "p0");
  const ZkxOp p1 = Parameter(&b, 1, tuple_param_shape, "p1");
  const ZkxOp p2 = Parameter(&b, 2, tuple_param_shape, "p2");

  Select(p0, p1, p2);

  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Conditional(m::Parameter(0), m::Parameter(1),
                                        m::Parameter(2))));
  EXPECT_THAT(module->entry_computation()
                  ->root_instruction()
                  ->branch_computation(0)
                  ->root_instruction(),
              GmockMatch(m::Parameter(0)));
  EXPECT_THAT(module->entry_computation()
                  ->root_instruction()
                  ->branch_computation(1)
                  ->root_instruction(),
              GmockMatch(m::Parameter(0)));
}

TEST(ZkxBuilderTest, DynamicPad) {
  ZkxBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(S32, {5, 4}, {true, false}),
       ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto pad_val = ConstantR0<int32_t>(&b, -1);
  auto gte = GetTupleElement(p0, 0);
  PaddingConfig padding_config;
  for (int i = 0; i < 2; i++) {
    auto dimension = padding_config.add_dimensions();
    dimension->set_edge_padding_low(0);
    dimension->set_edge_padding_high(0);
  }
  Pad(gte, pad_val, padding_config);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_THAT(result_shape.dynamic_dimensions(), ElementsAre(true, false))
      << result_shape;
}

TEST(ZkxBuilderTest, DynamicDot) {
  ZkxBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {2, 3, 4}, {true, true, false}),
       ShapeUtil::MakeShape(U32, {2, 4, 5}, {true, false, false}),
       ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");

  auto lhs = GetTupleElement(p0, 0);
  auto rhs = GetTupleElement(p0, 1);
  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(2);
  dnums.add_rhs_contracting_dimensions(1);
  dnums.add_lhs_batch_dimensions(0);
  dnums.add_rhs_batch_dimensions(0);
  DotGeneral(lhs, rhs, dnums);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_THAT(result_shape.dynamic_dimensions(), ElementsAre(true, true, false))
      << result_shape;
}

TEST(ZkxBuilderTest, DynamicReduce) {
  ZkxBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {5, 4, 3}, {false, true, false}),
       ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto init = ConstantR0<uint32_t>(&b, 0);
  auto gte = GetTupleElement(p0, 0);
  ZkxBuilder bsum(TestName());
  Add(Parameter(&bsum, 0, ShapeUtil::MakeShape(U32, {}), "x"),
      Parameter(&bsum, 1, ShapeUtil::MakeShape(U32, {}), "y"));
  TF_ASSERT_OK_AND_ASSIGN(const auto sum, bsum.Build());
  Reduce(gte, init, sum, {0});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_THAT(result_shape.dynamic_dimensions(), ElementsAre(true, false))
      << result_shape;
}

// TODO(batzor): Add test. Dependency: ReduceWindow with Padding overload
// TEST(ZkxBuilderTest, DynamicReduceWindow) {

// TODO(batzor): Add test. Dependency: ReduceWindow with Padding overload
// TEST(ZkxBuilderTest, VariadicDynamicReduceWindow) {

TEST(ZkxBuilderTest, DynamicReshape) {
  ZkxBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {2, 3, 4, 5, 6},
                            {false, false, true, true, false}),
       ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto gte = GetTupleElement(p0, 0);  // u32[2, 3, <=4, <=5, 6]
  Reshape(gte, /*new_sizes=*/{6, 4, 5, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(result_shape.is_dynamic_dimension(1));
  EXPECT_TRUE(result_shape.is_dynamic_dimension(2));
  EXPECT_THAT(result_shape.dynamic_dimensions(),
              ElementsAre(false, true, true, false, false))
      << result_shape;
}

TEST(ZkxBuilderTest, DynamicSelect) {
  ZkxBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {4, 5, 6}, {false, true, false}),
       ShapeUtil::MakeShape(U32, {4, 5, 6}, {false, true, false}),
       ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto pred = Parameter(&b, 1, ShapeUtil::MakeShape(PRED, {}), "pred");
  auto gte0 = GetTupleElement(p0, 0);
  auto gte1 = GetTupleElement(p0, 1);
  Select(pred, gte0, gte1);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(result_shape.is_dynamic_dimension(1));
  EXPECT_FALSE(result_shape.is_dynamic_dimension(2));
  EXPECT_THAT(result_shape.dynamic_dimensions(),
              ElementsAre(false, true, false))
      << result_shape;
}

TEST(ZkxBuilderTest, DynamicSelectNotCompatible) {
  ZkxBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {4, 5, 6}, {false, true, false}),
       ShapeUtil::MakeShape(U32, {4, 5, 6}, {false, false, true}),
       ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto pred = Parameter(&b, 1, ShapeUtil::MakeShape(PRED, {}), "pred");
  auto gte0 = GetTupleElement(p0, 0);  // u32[4,<=5,6]
  auto gte1 = GetTupleElement(p0, 1);  // u32[4,5,<=6]
  Select(pred, gte0, gte1);
  absl::Status status = BuildHloModule(b).status();
  ASSERT_TRUE(status.ok());
}

TEST(ZkxBuilderTest, DynamicTranspose) {
  ZkxBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U32, {3, 5}, {true, false}),
       ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto gte = GetTupleElement(p0, 0);
  Transpose(gte, /*permutation=*/{1, 0});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_THAT(result_shape.dynamic_dimensions(), ElementsAre(false, true))
      << result_shape;
}

// TODO(chokobole): Add test. Dependency: ZkxBuilder::DotGeneral with
// preferred_element_type.
// TEST(ZkxBuilderTest, DotWithPreferredElementType) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::SparseDot
// TEST(ZkxBuilderTest, SparseDot) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::RaggedDot
// TEST(ZkxBuilderTest, RaggedDotNonContractingWithPreferredElementType) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::RaggedDot
// TEST(ZkxBuilderTest, RaggedDotContractingWithPreferredElementType) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::AfterAll
// TEST(ZkxBuilderTest, AfterAllWithNonTokenOperands) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::AfterAll
// TEST(ZkxBuilderTest, AfterAllWithNoInputs) {

TEST(ZkxBuilderTest, CheckInputOutputAlias) {
  ZkxBuilder b(TestName());
  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {8, 4}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(U32, {8, 4}), "p1");
  auto add = Add(p0, p1);
  auto sub = Sub(p0, p1);
  auto root = Tuple(&b, {add, sub});

  b.SetUpAlias({1}, 0, {});
  b.SetUpAlias({0}, 1, {});

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b, root));

  const HloInputOutputAliasConfig& config = module->input_output_alias_config();
  EXPECT_TRUE(config.ParameterHasAlias(0, {}));
  EXPECT_TRUE(config.ParameterHasAlias(1, {}));

  auto alias_p0 = config.GetAliasedOutput(0, {});
  ASSERT_TRUE(alias_p0.has_value());
  EXPECT_EQ(*alias_p0, ShapeIndex({1}));

  auto alias_p1 = config.GetAliasedOutput(1, {});
  ASSERT_TRUE(alias_p1.has_value());
  EXPECT_EQ(*alias_p1, ShapeIndex({0}));
}

TEST(ZkxBuilderTest, CheckBufferDonor) {
  ZkxBuilder b(TestName());
  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {8, 4}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(U32, {8, 4}), "p1");
  auto add = Add(p0, p1);
  auto sub = Sub(p0, p1);
  auto root = Tuple(&b, {add, sub});

  b.AddBufferDonor(0, {});

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b, root));

  const HloBufferDonorConfig& config = module->buffer_donor_config();
  EXPECT_TRUE(config.ParameterIsBufferDonor(0, {}));
  EXPECT_FALSE(config.ParameterIsBufferDonor(1, {}));
}

TEST(ZkxBuilderTest, ConstantLiteral) {
  ZkxBuilder b(TestName());
  ConstantR1<uint32_t>(&b, {0, 1});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const HloInstruction* root = GetRoot(*module);
  ASSERT_THAT(root, GmockMatch(m::Constant()));
}

TEST(ZkxBuilderTest, InvalidInputOutputAliasBufferDonor) {
  ZkxBuilder b(TestName());

  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {8, 4}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(U32, {8, 4}), "p1");
  auto add = Add(p0, p1);
  auto sub = Sub(p0, p1);
  auto root = Tuple(&b, {add, sub});

  b.SetUpAlias({1}, 0, {});
  b.AddBufferDonor(0, {});

  auto statusor = BuildHloModule(b, root);
  EXPECT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("is already aliased with one output, thus it cannot be "
                        "added as a buffer donor for any output."));
}

TEST(ZkxBuilderTest, ValidInputOutputAliasBufferDonor) {
  ZkxBuilder b(TestName());

  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {8, 4}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(U32, {8, 4}), "p1");
  auto add = Add(p0, p1);
  auto sub = Sub(p0, p1);
  auto root = Tuple(&b, {add, sub});

  b.SetUpAlias({1}, 0, {});
  b.AddBufferDonor(1, {});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b, root));

  const HloInputOutputAliasConfig& io_alias_config =
      module->input_output_alias_config();
  const HloBufferDonorConfig& buffer_donor_config =
      module->buffer_donor_config();

  EXPECT_TRUE(io_alias_config.ParameterHasAlias(0, {}));
  EXPECT_FALSE(io_alias_config.ParameterHasAlias(1, {}));
  EXPECT_FALSE(buffer_donor_config.ParameterIsBufferDonor(0, {}));
  EXPECT_TRUE(buffer_donor_config.ParameterIsBufferDonor(1, {}));

  auto alias_p0 = io_alias_config.GetAliasedOutput(0, {});
  ASSERT_TRUE(alias_p0.has_value());
  EXPECT_EQ(*alias_p0, ShapeIndex({1}));
}

void ExpectAttributesMatch(const FrontendAttributes& attr,
                           const FrontendAttributes& ref) {
  EXPECT_EQ(ref.map_size(), attr.map_size());
  for (auto reference : ref.map()) {
    auto other = attr.map().find(reference.first);
    EXPECT_NE(other, attr.map().end());
    EXPECT_EQ(other->second, reference.second);
  }
}

void ExpectInstructionsAttributesMatch(
    const HloModule& module, const std::vector<FrontendAttributes>& expected) {
  ASSERT_EQ(module.computation_count(), 1);
  auto expected_it = expected.begin();
  for (auto inst : module.entry_computation()->instructions()) {
    ASSERT_NE(expected_it, expected.end());
    ExpectAttributesMatch(inst->frontend_attributes(), *expected_it);
    expected_it++;
  }
  EXPECT_EQ(expected_it, expected.end());
}

TEST(ZkxBuilderTest, SimpleSetFrontendAttributes) {
  ZkxBuilder b(TestName());
  FrontendAttributes attributes;

  ConstantR0(&b, 0);  // No attribute set

  (*attributes.mutable_map())["attr_a"] = "a";
  b.SetFrontendAttributes(attributes);
  ConstantR0(&b, 0);  // One attribute: { "attr_a": "a" }

  b.ClearFrontendAttributes();
  ConstantR0(&b, 0);  // No attribute set

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));

  std::vector<FrontendAttributes> expected{FrontendAttributes(), attributes,
                                           FrontendAttributes()};
  ExpectInstructionsAttributesMatch(*module, expected);
}

TEST(ZkxBuilderTest, ComplexSetFrontendAttributes) {
  ZkxBuilder b(TestName());

  ConstantR0(&b, 0);  // No attribute set.
  std::vector<FrontendAttributes> expected{FrontendAttributes()};

  {
    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_a"] = "a";
    b.SetFrontendAttributes(attributes);
    ConstantR0(&b, 0);  // One attribute: { "attr_a": "a" }
    expected.push_back(attributes);
  }

  {
    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_b"] = "b";
    b.SetFrontendAttributes(attributes);
    ConstantR0(&b, 0);  // One attribute: { "attr_b": "b" }
    expected.push_back(attributes);
  }

  {
    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_b"] = "b";
    (*attributes.mutable_map())["attr_c"] = "c";
    b.SetFrontendAttributes(attributes);
    ConstantR0(&b, 0);  // Two attributes: { "attr_b": "b", "attr_c": "c" }
    expected.push_back(attributes);
  }

  b.ClearFrontendAttributes();
  ConstantR0(&b, 0);  // No attribute set
  expected.push_back(FrontendAttributes());

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  ExpectInstructionsAttributesMatch(*module, expected);
}

TEST(ZkxBuilderTest, AddFrontendAttribute) {
  ZkxBuilder b(TestName());

  ConstantR0(&b, 0);
  std::vector<FrontendAttributes> expected{FrontendAttributes()};

  // One attribute: { "attr_a": "a" }
  {
    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_a"] = "a";
    b.SetFrontendAttributes(attributes);
    ConstantR0(&b, 0);
    expected.push_back(attributes);
  }

  // Two attributes: {"attra": "a", "attr_c": "c"}
  {
    auto op = ConstantR0(&b, 0);
    EXPECT_TRUE(b.SetInstructionFrontendAttribute(op, "attr_c", "c").ok());

    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_a"] = "a";
    (*attributes.mutable_map())["attr_c"] = "c";
    expected.push_back(attributes);
  }

  // Override value of existing "attr_a"
  // One attribute: { "attr_a", "a2"}
  {
    auto op = ConstantR0(&b, 0);
    EXPECT_TRUE(b.SetInstructionFrontendAttribute(op, "attr_a", "a2").ok());
    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_a"] = "a2";
    expected.push_back(attributes);
  }

  // Check "attr_a" is back to its original value
  // One attribute: { "attr_a", "a"}
  {
    auto op = ConstantR0(&b, 0);
    (void)op;
    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_a"] = "a";
    expected.push_back(attributes);
  }

  b.ClearFrontendAttributes();
  ConstantR0(&b, 0);  // No attribute set
  expected.push_back(FrontendAttributes());

  // One attribute: { "attr_d", "d"}
  {
    auto op = ConstantR0(&b, 0);
    EXPECT_TRUE(b.SetInstructionFrontendAttribute(op, "attr_d", "d").ok());
    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_d"] = "d";
    expected.push_back(attributes);
  }

  ConstantR0(&b, 0);  // No attribute set
  expected.push_back(FrontendAttributes());

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  ExpectInstructionsAttributesMatch(*module, expected);
}

// TODO(chokobole): Add test. Dependency: ZkxBuilder::sharding_builder
// TEST(ZkxBuilderTest, SetAndGetSharding) {

TEST(ZkxBuilderTest, Comparison) {
  ZkxBuilder b(TestName());
  (void)Le(ConstantR0<int32_t>(&b, 1), ConstantR0<int32_t>(&b, 2));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const HloInstruction* root = GetRoot(*module);
  ASSERT_THAT(root, GmockMatch(m::Compare(m::Constant(), m::Constant())));
}

TEST(ZkxBuilderTest, StableLookUpInstructionByHandle) {
  ZkxBuilder b(TestName());
  internal::ZkxBuilderFriend builder_friend;
  const ZkxOp le = Le(ConstantR0<int32_t>(&b, 1), ConstantR0<int32_t>(&b, 2));
  HloInstructionProto* first_op = builder_friend.GetInstruction(le);
  // Create some more instructions.
  for (int i = 0; i < 100; ++i) {
    (void)Le(ConstantR0<int32_t>(&b, 1), ConstantR0<int32_t>(&b, 2));
  }
  // Make sure first_op hasn't changed.
  HloInstructionProto* first_op_now = builder_friend.GetInstruction(le);
  EXPECT_EQ(first_op, first_op_now);
}

// TODO(chokobole): Add test. Dependency: ZkxBuilder::Outfeed
// TEST(ZkxBuilderTest, OutfeedDummyTupleSharding) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::Outfeed
// TEST(ZkxBuilderTest, OutfeedTokenSharding) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::sharding_builder
// TEST(ZkxBuilderTest, NormalizeTupleSharding) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::sharding_builder
// TEST(ZkxBuilderTest, InvalidSharding) {

//============================================================================//
// Experimental Test
//============================================================================//

TEST(ZkxBuilderTest, MhloDynamicBroadcastInDimExportSuccess) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[1, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_dimensions, ParseShape("s32[3]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_shape, ParseShape("u32[1, 2, 3]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[1, 2, 3]"));
  MhloDynamicBroadcastInDim(
      Parameter(&b, 0, operand, "operand"),
      Parameter(&b, 1, output_dimensions, "output_dimensions"),
      /*broadcast_dimensions=*/{1, 2}, output_shape);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(module->ToString(), HasSubstr("mhlo.dynamic_broadcast_in_dim"));
  EXPECT_THAT(module->ToString(), HasSubstr("broadcast_dimensions=[1,2]"));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest,
     MhloDynamicBroadcastInDimNonBroadcastDimSizeGreaterThanOne) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_dimensions, ParseShape("s32[3]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_shape, ParseShape("u32[2, 2, 3]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[2, 2, 3]"));
  MhloDynamicBroadcastInDim(
      Parameter(&b, 0, operand, "operand"),
      Parameter(&b, 1, output_dimensions, "output_dimensions"),
      /*broadcast_dimensions=*/{1, 2}, output_shape);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(module->ToString(), HasSubstr("mhlo.dynamic_broadcast_in_dim"));
  EXPECT_THAT(module->ToString(), HasSubstr("broadcast_dimensions=[1,2]"));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, MhloDynamicBroadcastInDimDynamicResultSize) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[1, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_dimensions, ParseShape("s32[3]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_shape, ParseShape("u32[1, 2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[1, 2, ?]"));
  MhloDynamicBroadcastInDim(
      Parameter(&b, 0, operand, "operand"),
      Parameter(&b, 1, output_dimensions, "output_dimensions"),
      /*broadcast_dimensions=*/{1, 2}, output_shape);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(module->ToString(), HasSubstr("mhlo.dynamic_broadcast_in_dim"));
  EXPECT_THAT(module->ToString(), HasSubstr("broadcast_dimensions=[1,2]"));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest,
     MhloDynamicBroadcastInDimInvalidOutputDimensionsElementType) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_dimensions, ParseShape("pred[3]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_shape, ParseShape("u32[2, 3, 3]"));
  MhloDynamicBroadcastInDim(
      Parameter(&b, 0, operand, "operand"),
      Parameter(&b, 1, output_dimensions, "output_dimensions"),
      /*broadcast_dimensions=*/{1, 2}, output_shape);
  EXPECT_THAT(
      BuildHloModule(b),
      StatusIs(_,
               HasSubstr("output_dimensions must be an integer type pred[3]")));
}

TEST(ZkxBuilderTest, MhloDynamicBroadcastInDimInvalidOutputDimensionsRank) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_dimensions,
                          ParseShape("s32[2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_shape, ParseShape("u32[2, 3, 3]"));
  MhloDynamicBroadcastInDim(
      Parameter(&b, 0, operand, "operand"),
      Parameter(&b, 1, output_dimensions, "output_dimensions"),
      /*broadcast_dimensions=*/{1, 2}, output_shape);
  EXPECT_THAT(
      BuildHloModule(b),
      StatusIs(_,
               HasSubstr("output_dimensions must be rank 1 but got rank 2")));
}

TEST(ZkxBuilderTest, MhloDynamicBroadcastInDimIncompatibleBroadcastSize) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_dimensions, ParseShape("s32[3]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_shape, ParseShape("u32[2, 3, 3]"));
  MhloDynamicBroadcastInDim(
      Parameter(&b, 0, operand, "operand"),
      Parameter(&b, 1, output_dimensions, "output_dimensions"),
      /*broadcast_dimensions=*/{1, 2}, output_shape);
  EXPECT_THAT(
      BuildHloModule(b),
      StatusIs(_, HasSubstr("size of operand dimension 0 (2) is not compatible "
                            "with size of result dimension 1 (3)")));
}

// TODO(chokobole): Add test. Dependency: ZkxBuilder::MhloDynamicReshape
// TEST(ZkxBuilderTest, MhloDynamicReshapeExportSuccess) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::MhloDynamicReshape
// TEST(ZkxBuilderTest, MhloDynamicReshapeIncompatibleElementType) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::MhloDynamicReshape
// TEST(ZkxBuilderTest, MhloDynamicReshapeElementCountMismatch) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::MhloDynamicReshape
// TEST(ZkxBuilderTest, MhloDynamicReshapeRankMismatch) {

//============================================================================//
// Unbounded Dynamism Test
//============================================================================//

struct UnaryOpTestCase {
  std::string operand;
  std::string expected;
  std::function<ZkxOp(ZkxOp)> unary_op;
};

struct BinaryOpTestCase {
  std::string lhs;
  std::string rhs;
  absl::Span<const int64_t> broadcast_dimensions;
  std::string expected;
  std::function<ZkxOp(ZkxOp, ZkxOp, absl::Span<const int64_t>)> binary_op;
  std::optional<std::string_view> error_message;
};

constexpr std::string_view kBroadcastDimensionMismatch =
    "Broadcast dimension 0 mismatch: 2 != -9223372036854775808; u32[2] and "
    "u32[?,10].";
std::array<const int64_t, 0> empty_array = {};
std::array<const int64_t, 1> zero_array = {0};

class ZkxBuilderUnboundedUnaryOpTest
    : public ::testing::TestWithParam<UnaryOpTestCase> {};

class ZkxBuilderUnboundedBinaryOpTest
    : public ::testing::TestWithParam<BinaryOpTestCase> {};

TEST_P(ZkxBuilderUnboundedUnaryOpTest, UnboundedUnaryOpTest) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape(GetParam().operand));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape(GetParam().expected));
  GetParam().unary_op(Parameter(&b, 0, operand, "operand"));
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST_P(ZkxBuilderUnboundedBinaryOpTest, UnboundedBinaryOpTest) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape(GetParam().expected));
  GetParam().binary_op(Parameter(&b, 0, lhs, "lhs"),
                       Parameter(&b, 1, rhs, "rhs"),
                       GetParam().broadcast_dimensions);
  if (const auto result = BuildHloModule(b); result.ok()) {
    ASSERT_NE(*result, nullptr);
    EXPECT_THAT(GetRoot(**result),
                GmockMatch(m::Op().WithShapeEqualTo(&expected)));
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(result, StatusIs(_, HasSubstr(*GetParam().error_message)));
  }
}

TEST(ZkxBuilderTest, UnboundedAddScalarBroadcast) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("u32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  Add(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
      /*broadcast_dimensions=*/empty_array);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedAddDegenerateBroadcast) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("u32[1, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  Add(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
      /*broadcast_dimensions=*/{0, 1});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedAddUnsupportedImplicitBroadcast) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("u32[2]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  Add(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
      /*broadcast_dimensions=*/zero_array);
  EXPECT_THAT(BuildHloModule(b),
              StatusIs(_, HasSubstr(kBroadcastDimensionMismatch)));
}

// TODO(chokobole): Add test. Dependency: ZkxBuilder::AllGather
// TEST(ZkxBuilderTest, UnboundedAllGather) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::AllReduce
// TEST(ZkxBuilderTest, UnboundedAllReduce) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::AllToAll
// TEST(ZkxBuilderTest, UnboundedAllToAllDynamicSplitDimension) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::AllToAll
// TEST(ZkxBuilderTest, UnboundedAllToAllDynamicConcatDimension) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::AllToAll
// TEST(ZkxBuilderTest, UnboundedAllToAllDynamicSplitAndConcatDimensionEqual) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::AllToAll
// TEST(ZkxBuilderTest, UnboundedAllToAllFullyDynamic) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::AllToAllTuple
// TEST(ZkxBuilderTest, UnboundedAllToAllTupleVariadicUnsupported) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::AllToAllTuple
// TEST(ZkxBuilderTest, UnboundedAllToAllTupleUnsupported) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::AllToAllTuple
// TEST(ZkxBuilderTest, BoundedAllToAllTupleUnsupported) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::AllToAllTuple
// TEST(ZkxBuilderTest, BoundedAllToAllUnsupported) {

TEST(ZkxBuilderTest, UnboundedAnd) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs,
                          ParseShape("s32[1, ?, 2, ?, <=2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs,
                          ParseShape("s32[?, 1, ?, 2, ?, <=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("s32[?, ?, 2, 2, <=2, <=2, ?]"));
  And(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
      /*broadcast_dimensions=*/empty_array);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedBitcastConvert) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u16[?, 10, 2]"));
  BitcastConvertType(Parameter(&b, 0, operand, "operand"), PrimitiveType::U16);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedBroadcastUnsupportedOperand) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[<=3, ?]"));
  Broadcast(Parameter(&b, 0, operand, "operand"), /*broadcast_sizes=*/{1});
  EXPECT_THAT(BuildHloModule(b),
              StatusIs(_, HasSubstr("is_unbounded_dynamic")));
}

TEST(ZkxBuilderTest, UnboundedBroadcastUnsupportedBroadcastSize) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[1]"));
  Broadcast(Parameter(&b, 0, operand, "operand"),
            /*broadcast_sizes=*/{Shape::kUnboundedSize});
  EXPECT_THAT(
      BuildHloModule(b),
      StatusIs(_, HasSubstr("Non-broadcast dimensions must not be dynamic.")));
}

TEST(ZkxBuilderTest, UnboundedBroadcastInDim) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[<=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[<=2, 3, 4]"));
  BroadcastInDim(Parameter(&b, 0, operand, "operand"),
                 /*out_dim_size=*/{2, 3, 4},
                 /*broadcast_dimensions=*/{0, 2});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedBroadcastInDimUnsupported) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[<=3, ?]"));
  BroadcastInDim(Parameter(&b, 0, operand, "operand"),
                 /*out_dim_size=*/{2, 3, Shape::kUnboundedSize},
                 /*broadcast_dimensions=*/{0, 2});
  EXPECT_THAT(BuildHloModule(b),
              StatusIs(_, HasSubstr("BroadcastInDim output must shape be "
                                    "static or bounded dynamic")));
}

TEST(ZkxBuilderTest, UnboundedCall) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));

  ZkxComputation computation;
  {
    const std::unique_ptr<ZkxBuilder> sub_builder = b.CreateSubBuilder("add");
    Add(Parameter(sub_builder.get(), 0, operand, "arg0"),
        Parameter(sub_builder.get(), 1, operand, "arg1"));
    TF_ASSERT_OK_AND_ASSIGN(computation, sub_builder->Build());
  }

  Call(/*builder=*/&b, /*computation=*/computation, /*operands=*/
       {Parameter(&b, 0, operand, "arg0"), Parameter(&b, 1, operand, "arg1")});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedClamp) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs,
                          ParseShape("u32[1, ?, 2, ?, <=2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs,
                          ParseShape("u32[?, 1, ?, 2, ?, <=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs,
                          ParseShape("u32[1, ?, 2, ?, <=2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("u32[?, 1, ?, 2, ?, <=2, ?]"));
  Clamp(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
        Parameter(&b, 2, ehs, "ehs"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedClampScalarMinImplicitBroadcast) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("u32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  Clamp(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
        Parameter(&b, 2, ehs, "ehs"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedClampScalarMinMaxImplicitBroadcast) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("u32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("u32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  Clamp(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
        Parameter(&b, 2, ehs, "ehs"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedClampScalarOperandMaxImplicitBroadcast) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("u32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("u32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  Clamp(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
        Parameter(&b, 2, ehs, "ehs"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedClampScalarMinOperandImplicitBroadcast) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("u32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("u32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  Clamp(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
        Parameter(&b, 2, ehs, "ehs"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest,
     UnboundedClampUnsupportedDegenerateOperandImplicitBroadcast) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("u32[1]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("u32[?, 10]"));
  Clamp(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
        Parameter(&b, 2, ehs, "ehs"));
  EXPECT_THAT(BuildHloModule(b),
              StatusIs(_, HasSubstr("Unimplemented implicit broadcast.")));
}

// TODO(chokobole): Add test. Dependency: ZkxBuilder::CollectiveBroadcast
// TEST(ZkxBuilderTest, UnboundedCollectiveBroadcast) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::CollectivePermute
// TEST(ZkxBuilderTest, UnboundedCollectivePermute) {

TEST(ZkxBuilderTest, UnboundedCompare) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs,
                          ParseShape("u32[1, ?, 2, ?, <=2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs,
                          ParseShape("u32[?, 1, ?, 2, ?, <=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("pred[?, ?, 2, 2, <=2, <=2, ?]"));
  Compare(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
          ComparisonDirection::kEq);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedConcatenate) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand1,
                          ParseShape("u32[3, ?, 2, ?, <=2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand2,
                          ParseShape("u32[?, 4, ?, 2, ?, <=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand3,
                          ParseShape("u32[?, ?, 2, 2, <=2, <=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("u32[3, 4, ?, 2, <=2, <=2, ?]"));
  ConcatInDim(&b,
              {Parameter(&b, 0, operand1, "operand1"),
               Parameter(&b, 1, operand2, "operand2"),
               Parameter(&b, 2, operand3, "operand3")},
              2);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedConvert) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("s32[?]"));
  ConvertElementType(Parameter(&b, 0, operand, "operand"), PrimitiveType::S32);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

// TODO(chokobole): Add test. Dependency: ZkxBuilder::Dot
// TEST(ZkxBuilderTest, UnboundedDot) {

TEST(ZkxBuilderTest, UnboundedDotGeneral) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("u32[?, <=3, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("u32[2, 4, 5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, <=3, 5]"));

  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(2);
  dnums.add_rhs_contracting_dimensions(1);
  dnums.add_lhs_batch_dimensions(0);
  dnums.add_rhs_batch_dimensions(0);

  DotGeneral(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"), dnums);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedDynamicSlice) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape start_indices, ParseShape("s32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[2, 2]"));
  DynamicSlice(Parameter(&b, 0, operand, "operand"),
               /*start_indices=*/
               {
                   Parameter(&b, 1, start_indices, "start_indices0"),
                   Parameter(&b, 2, start_indices, "start_indices1"),
               },
               /*slice_sizes=*/{2, 2});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedDynamicUpdateSlice) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape update, ParseShape("u32[?, 5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape start_indices, ParseShape("s32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  DynamicUpdateSlice(Parameter(&b, 0, operand, "operand"),
                     Parameter(&b, 1, update, "update"),
                     /*start_indices=*/
                     {Parameter(&b, 2, start_indices, "start_indices0"),
                      Parameter(&b, 3, start_indices, "start_indices1")});
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedGather) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[3, 4, 2]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape start_indices,
                          ParseShape("s32[?, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, ?, 2, 2]"));

  GatherDimensionNumbers dimension_numbers;
  dimension_numbers.add_offset_dims(2);
  dimension_numbers.add_offset_dims(3);
  dimension_numbers.add_collapsed_slice_dims(0);
  dimension_numbers.add_start_index_map(1);
  dimension_numbers.add_start_index_map(0);
  dimension_numbers.set_index_vector_dim(2);

  Gather(Parameter(&b, 0, operand, "operand"),
         Parameter(&b, 1, start_indices, "start_indices"), dimension_numbers,
         /*slice_sizes=*/{1, 2, 2});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedGetTupleElement) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  GetTupleElement(Tuple(&b, {Parameter(&b, 0, operand, "operand")}), 0);
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

// TODO(chokobole): Add test. Dependency: ZkxBuilder::Infeed
// TEST(ZkxBuilderTest, UnboundedInfeed) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::InfeedWithToken
// TEST(ZkxBuilderTest, UnboundedInfeedWithToken) {

TEST(ZkxBuilderTest, UnboundedMap) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand0, ParseShape("u32[2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand1, ParseShape("u32[?, 3, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[2, ?, ?]"));

  ZkxComputation computation;
  {
    const std::unique_ptr<ZkxBuilder> sub_builder = b.CreateSubBuilder("add");
    Add(Parameter(sub_builder.get(), 0, ShapeUtil::MakeScalarShape(U32),
                  "arg0"),
        Parameter(sub_builder.get(), 1, ShapeUtil::MakeScalarShape(U32),
                  "arg1"));
    TF_ASSERT_OK_AND_ASSIGN(computation, sub_builder->Build());
  }

  Map(&b, /*operands=*/
      {Parameter(&b, 0, operand0, "operand0"),
       Parameter(&b, 1, operand1, "operand1")},
      computation, /*dimensions=*/{0, 1, 2},
      /*static_operands=*/{});

  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

// TODO(chokobole): Add test. Dependency: ZkxBuilder::OptimizationBarrier
// TEST(ZkxBuilderTest, UnboundedOptimizationBarrier) {

TEST(ZkxBuilderTest, UnboundedOr) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs,
                          ParseShape("s32[1, ?, 2, ?, <=2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs,
                          ParseShape("s32[?, 1, ?, 2, ?, <=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("s32[?, ?, 2, 2, <=2, <=2, ?]"));
  Or(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
     /*broadcast_dimensions=*/empty_array);
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

// TODO(chokobole): Add test. Dependency: ZkxBuilder::Outfeed
// TEST(ZkxBuilderTest, UnboundedOutfeed) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::OutfeedWithToken
// TEST(ZkxBuilderTest, UnboundedOutfeedWithToken) {

TEST(ZkxBuilderTest, UnboundedPad) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 12]"));
  PaddingConfig padding_config;
  for (int i = 0; i < 2; i++) {
    auto dimension = padding_config.add_dimensions();
    dimension->set_edge_padding_low(1);
    dimension->set_edge_padding_high(1);
  }
  Pad(Parameter(&b, 0, operand, "operand"),
      /*padding_value=*/ConstantR0<uint32_t>(&b, 0), padding_config);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

// TODO(chokobole): Add test. Dependency: ZkxBuilder::Recv
// TEST(ZkxBuilderTest, UnboundedRecv) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::RecvFromHost
// TEST(ZkxBuilderTest, UnboundedRecvFromHost) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::RecvWithToken
// TEST(ZkxBuilderTest, UnboundedRecvWithToken) {

TEST(ZkxBuilderTest, UnboundedReduce) {
  ZkxBuilder b(TestName());
  const Shape shape = ShapeUtil::MakeShape(U32, {7}, {false});
  const Shape expected = ShapeUtil::MakeTupleShape({shape, shape, shape});

  TF_ASSERT_OK_AND_ASSIGN(const Shape input0, ParseShape("u32[7, 5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape input1, ParseShape("u32[?, 5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape input2, ParseShape("u32[7, ?]"));
  const Shape scalar_u32 = ShapeUtil::MakeShape(U32, {});
  const ZkxOp init = Parameter(&b, 3, scalar_u32, "init");

  ZkxBuilder bsum(TestName());
  std::vector<ZkxOp> output_operands = {
      Add(Parameter(&bsum, 0, scalar_u32, "arg0"),
          Parameter(&bsum, 1, scalar_u32, "arg1")),
      Add(Parameter(&bsum, 2, scalar_u32, "arg2"),
          Parameter(&bsum, 3, scalar_u32, "arg3")),
      Add(Parameter(&bsum, 4, scalar_u32, "arg4"),
          Parameter(&bsum, 5, scalar_u32, "arg5"))};
  Tuple(&bsum, absl::MakeSpan(output_operands));
  TF_ASSERT_OK_AND_ASSIGN(const ZkxComputation sum, bsum.Build());
  Reduce(
      &b,
      {Parameter(&b, 0, input0, "input0"), Parameter(&b, 1, input1, "input1"),
       Parameter(&b, 2, input2, "input2")},
      {init, init, init}, sum, {1});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

// TODO(chokobole): Add test. Dependency: ZkxBuilder::ReduceScatter
// TEST(ZkxBuilderTest, UnboundedReduceScatter) {

TEST(ZkxBuilderTest, UnboundedReduceWindow) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape input, ParseShape("u32[?, 4, 8]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 3, 5]"));

  ZkxBuilder bsum(TestName());
  Add(Parameter(&bsum, 0, ShapeUtil::MakeShape(U32, {}), "x"),
      Parameter(&bsum, 1, ShapeUtil::MakeShape(U32, {}), "y"));
  TF_ASSERT_OK_AND_ASSIGN(const ZkxComputation sum, bsum.Build());

  ReduceWindow(Parameter(&b, 0, input, "input"), ConstantR0<uint32_t>(&b, 0),
               sum,
               window_util::MakeWindow(/*sizes=*/{1, 2, 4},
                                       /*strides=*/{1, 1, 1}));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedReshape) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[2,3]"));
  Reshape(Parameter(&b, 0, operand, "operand"), /*dimensions=*/{0},
          /*new_sizes=*/{2, 3});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedReshapeUnsupportedOutputShape) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[6]"));
  Reshape(Parameter(&b, 0, operand, "operand"), /*dimensions=*/{0},
          /*new_sizes=*/{Shape::kUnboundedSize, Shape::kUnboundedSize});
  EXPECT_THAT(
      BuildHloModule(b),
      StatusIs(_,
               HasSubstr(
                   "Reshaping with unbounded result shape is not supported.")));
}

TEST(ZkxBuilderTest, UnboundedReshapeUnsupportedInferredShape) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?]"));
  Reshape(operand, Parameter(&b, 0, operand, "operand"));
  EXPECT_THAT(
      BuildHloModule(b),
      StatusIs(_,
               HasSubstr(
                   "Reshaping with unbounded result shape is not supported.")));
}

TEST(ZkxBuilderTest, UnboundedReverse) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  Rev(Parameter(&b, 0, operand, "operand"), /*dimensions=*/{0, 1});
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

// TODO(chokobole): Add test. Dependency: ZkxBuilder::RngBitGenerator
// TEST(ZkxBuilderTest, UnboundedRngBitGenerator) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::RngNormal
// TEST(ZkxBuilderTest, UnboundedRngNormal) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::RngUniform
// TEST(ZkxBuilderTest, UnboundedRngUniform) {

TEST(ZkxBuilderTest, UnboundedScatter) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape input, ParseShape("u32[?, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape scatter_indices,
                          ParseShape("s32[?, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape updates, ParseShape("u32[?, ?, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, ?, ?]"));

  ZkxComputation update_computation;
  {
    const std::unique_ptr<ZkxBuilder> sub_builder = b.CreateSubBuilder("add");
    Add(Parameter(sub_builder.get(), 0, ShapeUtil::MakeScalarShape(U32),
                  "arg0"),
        Parameter(sub_builder.get(), 1, ShapeUtil::MakeScalarShape(U32),
                  "arg1"));
    TF_ASSERT_OK_AND_ASSIGN(update_computation, sub_builder->Build());
  }

  ScatterDimensionNumbers dimension_numbers;
  dimension_numbers.add_update_window_dims(2);
  dimension_numbers.add_update_window_dims(3);
  dimension_numbers.add_inserted_window_dims(0);
  dimension_numbers.add_scatter_dims_to_operand_dims(1);
  dimension_numbers.add_scatter_dims_to_operand_dims(0);
  dimension_numbers.set_index_vector_dim(2);

  Scatter(Parameter(&b, 0, input, "input"),
          Parameter(&b, 1, scatter_indices, "scatter_indices"),
          Parameter(&b, 2, updates, "updates"), update_computation,
          dimension_numbers, /*indices_are_sorted=*/false,
          /*unique_indices=*/false);

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedSelect) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs,
                          ParseShape("pred[1, ?, 2, ?, <=2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs,
                          ParseShape("u32[?, 1, ?, 2, ?, <=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs,
                          ParseShape("u32[1, ?, 2, ?, <=2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("u32[1, 1, 2, 2, <=2, <=2, ?]"));
  Select(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
         Parameter(&b, 2, ehs, "ehs"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedSelectScalarPred) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("pred[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  Select(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
         Parameter(&b, 2, ehs, "ehs"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedSelectScalarOnTrueOnFalseImplicitBroadcast) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("pred[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("u32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("u32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  Select(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
         Parameter(&b, 2, ehs, "ehs"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedSelectScalarPredOnFalseImplicitBroadcast) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("pred[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("u32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  Select(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
         Parameter(&b, 2, ehs, "ehs"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedSelectScalarPredOnTrueImplicitBroadcast) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("pred[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("u32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  Select(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
         Parameter(&b, 2, ehs, "ehs"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest,
     UnboundedSelectUnsupportedDegenerateOperandImplicitBroadcast) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("pred[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("u32[1]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("u32[?, 10]"));
  Select(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
         Parameter(&b, 2, ehs, "ehs"));
  EXPECT_THAT(BuildHloModule(b),
              StatusIs(_, HasSubstr("Unimplemented implicit broadcast.")));
}

// TODO(batzor): Add test. Dependency: ZkxBuilder::SelectAndScatter
// TEST(ZkxBuilderTest, UnboundedSelectAndScatter) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::Send
// TEST(ZkxBuilderTest, UnboundedSend) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::SendToHost
// TEST(ZkxBuilderTest, UnboundedSendToHost) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::SendWithToken
// TEST(ZkxBuilderTest, UnboundedSendWithToken) {

TEST(ZkxBuilderTest, UnboundedSlice) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[1, <=3, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[1, <=2, 3]"));
  Slice(Parameter(&b, 0, operand, "operand"),
        /*start_indices=*/{0, 1, 2},
        /*limit_indices=*/{1, 3, 5},
        /*strides=*/{1, 1, 1});
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedSort) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));

  ZkxComputation comparator;
  {
    const std::unique_ptr<ZkxBuilder> sub_builder =
        b.CreateSubBuilder("compare");
    Compare(Parameter(sub_builder.get(), 0, ShapeUtil::MakeScalarShape(U32),
                      "arg0"),
            Parameter(sub_builder.get(), 1, ShapeUtil::MakeScalarShape(U32),
                      "arg1"),
            ComparisonDirection::kLt);
    TF_ASSERT_OK_AND_ASSIGN(comparator, sub_builder->Build());
  }

  Sort({Parameter(&b, 0, operand, "operand")}, comparator,
       /*dimension=*/0, /*is_stable=*/true);
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedTranspose) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand,
                          ParseShape("u32[1, ?, 2, ?, <=2]{4,3,2,1,0}"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("u32[<=2, 1, ?, 2, ?]{0,2,3,4,1}"));
  Transpose(Parameter(&b, 0, operand, "operand"),
            /*permutation=*/{4, 0, 3, 2, 1});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedTuple) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  const Shape expected = ShapeUtil::MakeTupleShape({operand});
  Tuple(&b, {Parameter(&b, 0, operand, "operand")});
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedWhile) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape init, ParseShape("u32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?]"));

  ZkxComputation add;
  {
    const std::unique_ptr<ZkxBuilder> sub_builder = b.CreateSubBuilder("add");
    Add(Parameter(sub_builder.get(), 0, ShapeUtil::MakeScalarShape(U32),
                  "arg0"),
        Parameter(sub_builder.get(), 1, ShapeUtil::MakeScalarShape(U32),
                  "arg1"));
    TF_ASSERT_OK_AND_ASSIGN(add, sub_builder->Build());
  }

  ZkxComputation condition;
  {
    const std::unique_ptr<ZkxBuilder> sub_builder =
        b.CreateSubBuilder("compare");
    Ge(/*lhs=*/ConstantR0<uint32_t>(sub_builder.get(), 10),
       /*rhs=*/Reduce(/*operand=*/Parameter(sub_builder.get(), 0, init, "prev"),
                      ConstantR0<uint32_t>(sub_builder.get(), 0), add,
                      /*dimensions_to_reduce=*/{0}));
    TF_ASSERT_OK_AND_ASSIGN(condition, sub_builder->Build());
  }

  ZkxComputation body;
  {
    const std::unique_ptr<ZkxBuilder> sub_builder = b.CreateSubBuilder("add");
    Add(ConstantR1<uint32_t>(sub_builder.get(), {1}),
        Parameter(sub_builder.get(), 0, init, "prev"),
        /*broadcast_dimensions=*/{0});
    TF_ASSERT_OK_AND_ASSIGN(body, sub_builder->Build());
  }

  While(condition, body, Parameter(&b, 0, init, "init"));
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(ZkxBuilderTest, UnboundedXor) {
  ZkxBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs,
                          ParseShape("s32[1, ?, 2, ?, <=2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs,
                          ParseShape("s32[?, 1, ?, 2, ?, <=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("s32[?, ?, 2, 2, <=2, <=2, ?]"));
  Xor(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
      /*broadcast_dimensions=*/empty_array);
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

INSTANTIATE_TEST_SUITE_P(UnboundedDynamism, ZkxBuilderUnboundedUnaryOpTest,
                         ::testing::ValuesIn<UnaryOpTestCase>(
                             {{"s32[?]", "s32[?]", &Abs},
                              {"u32[?]", "u32[?]", &Clz},
                              {"u32[?]", "u32[?]", &Neg},
                              {"s32[?]", "s32[?]", &Not},
                              {"u32[?]", "u32[?]", &PopulationCount},
                              {"s32[?]", "s32[?]", &Sign}}));
INSTANTIATE_TEST_SUITE_P(
    UnboundedDynamism, ZkxBuilderUnboundedBinaryOpTest,
    ::testing::ValuesIn<BinaryOpTestCase>({
        {"u32[1, ?, 2, ?, <=2, ?, ?]", "u32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "u32[?, ?, 2, 2, <=2, <=2, ?]",
         &Add},
        {"u32[?, 10]", "u32[1]", /*broadcast_dimensions=*/zero_array,
         "u32[?, 10]", &Add},
        {"u32[1, ?, 2, ?, <=2, ?, ?]", "u32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "u32[?, ?, 2, 2, <=2, <=2, ?]",
         &Div},
        {"u32[?, 10]", "u32[1]", /*broadcast_dimensions=*/zero_array,
         "u32[?, 10]", &Div},
        {"u32[1, ?, 2, ?, <=2, ?, ?]", "u32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "u32[?, ?, 2, 2, <=2, <=2, ?]",
         &Max},
        {"u32[?, 10]", "u32[1]", /*broadcast_dimensions=*/zero_array,
         "u32[?, 10]", &Max},
        {"u32[1, ?, 2, ?, <=2, ?, ?]", "u32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "u32[?, ?, 2, 2, <=2, <=2, ?]",
         &Min},
        {"u32[?, 10]", "u32[1]", /*broadcast_dimensions=*/zero_array,
         "u32[?, 10]", &Min},
        {"u32[1, ?, 2, ?, <=2, ?, ?]", "u32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "u32[?, ?, 2, 2, <=2, <=2, ?]",
         &Mul},
        {"u32[?, 10]", "u32[1]", /*broadcast_dimensions=*/zero_array,
         "u32[?, 10]", &Mul},
        {"u32[?, 10]", "u32[1]", /*broadcast_dimensions=*/zero_array,
         "pred[?, 10]", &Ne},
        {"u32[1, ?, 2, ?, <=2, ?, ?]", "u32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "u32[?, ?, 2, 2, <=2, <=2, ?]",
         &Pow},
        {"u32[?, 10]", "u32[1]", /*broadcast_dimensions=*/zero_array,
         "u32[?, 10]", &Pow},
        {"u32[1, ?, 2, ?, <=2, ?, ?]", "u32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "u32[?, ?, 2, 2, <=2, <=2, ?]",
         &Rem},
        {"u32[?, 10]", "u32[1]", /*broadcast_dimensions=*/zero_array,
         "u32[?, 10]", &Rem},
        {"u32[1, ?, 2, ?, <=2, ?, ?]", "u32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "u32[?, ?, 2, 2, <=2, <=2, ?]",
         &ShiftLeft},
        {"u32[?, 10]", "u32[1]", /*broadcast_dimensions=*/zero_array,
         "u32[?, 10]", &ShiftLeft},
        {"u32[1, ?, 2, ?, <=2, ?, ?]", "u32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "u32[?, ?, 2, 2, <=2, <=2, ?]",
         &ShiftRightArithmetic},
        {"u32[?, 10]", "u32[1]", /*broadcast_dimensions=*/zero_array,
         "u32[?, 10]", &ShiftRightArithmetic},
        {"u32[1, ?, 2, ?, <=2, ?, ?]", "u32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "u32[?, ?, 2, 2, <=2, <=2, ?]",
         &ShiftRightLogical},
        {"u32[?, 10]", "u32[1]", /*broadcast_dimensions=*/zero_array,
         "u32[?, 10]", &ShiftRightLogical},
        {"u32[1, ?, 2, ?, <=2, ?, ?]", "u32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "u32[?, ?, 2, 2, <=2, <=2, ?]",
         &Sub},
        {"u32[?, 10]", "u32[1]", /*broadcast_dimensions=*/zero_array,
         "u32[?, 10]", &Sub},
    }));

// TODO(chokobole): Add test. Dependency: ZkxBuilder::Infeed
// TEST(ZkxBuilderTest, UnorderedInfeed) {

// TODO(chokobole): Add test. Dependency: ZkxBuilder::Outfeed
// TEST(ZkxBuilderTest, UnorderedOutfeed) {

}  // namespace
}  // namespace zkx
