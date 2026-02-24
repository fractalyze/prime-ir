/* Copyright 2025 The ZKX Authors.

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

#include <stdint.h>

#include "zkx/backends/cpu/codegen/int_test.h"
#include "zkx/types.h"

namespace zkx::cpu {

using IntTypes = testing::Types<uint32_t, int32_t, u128>;
TYPED_TEST_SUITE(IntScalarUnaryTest, IntTypes);

TYPED_TEST(IntScalarUnaryTest, Abs) {
  if constexpr (std::is_signed_v<TypeParam>) {
    this->SetUpAbs();
    this->RunAndVerify();
  } else {
    GTEST_SKIP() << "Skipping test for unsigned type";
  }
}

TYPED_TEST(IntScalarUnaryTest, BitcastConvert) {
  if constexpr (is_big_int_v<TypeParam>) {
    GTEST_SKIP() << "BigInt has no signed equivalent for bitcast";
  } else {
    this->SetUpBitcastConvert();
    this->RunAndVerify();
  }
}

TYPED_TEST(IntScalarUnaryTest, CountLeadingZeros) {
  if constexpr (is_big_int_v<TypeParam>) {
    GTEST_SKIP() << "__builtin_clz not applicable to BigInt";
  } else {
    this->SetUpCountLeadingZeros();
    this->RunAndVerify();
  }
}

TYPED_TEST(IntScalarUnaryTest, ConvertUp) {
  if constexpr (is_big_int_v<TypeParam>) {
    GTEST_SKIP() << "Convert not supported for BigInt";
  } else {
    this->SetUpConvertUp();
    this->RunAndVerify();
  }
}

TYPED_TEST(IntScalarUnaryTest, ConvertDown) {
  if constexpr (is_big_int_v<TypeParam>) {
    GTEST_SKIP() << "Convert not supported for BigInt";
  } else {
    this->SetUpConvertDown();
    this->RunAndVerify();
  }
}

TYPED_TEST(IntScalarUnaryTest, Negate) {
  if constexpr (std::is_signed_v<TypeParam>) {
    this->SetUpNegate();
    this->RunAndVerify();
  } else {
    GTEST_SKIP() << "Skipping test for unsigned type";
  }
}

TYPED_TEST(IntScalarUnaryTest, Not) {
  if constexpr (is_big_int_v<TypeParam>) {
    GTEST_SKIP() << "Bitwise NOT not supported for BigInt in test setup";
  } else {
    this->SetUpNot();
    this->RunAndVerify();
  }
}

TYPED_TEST(IntScalarUnaryTest, PopulationCount) {
  if constexpr (is_big_int_v<TypeParam>) {
    GTEST_SKIP() << "__builtin_popcount not applicable to BigInt";
  } else {
    this->SetUpPopulationCount();
    this->RunAndVerify();
  }
}

TYPED_TEST(IntScalarUnaryTest, Sign) {
  if constexpr (std::is_signed_v<TypeParam>) {
    this->SetUpSign();
    this->RunAndVerify();
  } else {
    GTEST_SKIP() << "Skipping test for unsigned type";
  }
}

TYPED_TEST_SUITE(IntScalarBinaryTest, IntTypes);

TYPED_TEST(IntScalarBinaryTest, Add) {
  this->SetUpAdd();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarBinaryTest, And) {
  if constexpr (is_big_int_v<TypeParam>) {
    GTEST_SKIP() << "Bitwise AND not supported for BigInt in test setup";
  } else {
    this->SetUpAnd();
    this->RunAndVerify();
  }
}

TYPED_TEST(IntScalarBinaryTest, Compare) {
  if constexpr (is_big_int_v<TypeParam>) {
    GTEST_SKIP() << "Comparison direction not supported for BigInt";
  } else {
    this->SetUpCompare();
    this->RunAndVerify();
  }
}

TYPED_TEST(IntScalarBinaryTest, Div) {
  if constexpr (is_big_int_v<TypeParam>) {
    GTEST_SKIP()
        << "Div test uses std::numeric_limits not available for BigInt";
  } else {
    this->SetUpDiv();
    this->RunAndVerify();
  }
}

TYPED_TEST(IntScalarBinaryTest, Fusion) {
  if constexpr (is_big_int_v<TypeParam>) {
    GTEST_SKIP() << "HLO text parser doesn't support BigInt types";
  } else {
    this->SetUpFusion();
    this->RunAndVerify(true);
  }
}

TYPED_TEST(IntScalarBinaryTest, Maximum) {
  this->SetUpMaximum();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarBinaryTest, Minimum) {
  this->SetUpMinimum();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarBinaryTest, Mul) {
  this->SetUpMul();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarBinaryTest, ShiftLeft) {
  if constexpr (is_big_int_v<TypeParam>) {
    GTEST_SKIP()
        << "Shift test uses std::numeric_limits not available for BigInt";
  } else {
    this->SetUpShiftLeft();
    this->RunAndVerify();
  }
}

TYPED_TEST(IntScalarBinaryTest, ShiftRightArithmetic) {
  if constexpr (is_big_int_v<TypeParam>) {
    GTEST_SKIP()
        << "Shift test uses std::make_signed_t not available for BigInt";
  } else {
    this->SetUpShiftRightArithmetic();
    this->RunAndVerify();
  }
}

TYPED_TEST(IntScalarBinaryTest, ShiftRightLogical) {
  if constexpr (is_big_int_v<TypeParam>) {
    GTEST_SKIP()
        << "Shift test uses std::numeric_limits not available for BigInt";
  } else {
    this->SetUpShiftRightLogical();
    this->RunAndVerify();
  }
}

TYPED_TEST(IntScalarBinaryTest, Sub) {
  this->SetUpSub();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarBinaryTest, Or) {
  if constexpr (is_big_int_v<TypeParam>) {
    GTEST_SKIP() << "Bitwise OR not supported for BigInt in test setup";
  } else {
    this->SetUpOr();
    this->RunAndVerify();
  }
}

TYPED_TEST(IntScalarBinaryTest, Power) {
  if constexpr (is_big_int_v<TypeParam>) {
    GTEST_SKIP() << "Power test uses special cases not applicable to BigInt";
  } else {
    this->SetUpPower();
    this->RunAndVerify();
  }
}

TYPED_TEST(IntScalarBinaryTest, Remainder) {
  if constexpr (is_big_int_v<TypeParam>) {
    GTEST_SKIP()
        << "Remainder test uses std::numeric_limits not available for BigInt";
  } else {
    this->SetUpRemainder();
    this->RunAndVerify();
  }
}

TYPED_TEST(IntScalarBinaryTest, Xor) {
  if constexpr (is_big_int_v<TypeParam>) {
    GTEST_SKIP() << "Bitwise XOR not supported for BigInt in test setup";
  } else {
    this->SetUpXor();
    this->RunAndVerify();
  }
}

TYPED_TEST_SUITE(IntScalarTernaryTest, IntTypes);

TYPED_TEST(IntScalarTernaryTest, Clamp) {
  if constexpr (is_big_int_v<TypeParam>) {
    GTEST_SKIP() << "Clamp test uses std::clamp not available for BigInt";
  } else {
    this->SetUpClamp();
    this->RunAndVerify();
  }
}

TYPED_TEST(IntScalarTernaryTest, Select) {
  this->SetUpSelect();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(IntR2TensorBinaryTest, IntTypes);

TYPED_TEST(IntR2TensorBinaryTest, Add) {
  this->SetUpAdd();
  this->RunAndVerify();
}

TYPED_TEST(IntR2TensorBinaryTest, AddWithLayout) {
  this->SetUpAddWithLayout();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(IntTest, IntTypes);

TYPED_TEST(IntTest, BitReverse) {
  this->SetUpBitReverse();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, BroadcastScalar) {
  this->SetUpBroadcastScalar();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, BroadcastTensorR1ToR3WithD0) {
  this->SetUpBroadcastTensorR1ToR3WithD0();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, BroadcastTensorR1ToR3WithD1) {
  this->SetUpBroadcastTensorR1ToR3WithD1();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, BroadcastTensorR1ToR3WithD2) {
  this->SetUpBroadcastTensorR1ToR3WithD2();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, Call) {
  this->SetUpCall();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, Concatenate) {
  this->SetUpConcatenate();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, Conditional) {
  if constexpr (!std::is_signed_v<TypeParam> || is_big_int_v<TypeParam>) {
    GTEST_SKIP() << "Conditional test uses negate, requires signed type";
  } else {
    this->SetUpConditional();
    this->RunAndVerify();
  }
}

TYPED_TEST(IntTest, DynamicSlice) {
  this->SetUpDynamicSlice();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, DynamicUpdateSlice) {
  this->SetUpDynamicUpdateSlice();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, Gather) {
  this->SetUpGather();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, IotaWithD0) {
  this->SetUpIotaWithD0();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, IotaWithD1) {
  this->SetUpIotaWithD1();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, Map) {
  this->SetUpMap();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, Pad) {
  this->SetUpPad();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, Reduce) {
  this->SetUpReduce();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, ReducePartial) {
  this->SetUpReducePartial();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, ReduceWindow) {
  this->SetUpReduceWindow();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, ReshapeScalar) {
  this->SetUpReshapeScalar();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, Reshape) {
  this->SetUpReshape();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, Reverse) {
  this->SetUpReverse();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, ReverseWithEmptyDimensions) {
  this->SetUpReverseWithEmptyDimensions();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, Scatter) {
  this->SetUpScatter();
  this->RunAndVerify(true);
}

TYPED_TEST(IntTest, Slice) {
  this->SetUpSlice();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, Sort) {
  if constexpr (is_big_int_v<TypeParam>) {
    GTEST_SKIP() << "Sort uses comparison not supported for BigInt";
  } else {
    this->SetUpSort();
    this->RunAndVerify();
  }
}

TYPED_TEST(IntTest, Transpose) {
  this->SetUpTranspose();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, TransposeWithLayout) {
  this->SetUpTransposeWithLayout();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, TransposeWithEmptyDimensions) {
  this->SetUpTransposeWithEmptyDimensions();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, While) {
  if constexpr (is_big_int_v<TypeParam>) {
    GTEST_SKIP() << "While test uses mixed-type multiplication";
  } else {
    this->SetUpWhile();
    this->RunAndVerify();
  }
}

}  // namespace zkx::cpu
