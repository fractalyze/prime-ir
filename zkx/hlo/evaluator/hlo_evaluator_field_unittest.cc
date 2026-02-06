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

#include "zkx/hlo/evaluator/hlo_evaluator_field_test.h"

namespace zkx {
namespace {

using FieldTypes = testing::Types<
    // clang-format off
    zk_dtypes::BabybearMont,
    zk_dtypes::GoldilocksMont,
    zk_dtypes::bn254::FrMont
    // clang-format on
    >;

// FieldScalarBinaryTest
TYPED_TEST_SUITE(FieldScalarBinaryTest, FieldTypes);

TYPED_TEST(FieldScalarBinaryTest, Add) {
  this->SetUpAdd();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, Subtract) {
  this->SetUpSubtract();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, Multiply) {
  this->SetUpMultiply();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, Divide) {
  this->SetUpDivide();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, Maximum) {
  this->SetUpMaximum();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, Minimum) {
  this->SetUpMinimum();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, CompareEq) {
  this->SetUpCompareEq();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, CompareNe) {
  this->SetUpCompareNe();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, CompareLt) {
  this->SetUpCompareLt();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, CompareLe) {
  this->SetUpCompareLe();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, CompareGt) {
  this->SetUpCompareGt();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, CompareGe) {
  this->SetUpCompareGe();
  this->RunAndVerify();
}

// FieldScalarTernaryTest
TYPED_TEST_SUITE(FieldScalarTernaryTest, FieldTypes);

TYPED_TEST(FieldScalarTernaryTest, Clamp) {
  this->SetUpClamp();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarTernaryTest, Select) {
  this->SetUpSelect();
  this->RunAndVerify();
}

// FieldR1TensorTest
TYPED_TEST_SUITE(FieldR1TensorTest, FieldTypes);

TYPED_TEST(FieldR1TensorTest, Broadcast) {
  this->SetUpBroadcast();
  this->RunAndVerify();
}

TYPED_TEST(FieldR1TensorTest, Slice) {
  this->SetUpSlice();
  this->RunAndVerify();
}

TYPED_TEST(FieldR1TensorTest, Concatenate) {
  this->SetUpConcatenate();
  this->RunAndVerify();
}

TYPED_TEST(FieldR1TensorTest, Reverse) {
  this->SetUpReverse();
  this->RunAndVerify();
}

TYPED_TEST(FieldR1TensorTest, ReduceSum) {
  this->SetUpReduceSum();
  this->RunAndVerify();
}

TYPED_TEST(FieldR1TensorTest, ReduceProduct) {
  this->SetUpReduceProduct();
  this->RunAndVerify();
}

TYPED_TEST(FieldR1TensorTest, ReduceMax) {
  this->SetUpReduceMax();
  this->RunAndVerify();
}

TYPED_TEST(FieldR1TensorTest, DynamicSlice) {
  this->SetUpDynamicSlice();
  this->RunAndVerify();
}

TYPED_TEST(FieldR1TensorTest, DynamicUpdateSlice) {
  this->SetUpDynamicUpdateSlice();
  this->RunAndVerify();
}

// FieldR2TensorTest
TYPED_TEST_SUITE(FieldR2TensorTest, FieldTypes);

TYPED_TEST(FieldR2TensorTest, Transpose) {
  this->SetUpTranspose();
  this->RunAndVerify();
}

TYPED_TEST(FieldR2TensorTest, Reshape) {
  this->SetUpReshape();
  this->RunAndVerify();
}

// FieldDotTest
TYPED_TEST_SUITE(FieldDotTest, FieldTypes);

TYPED_TEST(FieldDotTest, DotVectorVector) {
  this->SetUpDotVectorVector();
  this->RunAndVerify();
}

TYPED_TEST(FieldDotTest, DotMatrixVector) {
  this->SetUpDotMatrixVector();
  this->RunAndVerify();
}

TYPED_TEST(FieldDotTest, DotMatrixMatrix) {
  this->SetUpDotMatrixMatrix();
  this->RunAndVerify();
}

}  // namespace
}  // namespace zkx
