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

#include "zkx/backends/cpu/codegen/field_test.h"

namespace zkx::cpu {

using FieldTypes = testing::Types<
    // clang-format off
    zk_dtypes::BabybearMont,
    zk_dtypes::GoldilocksMont,
    zk_dtypes::bn254::FrMont
    // clang-format on
    >;
TYPED_TEST_SUITE(FieldScalarUnaryTest, FieldTypes);

TYPED_TEST(FieldScalarUnaryTest, ConvertFromInt) {
  this->SetUpConvertFromInt();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarUnaryTest, ConvertFromIntToStd) {
  this->SetUpConvertFromIntToStd();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarUnaryTest, ConvertToStd) {
  this->SetUpConvertToStd();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarUnaryTest, Negate) {
  this->SetUpNegate();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarUnaryTest, Inverse) {
  this->SetUpInverse();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(FieldScalarBinaryTest, FieldTypes);

TYPED_TEST(FieldScalarBinaryTest, Add) {
  this->SetUpAdd();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, Compare) {
  this->SetUpCompare();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, Div) {
  this->SetUpDiv();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, Double) {
  this->SetUpDouble();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, Fusion) {
  this->SetUpFusion();
  this->RunAndVerify(true);
}

TYPED_TEST(FieldScalarBinaryTest, Maximum) {
  this->SetUpMaximum();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, Minimum) {
  this->SetUpMinimum();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, Mul) {
  this->SetUpMul();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, Pow) {
  this->SetUpPow();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, PowWithSignedExponentShouldFail) {
  this->SetUpPowWithSignedExponentShouldFail();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, Square) {
  this->SetUpSquare();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, Sub) {
  this->SetUpSub();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(FieldScalarTernaryTest, FieldTypes);

TYPED_TEST(FieldScalarTernaryTest, Clamp) {
  this->SetUpClamp();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarTernaryTest, Select) {
  this->SetUpSelect();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(FieldR1TensorUnaryTest, FieldTypes);

TYPED_TEST(FieldR1TensorUnaryTest, BatchInverse) {
  this->SetUpBatchInverse();
  this->RunAndVerify();
}

TYPED_TEST(FieldR1TensorUnaryTest, FFT) {
  this->SetUpFFT();
  this->RunAndVerify();
}

TYPED_TEST(FieldR1TensorUnaryTest, FFTWithTwiddles) {
  this->SetUpFFTWithTwiddles();
  this->RunAndVerify();
}

TYPED_TEST(FieldR1TensorUnaryTest, IFFT) {
  this->SetUpIFFT();
  this->RunAndVerify();
}

TYPED_TEST(FieldR1TensorUnaryTest, IFFTWithTwiddles) {
  this->SetUpIFFTWithTwiddles();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(FieldR2TensorBinaryTest, FieldTypes);

TYPED_TEST(FieldR2TensorBinaryTest, Add) { this->SetUpAdd(); }

TYPED_TEST_SUITE(FieldTest, FieldTypes);

TYPED_TEST(FieldTest, DynamicUpdateSlice) {
  this->SetUpDynamicUpdateSliceBug();
  this->RunAndVerify();
}

TYPED_TEST(FieldTest, MatrixMatrixMultiplication) {
  this->SetUpDenseMatrixMatrixMultiplication();
  this->RunAndVerify();
}

TYPED_TEST(FieldTest, MatrixVectorMultiplication) {
  this->SetUpDenseMatrixVectorMultiplication();
  this->RunAndVerify();
}

TYPED_TEST(FieldTest, VectorVectorMultiplication) {
  this->SetUpDenseVectorVectorMultiplication();
  this->RunAndVerify();
}

}  // namespace zkx::cpu
