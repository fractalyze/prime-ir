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

#include "zkx/backends/gpu/codegen/field_test.h"

namespace zkx::gpu {

using FieldTypes = testing::Types<
    // clang-format off
    zk_dtypes::BabybearMont,
    zk_dtypes::GoldilocksMont,
    zk_dtypes::bn254::FrMont
    // clang-format on
    >;
TYPED_TEST_SUITE(FieldScalarUnaryTest, FieldTypes);

// TODO(chokobole): Enable this test.
TYPED_TEST(FieldScalarUnaryTest, DISABLED_ConvertFromInt) {
  this->SetUpConvertFromInt();
  this->RunAndVerify();
}

// TODO(chokobole): Enable this test.
TYPED_TEST(FieldScalarUnaryTest, DISABLED_ConvertFromIntToStd) {
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

// TODO(chokobole): Add FieldScalarInverse test after creating inverse operation
// in mhlo dialect.

TYPED_TEST_SUITE(FieldScalarBinaryTest, FieldTypes);

TYPED_TEST(FieldScalarBinaryTest, Add) {
  this->SetUpAdd();
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

TYPED_TEST(FieldScalarBinaryTest, Mul) {
  this->SetUpMul();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, Pow) {
  this->SetUpPow();
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

TYPED_TEST_SUITE(FieldR2TensorBinaryTest, FieldTypes);

TYPED_TEST(FieldR2TensorBinaryTest, Add) {
  this->SetUpAdd();
  this->RunAndVerify();
}

// TODO(chokobole): Add FFT, IFFT test
// TODO(chokobole): Add CSRMatrixVectorMultiplication test

}  // namespace zkx::gpu
