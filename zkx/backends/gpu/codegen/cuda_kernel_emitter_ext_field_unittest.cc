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

#include "zk_dtypes/include/all_types.h"
#include "zkx/backends/gpu/codegen/field_test.h"

namespace zkx::gpu {

using ExtensionFieldTypes = testing::Types<
    // clang-format off
    zk_dtypes::BabybearX4Mont,
    zk_dtypes::GoldilocksX3Mont,
    zk_dtypes::KoalabearX4Mont,
    zk_dtypes::Mersenne31X2
    // clang-format on
    >;

TYPED_TEST_SUITE(FieldScalarUnaryTest, ExtensionFieldTypes);

TYPED_TEST(FieldScalarUnaryTest, Negate) {
  this->SetUpNegate();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(FieldScalarBinaryTest, ExtensionFieldTypes);

TYPED_TEST(FieldScalarBinaryTest, Add) {
  this->SetUpAdd();
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

TYPED_TEST(FieldScalarBinaryTest, Square) {
  this->SetUpSquare();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, Sub) {
  this->SetUpSub();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(FieldScalarTernaryTest, ExtensionFieldTypes);

TYPED_TEST(FieldScalarTernaryTest, SelectTrue) {
  this->SetUpSelectTrue();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarTernaryTest, SelectFalse) {
  this->SetUpSelectFalse();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(FieldR2TensorBinaryTest, ExtensionFieldTypes);

TYPED_TEST(FieldR2TensorBinaryTest, Add) {
  this->SetUpAdd();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(ExtFieldBitcastConvertTest, ExtensionFieldTypes);

TYPED_TEST(ExtFieldBitcastConvertTest, ExtFieldToField) {
  // BitcastDtypesExpander needs a matching unsigned integral type for the
  // EF storage width. Skip non-power-of-2 sizes (e.g., GoldilocksX3Mont
  // is 24 bytes / 192 bits — no U192 in HLO).
  if constexpr ((sizeof(TypeParam) & (sizeof(TypeParam) - 1)) != 0) {
    GTEST_SKIP() << "No unsigned integral type for " << sizeof(TypeParam) * 8
                 << "-bit EF";
  } else {
    this->SetUpExtFieldToField();
    this->RunAndVerify(/*run_hlo_passes=*/true);
  }
}

TYPED_TEST(ExtFieldBitcastConvertTest, FieldToExtField) {
  if constexpr ((sizeof(TypeParam) & (sizeof(TypeParam) - 1)) != 0) {
    GTEST_SKIP() << "No unsigned integral type for " << sizeof(TypeParam) * 8
                 << "-bit EF";
  } else {
    this->SetUpFieldToExtField();
    this->RunAndVerify(/*run_hlo_passes=*/true);
  }
}

}  // namespace zkx::gpu
