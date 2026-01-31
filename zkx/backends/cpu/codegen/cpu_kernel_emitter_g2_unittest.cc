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

#include "zkx/backends/cpu/codegen/group_test.h"

namespace zkx::cpu {

using GroupTypes = testing::Types<zk_dtypes::bn254::G2AffinePointMont>;
TYPED_TEST_SUITE(GroupScalarUnaryTest, GroupTypes);

TYPED_TEST(GroupScalarUnaryTest, Convert) {
  this->SetUpConvert();
  this->RunAndVerify();
}

TYPED_TEST(GroupScalarUnaryTest, Negate) {
  this->SetUpNegate();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(GroupScalarBinaryTest, GroupTypes);

TYPED_TEST(GroupScalarBinaryTest, Add) {
  this->SetUpAdd();
  this->RunAndVerify();
}

TYPED_TEST(GroupScalarBinaryTest, Compare) {
  this->SetUpCompare();
  this->RunAndVerify();
}

TYPED_TEST(GroupScalarBinaryTest, Double) {
  this->SetUpDouble();
  this->RunAndVerify();
}

TYPED_TEST(GroupScalarBinaryTest, ScalarMul) {
  this->SetUpScalarMul();
  this->RunAndVerify();
}

TYPED_TEST(GroupScalarBinaryTest, Sub) {
  this->SetUpSub();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(GroupScalarTernaryTest, GroupTypes);

TYPED_TEST(GroupScalarTernaryTest, Select) {
  this->SetUpSelect();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(GroupR2TensorUnaryTest, GroupTypes);

TYPED_TEST(GroupR2TensorUnaryTest, Negate) {
  this->SetUpNegate();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(GroupR2TensorBinaryTest, GroupTypes);

TYPED_TEST(GroupR2TensorBinaryTest, Add) {
  this->SetUpAdd();
  this->RunAndVerify();
}

}  // namespace zkx::cpu
