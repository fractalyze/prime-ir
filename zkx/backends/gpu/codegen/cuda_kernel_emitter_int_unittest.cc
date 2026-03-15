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

#include "zkx/backends/gpu/codegen/int_test.h"

namespace zkx::gpu {

using IntTypes = testing::Types<int32_t, uint32_t>;
TYPED_TEST_SUITE(IntScalarUnaryTest, IntTypes);

TYPED_TEST(IntScalarUnaryTest, ConvertUp) {
  this->SetUpConvertUp();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarUnaryTest, ConvertDown) {
  this->SetUpConvertDown();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarUnaryTest, Negate) {
  this->SetUpNegate();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(IntScalarBinaryTest, IntTypes);

TYPED_TEST(IntScalarBinaryTest, Add) {
  this->SetUpAdd();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarBinaryTest, Div) {
  this->SetUpDiv();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarBinaryTest, Mul) {
  this->SetUpMul();
  this->RunAndVerify();
}

// TODO(chokobole): Add tests for power.

TYPED_TEST(IntScalarBinaryTest, Sub) {
  this->SetUpSub();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(IntR2TensorBinaryTest, IntTypes);

TYPED_TEST(IntR2TensorBinaryTest, Add) {
  this->SetUpAdd();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(IntTest, IntTypes);

TYPED_TEST(IntTest, Slice) {
  this->SetUpSlice();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, BitcastConvertDtypesToSmaller) {
  this->SetUpBitcastConvertDtypesToSmaller();
  this->RunAndVerify(/*run_hlo_passes=*/true);
}

TYPED_TEST(IntTest, BitcastConvertDtypesToLarger) {
  this->SetUpBitcastConvertDtypesToLarger();
  this->RunAndVerify(/*run_hlo_passes=*/true);
}

}  // namespace zkx::gpu
