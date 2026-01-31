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

#include "zkx/backends/cpu/codegen/msm_test.h"

namespace zkx::cpu {

using GroupTypes = testing::Types<zk_dtypes::bn254::G1AffinePointMont>;
TYPED_TEST_SUITE(MSMTest, GroupTypes);

TYPED_TEST(MSMTest, G1MSM) {
  this->SetUpMSM();
  this->RunAndVerify();
}

}  // namespace zkx::cpu
