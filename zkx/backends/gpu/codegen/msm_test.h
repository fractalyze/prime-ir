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

#ifndef ZKX_BACKENDS_GPU_CODEGEN_MSM_TEST_H_
#define ZKX_BACKENDS_GPU_CODEGEN_MSM_TEST_H_

#include <stddef.h>

#include <string_view>
#include <vector>

#include "zkx/backends/gpu/codegen/cuda_kernel_emitter_test.h"
#include "zkx/base/containers/container_util.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"

namespace zkx::gpu {

template <typename AffinePoint>
class MSMTest : public CudaKernelEmitterTest {
 public:
  using ScalarField = typename AffinePoint::ScalarField;
  using JacobianPoint = typename AffinePoint::JacobianPoint;

  void SetUp() override {
    CudaKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<ScalarField>());
    y_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<AffinePoint>());
    // GPU MSM (ICICLE) always returns affine Montgomery form.
    ret_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<AffinePoint>());
    num_scalar_muls_ = 1024;
    x_ = base::CreateVector(num_scalar_muls_,
                            []() { return ScalarField::Random(); });
    y_ = base::CreateVector(num_scalar_muls_,
                            []() { return AffinePoint::Random(); });
    literals_.push_back(LiteralUtil::CreateR1<ScalarField>(x_));
    literals_.push_back(LiteralUtil::CreateR1<AffinePoint>(y_));
  }

 protected:
  void SetUpMSM() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$3] parameter(0)
        %y = $1[$3] parameter(1)

        ROOT %ret = $2[] msm(%x, %y)
      }
    )",
                                 x_typename_, y_typename_, ret_typename_,
                                 num_scalar_muls_);
    JacobianPoint ret;
    for (size_t i = 0; i < x_.size(); ++i) {
      ret += x_[i] * y_[i];
    }
    expected_literal_ = LiteralUtil::CreateR0<AffinePoint>(ret.ToAffine());
  }

 private:
  std::string_view y_typename_;
  std::string_view ret_typename_;
  size_t num_scalar_muls_;
  std::vector<ScalarField> x_;
  std::vector<AffinePoint> y_;
};

}  // namespace zkx::gpu

#endif  // ZKX_BACKENDS_GPU_CODEGEN_MSM_TEST_H_
