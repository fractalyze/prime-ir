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

#ifndef ZKX_BACKENDS_CPU_CODEGEN_MSM_TEST_H_
#define ZKX_BACKENDS_CPU_CODEGEN_MSM_TEST_H_

#include <stddef.h>

#include <string_view>
#include <vector>

#include "xla/tsl/platform/cpu_info.h"
#include "zkx/backends/cpu/codegen/cpu_kernel_emitter_test.h"
#include "zkx/base/containers/container_util.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"

namespace zkx::cpu {

template <typename AffinePoint>
class MSMTest : public CpuKernelEmitterTest {
 public:
  using ScalarField = typename AffinePoint::ScalarField;
  using JacobianPoint = typename AffinePoint::JacobianPoint;

  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<ScalarField>());
    y_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<AffinePoint>());
    ret_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<JacobianPoint>());
  }

 protected:
  // MSM with num_scalar_muls > num_threads (parallel path).
  void SetUpMSM() {
    num_scalar_muls_ = tsl::port::MaxParallelism() + 1;
    InitInputs();
    SetUpMSMImpl();
  }

  // MSM with num_scalar_muls = 1, which is < num_threads (small path).
  // Exercises the single-threaded MSMOp + FromElementsOp wrapping.
  void SetUpMSMSmall() {
    num_scalar_muls_ = 1;
    InitInputs();
    SetUpMSMImpl();
  }

 private:
  void InitInputs() {
    x_ = base::CreateVector(num_scalar_muls_,
                            []() { return ScalarField::Random(); });
    y_ = base::CreateVector(num_scalar_muls_,
                            []() { return AffinePoint::Random(); });
    literals_.push_back(LiteralUtil::CreateR1<ScalarField>(x_));
    literals_.push_back(LiteralUtil::CreateR1<AffinePoint>(y_));
  }

  void SetUpMSMImpl() {
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
    expected_literal_ = LiteralUtil::CreateR0<JacobianPoint>(ret);
  }

 private:
  std::string_view y_typename_;
  std::string_view ret_typename_;
  size_t num_scalar_muls_;
  std::vector<ScalarField> x_;
  std::vector<AffinePoint> y_;
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_CODEGEN_MSM_TEST_H_
