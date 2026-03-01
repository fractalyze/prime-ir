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

#ifndef ZKX_BACKENDS_CPU_CODEGEN_PAIRING_TEST_H_
#define ZKX_BACKENDS_CPU_CODEGEN_PAIRING_TEST_H_

#include <string_view>
#include <vector>

#include "absl/strings/substitute.h"

#include "zkx/backends/cpu/codegen/cpu_kernel_emitter_test.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"

namespace zkx::cpu {

template <typename G1AffinePoint, typename G2AffinePoint>
class PairingTest : public CpuKernelEmitterTest {
 public:
  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    g1_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<G1AffinePoint>());
    g2_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<G2AffinePoint>());
    p_ = G1AffinePoint::Random();
    q_ = G2AffinePoint::Random();
  }

 protected:
  // e(P, Q) * e(-P, Q) = e(P, Q) * e(P, Q)⁻¹ = 1
  void SetUpPairingCheckValid() {
    std::vector<G1AffinePoint> g1 = {p_, -p_};
    std::vector<G2AffinePoint> g2 = {q_, q_};
    literals_.push_back(LiteralUtil::CreateR1<G1AffinePoint>(g1));
    literals_.push_back(LiteralUtil::CreateR1<G2AffinePoint>(g2));

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %g1 = $0[2] parameter(0)
        %g2 = $1[2] parameter(1)

        ROOT %ret = pred[] pairing-check(%g1, %g2)
      }
    )",
                                 g1_typename_, g2_typename_);
    expected_literal_ = LiteralUtil::CreateR0<bool>(true);
  }

  // e(P, Q) * e(P, Q) = e(P, Q)² != 1
  // (GT has prime order r, so no element of order 2)
  void SetUpPairingCheckInvalid() {
    std::vector<G1AffinePoint> g1 = {p_, p_};
    std::vector<G2AffinePoint> g2 = {q_, q_};
    literals_.push_back(LiteralUtil::CreateR1<G1AffinePoint>(g1));
    literals_.push_back(LiteralUtil::CreateR1<G2AffinePoint>(g2));

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %g1 = $0[2] parameter(0)
        %g2 = $1[2] parameter(1)

        ROOT %ret = pred[] pairing-check(%g1, %g2)
      }
    )",
                                 g1_typename_, g2_typename_);
    expected_literal_ = LiteralUtil::CreateR0<bool>(false);
  }

 private:
  std::string_view g1_typename_;
  std::string_view g2_typename_;
  G1AffinePoint p_;
  G2AffinePoint q_;
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_CODEGEN_PAIRING_TEST_H_
