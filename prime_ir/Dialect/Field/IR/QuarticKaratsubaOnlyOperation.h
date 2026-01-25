// Copyright 2026 The PrimeIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#ifndef PRIME_IR_DIALECT_FIELD_IR_QUARTICKARATSUBAONLYOPERATION_H_
#define PRIME_IR_DIALECT_FIELD_IR_QUARTICKARATSUBAONLYOPERATION_H_

#include <array>

#include "zk_dtypes/include/field/extension_field_operation.h"
#include "zk_dtypes/include/field/karatsuba_operation.h"

namespace mlir::prime_ir::field {

// A quartic extension field operation class that only uses Karatsuba.
// This is used for tower extensions where ToomCook's Vandermonde matrix
// interpolation doesn't work (it requires scalar multiplication with prime
// field elements, but tower base field elements are extension fields).
template <typename Derived>
class QuarticKaratsubaOnlyOperation
    : public zk_dtypes::ExtensionFieldOperation<Derived>,
      public zk_dtypes::KaratsubaOperation<Derived> {
public:
  using BaseField =
      typename zk_dtypes::ExtensionFieldOperationTraits<Derived>::BaseField;

  // Multiplication using Karatsuba only.
  Derived operator*(const Derived &other) const {
    return this->KaratsubaMultiply(other);
  }

  // Square using Karatsuba only.
  Derived Square() const { return this->KaratsubaSquare(); }

  // Returns the multiplicative inverse using tower formula.
  // This is the same algorithm as QuarticExtensionFieldOperation::Inverse().
  Derived Inverse() const {
    // Fp4 inverse via quadratic tower (u⁴ = ξ) with v = u²:
    // Write x = A + B·u where A = x₀ + x₂·v and B = x₁ + x₃·v in Fp₂[v]/(v² −
    // ξ). Then x⁻¹ = (A − B·u) · (A² − v·B²)⁻¹. If D = A² − v·B² = D₀ + D₁·v,
    // we invert D in Fp₂ by D⁻¹ = (D₀ − D₁·v)/(D₀² − ξ·D₁²), and expand back
    // to {1,u,u²,u³}.

    const std::array<BaseField, 4> &x =
        static_cast<const Derived &>(*this).ToCoeffs();
    BaseField xi = static_cast<const Derived &>(*this).NonResidue(); // ξ

    // 1) Compute A² and B² in Fp₂[v]/(v² − ξ), for A = x₀ + x₂·v and B = x₁ +
    // x₃·v (v² = ξ): A² = (x₀² + ξ·x₂²) + (2·x₀·x₂)·v B² = (x₁² + ξ·x₃²) +
    // (2·x₁·x₃)·v
    BaseField x0_sq = x[0].Square();
    BaseField x1_sq = x[1].Square();
    BaseField x2_sq = x[2].Square();
    BaseField x3_sq = x[3].Square();

    BaseField A0 = x0_sq + xi * x2_sq;      // real part of A²
    BaseField A1 = (x[0] * x[2]).Double();  // v part of A²
    BaseField B0 = x1_sq + xi * x3_sq;      // real part of B²
    BaseField B1 = (x[1] * x[3]).Double();  // v part of B²

    // 2) Compute D = A² − v·B². Since v·(B₀ + B₁·v) = (ξ·B₁) + (B₀)·v,
    // D = (A₀ − ξ·B₁) + (A₁ − B₀)·v.
    BaseField D0 = A0 - xi * B1;
    BaseField D1 = A1 - B0;

    // 3) Compute the norm N = D₀² − ξ·D₁² ∈ Fp and its inverse N⁻¹ in the base
    // field. Inverse() returns Zero() if not invertible.
    BaseField N = D0.Square() - xi * D1.Square();
    BaseField N_inv = N.Inverse();

    // 4) Invert D in Fp₂: D⁻¹ = (D₀ − D₁·v) · N⁻¹ = C₀ + C₁·v, where C₀ =
    // D₀·N⁻¹ and C₁ = −D₁·N⁻¹.
    BaseField C0 = D0 * N_inv;
    BaseField C1 = -D1 * N_inv;

    // 5) Final expansion: x⁻¹ = (A − B·u) · (C₀ + C₁·v), with v = u².
    // In the basis {1, u, u², u³}:
    // y₀ = x₀·C₀ + ξ·x₂·C₁
    // y₁ = −(x₁·C₀ + ξ·x₃·C₁)
    // y₂ = x₂·C₀ + x₀·C₁
    // y₃ = −(x₃·C₀ + x₁·C₁)
    BaseField y0 = x[0] * C0 + xi * (x[2] * C1);
    BaseField y1 = -(x[1] * C0 + xi * (x[3] * C1));
    BaseField y2 = x[2] * C0 + x[0] * C1;
    BaseField y3 = -(x[3] * C0 + x[1] * C1);

    return static_cast<const Derived &>(*this).FromCoeffs({y0, y1, y2, y3});
  }
};

} // namespace mlir::prime_ir::field

#endif // PRIME_IR_DIALECT_FIELD_IR_QUARTICKARATSUBAONLYOPERATION_H_
