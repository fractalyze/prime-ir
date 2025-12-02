/* Copyright 2025 The ZKIR Authors.

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

#include "zkir/Dialect/Field/Conversions/FieldToModArith/Extension/QuadraticExtensionField.h"

#include "zkir/Dialect/Field/Conversions/FieldToModArith/ConversionUtils.h"
#include "zkir/Dialect/ModArith/IR/ModArithOps.h"

namespace mlir::zkir::field {

Value QuadraticExtensionField::square(Value x) {
  // For x = x₀ + x₁ * u where u² = β:

  auto coeffs = toCoeffs(b, x);
  auto x0 = coeffs[0];
  auto x1 = coeffs[1];

  // v₀ = x₀ - x₁
  auto v0 = b.create<mod_arith::SubOp>(x0, x1);

  // v₁ = x₀ - βx₁
  auto betaTimesX1 = b.create<mod_arith::MulOp>(nonResidue, x1);
  auto v1 = b.create<mod_arith::SubOp>(x0, betaTimesX1);

  // v₂ = x₀ * x₁
  auto v2 = b.create<mod_arith::MulOp>(x0, x1);

  // v₃ = v₀ * v₁ + v₂
  auto v0TimesV1 = b.create<mod_arith::MulOp>(v0, v1);
  auto v3 = b.create<mod_arith::AddOp>(v0TimesV1, v2);

  // y₁ = 2 * v₂
  auto y1 = b.create<mod_arith::DoubleOp>(v2);
  // y₀ = v₃ + βv₂
  auto betaTimesV2 = b.create<mod_arith::MulOp>(nonResidue, v2);
  auto y0 = b.create<mod_arith::AddOp>(v3, betaTimesV2);
  return fromCoeffs(b, type, {y0, y1});
}

Value QuadraticExtensionField::mul(Value x, Value y) {
  // For x = x₀ + x₁ * u and y = y₀ + y₁ * u where u² = β:

  auto xCoeffs = toCoeffs(b, x);
  auto x0 = xCoeffs[0];
  auto x1 = xCoeffs[1];
  auto yCoeffs = toCoeffs(b, y);
  auto y0 = yCoeffs[0];
  auto y1 = yCoeffs[1];

  // v₀ = x₀ * y₀
  // v₁ = x₁ * y₁
  auto v0 = b.create<mod_arith::MulOp>(x0, y0);
  auto v1 = b.create<mod_arith::MulOp>(x1, y1);

  // z₀ = v₀ + βv₁
  auto betaTimesV1 = b.create<mod_arith::MulOp>(nonResidue, v1);
  auto z0 = b.create<mod_arith::AddOp>(v0, betaTimesV1);

  // z₁ = (x₀ + x₁)(y₀ + y₁) - v₀ - v₁
  auto sumX = b.create<mod_arith::AddOp>(x0, x1);
  auto sumY = b.create<mod_arith::AddOp>(y0, y1);
  auto sumProduct = b.create<mod_arith::MulOp>(sumX, sumY);
  auto z1 = b.create<mod_arith::SubOp>(sumProduct, v0);
  z1 = b.create<mod_arith::SubOp>(z1, v1);

  return fromCoeffs(b, type, {z0, z1});
}

Value QuadraticExtensionField::inverse(Value x) {
  // For x = x₀ + x₁ * u where u² = β:

  auto coeffs = toCoeffs(b, x);
  auto x0 = coeffs[0];
  auto x1 = coeffs[1];

  // denominator = x₀² - x₁²β
  auto x0Squared = b.create<mod_arith::SquareOp>(x0);
  auto x1Squared = b.create<mod_arith::SquareOp>(x1);
  auto betaTimesX1Squared = b.create<mod_arith::MulOp>(nonResidue, x1Squared);
  auto denominator = b.create<mod_arith::SubOp>(x0Squared, betaTimesX1Squared);
  auto denominatorInv = b.create<mod_arith::InverseOp>(denominator);

  // y₀ = x₀ / denominator
  auto y0 = b.create<mod_arith::MulOp>(x0, denominatorInv);
  // y₁ = -x₁ / denominator
  auto x1Negated = b.create<mod_arith::NegateOp>(x1);
  auto y1 = b.create<mod_arith::MulOp>(x1Negated, denominatorInv);
  return fromCoeffs(b, type, {y0, y1});
}

} // namespace mlir::zkir::field
