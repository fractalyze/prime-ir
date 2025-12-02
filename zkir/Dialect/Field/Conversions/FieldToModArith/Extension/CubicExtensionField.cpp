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

#include "zkir/Dialect/Field/Conversions/FieldToModArith/Extension/CubicExtensionField.h"

#include "zkir/Dialect/Field/Conversions/FieldToModArith/ConversionUtils.h"
#include "zkir/Dialect/ModArith/IR/ModArithOps.h"

namespace mlir::zkir::field {

Value CubicExtensionField::square(Value x) {
  // CH-SQR2 algorithm from "Multiplication and Squaring on Pairing-Friendly
  // Fields" by Devegili, OhEigeartaigh, Scott, Dahab (Section 4).
  // https://eprint.iacr.org/2006/471.pdf
  //
  // For x = x₀ + x₁ * u + x₂ * u² where u³ = xi:
  //
  // s₀ = x₀²
  // s₁ = 2 * x₀ * x₁
  // s₂ = (x₀ - x₁ + x₂)²
  // s₃ = 2 * x₁ * x₂
  // s₄ = x₂²
  //
  // Result:
  // y₀ = s₀ + xi * s₃
  // y₁ = s₁ + xi * s₄
  // y₂ = s₁ + s₂ + s₃ - s₀ - s₄

  auto coeffs = toCoeffs(b, x);
  auto x0 = coeffs[0];
  auto x1 = coeffs[1];
  auto x2 = coeffs[2];

  // s₀ = x₀²
  auto s0 = b.create<mod_arith::SquareOp>(x0);

  // s₁ = 2 * x₀ * x₁
  auto c0C1 = b.create<mod_arith::MulOp>(x0, x1);
  auto s1 = b.create<mod_arith::DoubleOp>(c0C1);

  // s₂ = (x₀ - x₁ + x₂)²
  auto x0MinusX1 = b.create<mod_arith::SubOp>(x0, x1);
  auto x0MinusX1PlusX2 = b.create<mod_arith::AddOp>(x0MinusX1, x2);
  auto s2 = b.create<mod_arith::SquareOp>(x0MinusX1PlusX2);

  // s₃ = 2 * x₁ * x₂
  auto x1TimesX2 = b.create<mod_arith::MulOp>(x1, x2);
  auto s3 = b.create<mod_arith::DoubleOp>(x1TimesX2);

  // s₄ = x₂²
  auto s4 = b.create<mod_arith::SquareOp>(x2);

  // y₀ = s₀ + xi * s₃
  auto xiTimesS3 = b.create<mod_arith::MulOp>(nonResidue, s3);
  auto y0 = b.create<mod_arith::AddOp>(s0, xiTimesS3);

  // y₁ = s₁ + xi * s₄
  auto xiTimesS4 = b.create<mod_arith::MulOp>(nonResidue, s4);
  auto y1 = b.create<mod_arith::AddOp>(s1, xiTimesS4);

  // y₂ = s₁ + s₂ + s₃ - s₀ - s₄
  auto s1PlusS2 = b.create<mod_arith::AddOp>(s1, s2);
  auto s1PlusS2PlusS3 = b.create<mod_arith::AddOp>(s1PlusS2, s3);
  auto s1PlusS2PlusS3MinusS0 = b.create<mod_arith::SubOp>(s1PlusS2PlusS3, s0);
  auto y2 = b.create<mod_arith::SubOp>(s1PlusS2PlusS3MinusS0, s4);

  return fromCoeffs(b, type, {y0, y1, y2});
}

Value CubicExtensionField::mul(Value x, Value y) {
  // See https://eprint.iacr.org/2006/471.pdf
  // Devegili OhEig Scott Dahab --- Multiplication and Squaring on
  // AbstractPairing-Friendly Fields.pdf; Section 4 (Karatsuba)
  //
  // For x = x₀ + x₁ * u + x₂ * u² and y = y₀ + y₁ * u + y₂ * u²
  // where u³ = xi:
  //
  // v₀ = x₀ * y₀
  // v₁ = x₁ * y₁
  // v₂ = x₂ * y₂
  // v₃ = (x₀ + x₁) * (y₀ + y₁) - v₀ - v₁
  // v₄ = (x₀ + x₂) * (y₀ + y₂) - v₀ - v₂
  // v₅ = (x₁ + x₂) * (y₁ + y₂) - v₁ - v₂
  //
  // Result:
  // z₀ = v₀ + xi * v₅
  // z₁ = v₃ + xi * v₂
  // z₂ = v₄ + v₁

  auto xCoeffs = toCoeffs(b, x);
  auto x0 = xCoeffs[0];
  auto x1 = xCoeffs[1];
  auto x2 = xCoeffs[2];
  auto yCoeffs = toCoeffs(b, y);
  auto y0 = yCoeffs[0];
  auto y1 = yCoeffs[1];
  auto y2 = yCoeffs[2];

  auto v0 = b.create<mod_arith::MulOp>(x0, y0);
  auto v1 = b.create<mod_arith::MulOp>(x1, y1);
  auto v2 = b.create<mod_arith::MulOp>(x2, y2);

  // v₃ = (x₀ + x₁) * (y₀ + y₁) - v₀ - v₁
  auto x0PlusX1 = b.create<mod_arith::AddOp>(x0, x1);
  auto y0PlusY1 = b.create<mod_arith::AddOp>(y0, y1);
  Value v3 = b.create<mod_arith::MulOp>(x0PlusX1, y0PlusY1);
  v3 = b.create<mod_arith::SubOp>(v3, v0);
  v3 = b.create<mod_arith::SubOp>(v3, v1);

  // v₄ = (x₀ + x₂) * (y₀ + y₂) - v₀ - v₂
  auto x0PlusX2 = b.create<mod_arith::AddOp>(x0, x2);
  auto y0PlusY2 = b.create<mod_arith::AddOp>(y0, y2);
  Value v4 = b.create<mod_arith::MulOp>(x0PlusX2, y0PlusY2);
  v4 = b.create<mod_arith::SubOp>(v4, v0);
  v4 = b.create<mod_arith::SubOp>(v4, v2);

  // v₅ = (x₁ + x₂) * (y₁ + y₂) - v₁ - v₂
  auto x1PlusX2 = b.create<mod_arith::AddOp>(x1, x2);
  auto y1PlusY2 = b.create<mod_arith::AddOp>(y1, y2);
  Value v5 = b.create<mod_arith::MulOp>(x1PlusX2, y1PlusY2);
  v5 = b.create<mod_arith::SubOp>(v5, v1);
  v5 = b.create<mod_arith::SubOp>(v5, v2);

  // z₀ = v₀ + xi * v₅
  auto z0 = b.create<mod_arith::AddOp>(
      v0, b.create<mod_arith::MulOp>(nonResidue, v5));
  // z₁ = v₃ + xi * v₂
  auto z1 = b.create<mod_arith::AddOp>(
      v3, b.create<mod_arith::MulOp>(nonResidue, v2));
  // z₂ = v₄ + v₁
  auto z2 = b.create<mod_arith::AddOp>(v4, v1);

  return fromCoeffs(b, type, {z0, z1, z2});
}

Value CubicExtensionField::inverse(Value x) {
  // Inverse of a cubic extension field element.
  // For x = x₀ + x₁ * u + x₂ * u² where u³ = xi:
  //
  // t₀ = x₀² - xi * x₁ * x₂
  // t₁ = xi * x₂² - x₀ * x₁
  // t₂ = x₁² - x₀ * x₂
  // t₃ = x₀ * t₀ + xi * (x₂ * t₁ + x₁ * t₂)
  // t₄ = t₃^(-1)
  //
  // y₀ = t₀ * t₄
  // y₁ = t₁ * t₄
  // y₂ = t₂ * t₄

  auto coeffs = toCoeffs(b, x);
  auto x0 = coeffs[0];
  auto x1 = coeffs[1];
  auto x2 = coeffs[2];

  // t₀ = x₀² - xi * x₁ * x₂
  auto x0Squared = b.create<mod_arith::SquareOp>(x0);
  auto x1TimesX2 = b.create<mod_arith::MulOp>(x1, x2);
  auto xiTimesX1X2 = b.create<mod_arith::MulOp>(nonResidue, x1TimesX2);
  auto t0 = b.create<mod_arith::SubOp>(x0Squared, xiTimesX1X2);

  // t₁ = xi * x₂² - x₀ * x₁
  auto x2Squared = b.create<mod_arith::SquareOp>(x2);
  auto xiTimesX2Squared = b.create<mod_arith::MulOp>(nonResidue, x2Squared);
  auto x0TimesX1 = b.create<mod_arith::MulOp>(x0, x1);
  auto t1 = b.create<mod_arith::SubOp>(xiTimesX2Squared, x0TimesX1);

  // t₂ = x₁² - x₀ * x₂
  auto x1Squared = b.create<mod_arith::SquareOp>(x1);
  auto x0TimesX2 = b.create<mod_arith::MulOp>(x0, x2);
  auto t2 = b.create<mod_arith::SubOp>(x1Squared, x0TimesX2);

  // t₃ = x₀ * t₀ + xi * (x₂ * t₁ + x₁ * t₂)
  auto x0TimesT0 = b.create<mod_arith::MulOp>(x0, t0);
  auto x2TimesT1 = b.create<mod_arith::MulOp>(x2, t1);
  auto x1TimesT2 = b.create<mod_arith::MulOp>(x1, t2);
  auto x2T1PlusX1T2 = b.create<mod_arith::AddOp>(x2TimesT1, x1TimesT2);
  auto xiTimesX2T1PlusX1T2 =
      b.create<mod_arith::MulOp>(nonResidue, x2T1PlusX1T2);
  auto t3 = b.create<mod_arith::AddOp>(x0TimesT0, xiTimesX2T1PlusX1T2);

  // t₄ = t₃^(-1)
  auto t4 = b.create<mod_arith::InverseOp>(t3);

  // y₀ = t₀ * t₄
  // y₁ = t₁ * t₄
  // y₂ = t₂ * t₄
  auto y0 = b.create<mod_arith::MulOp>(t0, t4);
  auto y1 = b.create<mod_arith::MulOp>(t1, t4);
  auto y2 = b.create<mod_arith::MulOp>(t2, t4);

  return fromCoeffs(b, type, {y0, y1, y2});
}

} // namespace mlir::zkir::field
