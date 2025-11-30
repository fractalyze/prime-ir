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

Value CubicExtensionField::square(Value v) {
  // CH-SQR2 algorithm from "Multiplication and Squaring on Pairing-Friendly
  // Fields" by Devegili, OhEigeartaigh, Scott, Dahab (Section 4).
  // https://eprint.iacr.org/2006/471.pdf
  //
  // For v = c0 + c1 * u + c2 * u² where u³ = xi:
  //
  // s0 = c0²
  // s1 = 2 * c0 * c1
  // s2 = (c0 - c1 + c2)²
  // s3 = 2 * c1 * c2
  // s4 = c2²
  //
  // Result:
  // r0 = s0 + xi * s3
  // r1 = s1 + xi * s4
  // r2 = s1 + s2 + s3 - s0 - s4

  auto coeffs = toCoeffs(b, v);
  auto c0 = coeffs[0];
  auto c1 = coeffs[1];
  auto c2 = coeffs[2];

  // s0 = c0²
  auto s0 = b.create<mod_arith::SquareOp>(c0);

  // s1 = 2 * c0 * c1
  auto c0C1 = b.create<mod_arith::MulOp>(c0, c1);
  auto s1 = b.create<mod_arith::DoubleOp>(c0C1);

  // s2 = (c0 - c1 + c2)²
  auto c0MinusC1 = b.create<mod_arith::SubOp>(c0, c1);
  auto c0MinusC1PlusC2 = b.create<mod_arith::AddOp>(c0MinusC1, c2);
  auto s2 = b.create<mod_arith::SquareOp>(c0MinusC1PlusC2);

  // s3 = 2 * c1 * c2
  auto c1C2 = b.create<mod_arith::MulOp>(c1, c2);
  auto s3 = b.create<mod_arith::DoubleOp>(c1C2);

  // s4 = c2²
  auto s4 = b.create<mod_arith::SquareOp>(c2);

  // r0 = s0 + xi * s3
  auto xiS3 = b.create<mod_arith::MulOp>(nonResidue, s3);
  auto r0 = b.create<mod_arith::AddOp>(s0, xiS3);

  // r1 = s1 + xi * s4
  auto xiS4 = b.create<mod_arith::MulOp>(nonResidue, s4);
  auto r1 = b.create<mod_arith::AddOp>(s1, xiS4);

  // r2 = s1 + s2 + s3 - s0 - s4
  auto r2Tmp = b.create<mod_arith::AddOp>(s1, s2);
  auto r2Tmp2 = b.create<mod_arith::AddOp>(r2Tmp, s3);
  auto r2Tmp3 = b.create<mod_arith::SubOp>(r2Tmp2, s0);
  auto r2 = b.create<mod_arith::SubOp>(r2Tmp3, s4);

  return fromCoeffs(b, type, {r0, r1, r2});
}

Value CubicExtensionField::mul(Value x, Value y) {
  // Schoolbook multiplication for cubic extension field.
  // For x = x0 + x1 * u + x2 * u² and y = y0 + y1 * u + y2 * u²
  // where u³ = xi:
  //
  // v0 = x0 * y0
  // v1 = x1 * y2
  // v2 = x2 * y1
  // v3 = x0 * y1
  // v4 = x1 * y0
  // v5 = x2 * y2
  // v6 = x0 * y2
  // v7 = x1 * y1
  // v8 = x2 * y0
  //
  // r0 = v0 + xi * (v1 + v2)
  // r1 = v3 + v4 + xi * v5
  // r2 = v6 + v7 + v8

  auto xCoeffs = toCoeffs(b, x);
  auto x0 = xCoeffs[0];
  auto x1 = xCoeffs[1];
  auto x2 = xCoeffs[2];
  auto yCoeffs = toCoeffs(b, y);
  auto y0 = yCoeffs[0];
  auto y1 = yCoeffs[1];
  auto y2 = yCoeffs[2];

  auto v0 = b.create<mod_arith::MulOp>(x0, y0);
  auto v1 = b.create<mod_arith::MulOp>(x1, y2);
  auto v2 = b.create<mod_arith::MulOp>(x2, y1);
  auto v3 = b.create<mod_arith::MulOp>(x0, y1);
  auto v4 = b.create<mod_arith::MulOp>(x1, y0);
  auto v5 = b.create<mod_arith::MulOp>(x2, y2);
  auto v6 = b.create<mod_arith::MulOp>(x0, y2);
  auto v7 = b.create<mod_arith::MulOp>(x1, y1);
  auto v8 = b.create<mod_arith::MulOp>(x2, y0);

  // r0 = v0 + xi * (v1 + v2)
  auto sumV1V2 = b.create<mod_arith::AddOp>(v1, v2);
  auto xiTimesSumV1V2 = b.create<mod_arith::MulOp>(nonResidue, sumV1V2);
  auto r0 = b.create<mod_arith::AddOp>(v0, xiTimesSumV1V2);

  // r1 = v3 + v4 + xi * v5
  auto xiTimesV5 = b.create<mod_arith::MulOp>(nonResidue, v5);
  auto sumV3V4 = b.create<mod_arith::AddOp>(v3, v4);
  auto r1 = b.create<mod_arith::AddOp>(sumV3V4, xiTimesV5);

  // r2 = v6 + v7 + v8
  auto sumV6V7 = b.create<mod_arith::AddOp>(v6, v7);
  auto r2 = b.create<mod_arith::AddOp>(sumV6V7, v8);

  return fromCoeffs(b, type, {r0, r1, r2});
}

Value CubicExtensionField::inverse(Value v) {
  // Inverse of a cubic extension field element.
  // For v = c0 + c1 * u + c2 * u² where u³ = xi:
  //
  // t0 = c0² - xi * c1 * c2
  // t1 = xi * c2² - c0 * c1
  // t2 = c1² - c0 * c2
  // t3 = c0 * t0 + xi * (c2 * t1 + c1 * t2)
  // t4 = t3^(-1)
  //
  // r0 = t0 * t4
  // r1 = t1 * t4
  // r2 = t2 * t4

  auto coeffs = toCoeffs(b, v);
  auto c0 = coeffs[0];
  auto c1 = coeffs[1];
  auto c2 = coeffs[2];

  // t0 = c0² - xi * c1 * c2
  auto c0Squared = b.create<mod_arith::SquareOp>(c0);
  auto c1TimesC2 = b.create<mod_arith::MulOp>(c1, c2);
  auto xiTimesC1C2 = b.create<mod_arith::MulOp>(nonResidue, c1TimesC2);
  auto t0 = b.create<mod_arith::SubOp>(c0Squared, xiTimesC1C2);

  // t1 = xi * c2² - c0 * c1
  auto c2Squared = b.create<mod_arith::SquareOp>(c2);
  auto xiTimesC2Squared = b.create<mod_arith::MulOp>(nonResidue, c2Squared);
  auto c0TimesC1 = b.create<mod_arith::MulOp>(c0, c1);
  auto t1 = b.create<mod_arith::SubOp>(xiTimesC2Squared, c0TimesC1);

  // t2 = c1² - c0 * c2
  auto c1Squared = b.create<mod_arith::SquareOp>(c1);
  auto c0TimesC2 = b.create<mod_arith::MulOp>(c0, c2);
  auto t2 = b.create<mod_arith::SubOp>(c1Squared, c0TimesC2);

  // t3 = c0 * t0 + xi * (c2 * t1 + c1 * t2)
  auto c0TimesT0 = b.create<mod_arith::MulOp>(c0, t0);
  auto c2TimesT1 = b.create<mod_arith::MulOp>(c2, t1);
  auto c1TimesT2 = b.create<mod_arith::MulOp>(c1, t2);
  auto sumT1T2 = b.create<mod_arith::AddOp>(c2TimesT1, c1TimesT2);
  auto xiTimesSumT1T2 = b.create<mod_arith::MulOp>(nonResidue, sumT1T2);
  auto t3 = b.create<mod_arith::AddOp>(c0TimesT0, xiTimesSumT1T2);

  // t4 = t3^(-1)
  auto t4 = b.create<mod_arith::InverseOp>(t3);

  // r0 = t0 * t4
  // r1 = t1 * t4
  // r2 = t2 * t4
  auto r0 = b.create<mod_arith::MulOp>(t0, t4);
  auto r1 = b.create<mod_arith::MulOp>(t1, t4);
  auto r2 = b.create<mod_arith::MulOp>(t2, t4);

  return fromCoeffs(b, type, {r0, r1, r2});
}

} // namespace mlir::zkir::field
