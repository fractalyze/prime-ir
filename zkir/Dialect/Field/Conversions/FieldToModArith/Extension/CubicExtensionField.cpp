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

#include "zkir/Dialect/ModArith/IR/ModArithOps.h"

namespace mlir::zkir::field {

CubicExtensionField::CubicExtensionField(ImplicitLocOpBuilder &b, Value xi)
    : b_(b), xi_(xi) {}

Value CubicExtensionField::mulByNonResidue(Value v) {
  return b_.create<mod_arith::MulOp>(xi_, v);
}

SmallVector<Value, 3> CubicExtensionField::square(Value c0, Value c1,
                                                  Value c2) {
  // CH-SQR2 algorithm from "Multiplication and Squaring on Pairing-Friendly
  // Fields" by Devegili, OhEigeartaigh, Scott, Dahab (Section 4).
  // https://eprint.iacr.org/2006/471.pdf
  //
  // For a = c0 + c1*u + c2*u² where u^3 = xi:
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

  // s0 = c0²
  auto s0 = b_.create<mod_arith::SquareOp>(c0);

  // s1 = 2 * c0 * c1
  auto c0C1 = b_.create<mod_arith::MulOp>(c0, c1);
  auto s1 = b_.create<mod_arith::DoubleOp>(c0C1);

  // s2 = (c0 - c1 + c2)²
  auto c0MinusC1 = b_.create<mod_arith::SubOp>(c0, c1);
  auto c0MinusC1PlusC2 = b_.create<mod_arith::AddOp>(c0MinusC1, c2);
  auto s2 = b_.create<mod_arith::SquareOp>(c0MinusC1PlusC2);

  // s3 = 2 * c1 * c2
  auto c1C2 = b_.create<mod_arith::MulOp>(c1, c2);
  auto s3 = b_.create<mod_arith::DoubleOp>(c1C2);

  // s4 = c2²
  auto s4 = b_.create<mod_arith::SquareOp>(c2);

  // r0 = s0 + xi * s3
  auto xiS3 = mulByNonResidue(s3);
  auto r0 = b_.create<mod_arith::AddOp>(s0, xiS3);

  // r1 = s1 + xi * s4
  auto xiS4 = mulByNonResidue(s4);
  auto r1 = b_.create<mod_arith::AddOp>(s1, xiS4);

  // r2 = s1 + s2 + s3 - s0 - s4
  auto r2Tmp = b_.create<mod_arith::AddOp>(s1, s2);
  auto r2Tmp2 = b_.create<mod_arith::AddOp>(r2Tmp, s3);
  auto r2Tmp3 = b_.create<mod_arith::SubOp>(r2Tmp2, s0);
  auto r2 = b_.create<mod_arith::SubOp>(r2Tmp3, s4);

  return {r0, r1, r2};
}

SmallVector<Value, 3> CubicExtensionField::mul(Value a0, Value a1, Value a2,
                                               Value b0, Value b1, Value b2) {
  // Schoolbook multiplication for cubic extension field.
  // For a = a0 + a1*u + a2*u² and b = b0 + b1*u + b2*u²
  // where u^3 = xi:
  //
  // v0 = a0 * b0
  // v1 = a1 * b2
  // v2 = a2 * b1
  // v3 = a0 * b1
  // v4 = a1 * b0
  // v5 = a2 * b2
  // v6 = a0 * b2
  // v7 = a1 * b1
  // v8 = a2 * b0
  //
  // r0 = v0 + xi * (v1 + v2)
  // r1 = v3 + v4 + xi * v5
  // r2 = v6 + v7 + v8

  auto v0 = b_.create<mod_arith::MulOp>(a0, b0);
  auto v1 = b_.create<mod_arith::MulOp>(a1, b2);
  auto v2 = b_.create<mod_arith::MulOp>(a2, b1);
  auto v3 = b_.create<mod_arith::MulOp>(a0, b1);
  auto v4 = b_.create<mod_arith::MulOp>(a1, b0);
  auto v5 = b_.create<mod_arith::MulOp>(a2, b2);
  auto v6 = b_.create<mod_arith::MulOp>(a0, b2);
  auto v7 = b_.create<mod_arith::MulOp>(a1, b1);
  auto v8 = b_.create<mod_arith::MulOp>(a2, b0);

  // r0 = v0 + xi * (v1 + v2)
  auto sumV1V2 = b_.create<mod_arith::AddOp>(v1, v2);
  auto xiTimesSumV1V2 = mulByNonResidue(sumV1V2);
  auto r0 = b_.create<mod_arith::AddOp>(v0, xiTimesSumV1V2);

  // r1 = v3 + v4 + xi * v5
  auto xiTimesV5 = mulByNonResidue(v5);
  auto sumV3V4 = b_.create<mod_arith::AddOp>(v3, v4);
  auto r1 = b_.create<mod_arith::AddOp>(sumV3V4, xiTimesV5);

  // r2 = v6 + v7 + v8
  auto sumV6V7 = b_.create<mod_arith::AddOp>(v6, v7);
  auto r2 = b_.create<mod_arith::AddOp>(sumV6V7, v8);

  return {r0, r1, r2};
}

SmallVector<Value, 3> CubicExtensionField::inverse(Value c0, Value c1,
                                                   Value c2) {
  // Inverse of a cubic extension field element.
  // For a = c0 + c1*u + c2*u² where u^3 = xi:
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

  // t0 = c0² - xi * c1 * c2
  auto c0Squared = b_.create<mod_arith::SquareOp>(c0);
  auto c1TimesC2 = b_.create<mod_arith::MulOp>(c1, c2);
  auto xiTimesC1C2 = mulByNonResidue(c1TimesC2);
  auto t0 = b_.create<mod_arith::SubOp>(c0Squared, xiTimesC1C2);

  // t1 = xi * c2² - c0 * c1
  auto c2Squared = b_.create<mod_arith::SquareOp>(c2);
  auto xiTimesC2Squared = mulByNonResidue(c2Squared);
  auto c0TimesC1 = b_.create<mod_arith::MulOp>(c0, c1);
  auto t1 = b_.create<mod_arith::SubOp>(xiTimesC2Squared, c0TimesC1);

  // t2 = c1² - c0 * c2
  auto c1Squared = b_.create<mod_arith::SquareOp>(c1);
  auto c0TimesC2 = b_.create<mod_arith::MulOp>(c0, c2);
  auto t2 = b_.create<mod_arith::SubOp>(c1Squared, c0TimesC2);

  // t3 = c0 * t0 + xi * (c2 * t1 + c1 * t2)
  auto c0TimesT0 = b_.create<mod_arith::MulOp>(c0, t0);
  auto c2TimesT1 = b_.create<mod_arith::MulOp>(c2, t1);
  auto c1TimesT2 = b_.create<mod_arith::MulOp>(c1, t2);
  auto sumT1T2 = b_.create<mod_arith::AddOp>(c2TimesT1, c1TimesT2);
  auto xiTimesSumT1T2 = mulByNonResidue(sumT1T2);
  auto t3 = b_.create<mod_arith::AddOp>(c0TimesT0, xiTimesSumT1T2);

  // t4 = t3^(-1)
  auto t4 = b_.create<mod_arith::InverseOp>(t3);

  // r0 = t0 * t4
  // r1 = t1 * t4
  // r2 = t2 * t4
  auto r0 = b_.create<mod_arith::MulOp>(t0, t4);
  auto r1 = b_.create<mod_arith::MulOp>(t1, t4);
  auto r2 = b_.create<mod_arith::MulOp>(t2, t4);

  return {r0, r1, r2};
}

} // namespace mlir::zkir::field
