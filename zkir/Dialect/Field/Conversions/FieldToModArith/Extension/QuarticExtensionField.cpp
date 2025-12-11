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

#include "zkir/Dialect/Field/Conversions/FieldToModArith/Extension/QuarticExtensionField.h"

#include "mlir/Support/LLVM.h"
#include "zkir/Dialect/Field/Conversions/FieldToModArith/ConversionUtils.h"
#include "zkir/Dialect/Field/IR/FieldAttributes.h"
#include "zkir/Dialect/ModArith/IR/ModArithOps.h"
#include "zkir/Utils/APIntUtils.h"

namespace mlir::zkir::field {

Value QuarticExtensionField::toomCookInterpolate(Value v0, Value v1, Value v2,
                                                 Value v3, Value v4, Value v5,
                                                 Value v6) {
  // Toom-Cook interpolation for quartic extension fields.
  // Takes 7 evaluation points and computes result coefficients.
  // Following Tachyon's quartic_extension_field.h implementation.
  // https://github.com/kroma-network/tachyon/blob/e7b1306/tachyon/math/finite_fields/quartic_extension_field.h#L436-L468
  //
  // Result coefficients (Tachyon formulas):
  // z₀ = v₀ + ξ * (1/4 * v₀ - 1/6 * (v₁ + v₂) + 1/24 * (v₃ + v₄) - 5 * v₆)
  // z₁ = -1/3 * v₀ + v₁ - 1/2 * v₂ - 1/4 * v₃ + 1/20 * v₄ + 1/30 * v₅
  //      - 12 * v₆ + ξ * (-1/12 * (v₀ - v₁) + 1/24 * (v₂ - v₃)
  //      - 1/120 * (v₄ - v₅) - 3 * v₆)
  // z₂ = -5/4 * v₀ + 2/3 * (v₁ + v₂) - 1/24 * (v₃ + v₄) + 4 * v₆ + ξ * v₆
  // z₃ = 1/12 * (5 * v₀ - 7 * v₁) - 1/24 * (v₂ - 7 * v₃ + v₄ + v₅) + 15 * v₆

  auto baseField = cast<PrimeFieldType>(type.getBaseFieldType());

  // Create constants.
  auto inv2 = createInvConst(b, baseField, converter, 2);
  auto inv3 = createInvConst(b, baseField, converter, 3);
  auto inv4 = createInvConst(b, baseField, converter, 4);
  auto inv6 = createInvConst(b, baseField, converter, 6);
  auto inv12 = createInvConst(b, baseField, converter, 12);
  auto inv20 = createInvConst(b, baseField, converter, 20);
  auto inv24 = createInvConst(b, baseField, converter, 24);
  auto inv30 = createInvConst(b, baseField, converter, 30);
  auto inv120 = createInvConst(b, baseField, converter, 120);
  auto const3 = createConst(b, baseField, converter, 3);
  auto const5 = createConst(b, baseField, converter, 5);
  auto const7 = createConst(b, baseField, converter, 7);
  auto const12 = createConst(b, baseField, converter, 12);
  auto const15 = createConst(b, baseField, converter, 15);

  // Compute intermediate values for the result
  auto v1Plusv2 = b.create<mod_arith::AddOp>(v1, v2);
  auto v3Plusv4 = b.create<mod_arith::AddOp>(v3, v4);
  auto v0Minusv1 = b.create<mod_arith::SubOp>(v0, v1);
  auto v2Minusv3 = b.create<mod_arith::SubOp>(v2, v3);
  auto v4Minusv5 = b.create<mod_arith::SubOp>(v4, v5);
  auto v6x3 = b.create<mod_arith::MulOp>(v6, const3);
  auto v6x4 = b.create<mod_arith::DoubleOp>(b.create<mod_arith::DoubleOp>(v6));
  auto v6x5 = b.create<mod_arith::MulOp>(v6, const5);
  auto v6x12 = b.create<mod_arith::MulOp>(v6, const12);
  auto v6x15 = b.create<mod_arith::MulOp>(v6, const15);
  auto v0x5 = b.create<mod_arith::MulOp>(v0, const5);
  auto v1x7 = b.create<mod_arith::MulOp>(v1, const7);
  auto v3x7 = b.create<mod_arith::MulOp>(v3, const7);

  // z₀ = v₀ + ξ * (1/4 * v₀ - 1/6 * (v₁ + v₂) + 1/24 * (v₃ + v₄) - 5 * v₆)
  Value z0Inner = b.create<mod_arith::MulOp>(v0, inv4);
  z0Inner = b.create<mod_arith::SubOp>(
      z0Inner, b.create<mod_arith::MulOp>(v1Plusv2, inv6));
  z0Inner = b.create<mod_arith::AddOp>(
      z0Inner, b.create<mod_arith::MulOp>(v3Plusv4, inv24));
  z0Inner = b.create<mod_arith::SubOp>(z0Inner, v6x5);
  auto z0 = b.create<mod_arith::AddOp>(
      v0, b.create<mod_arith::MulOp>(nonResidue, z0Inner));

  // z₁ = -1/3 * v₀ + v₁ - 1/2 * v₂ - 1/4 * v₃ + 1/20 * v₄ + 1/30 * v₅
  //      - 12 * v₆ + ξ * (-1/12 * (v₀ - v₁) + 1/24 * (v₂ - v₃)
  //      - 1/120 * (v₄ - v₅) - 3 * v₆)
  Value z1Base =
      b.create<mod_arith::SubOp>(v1, b.create<mod_arith::MulOp>(v0, inv3));
  z1Base =
      b.create<mod_arith::SubOp>(z1Base, b.create<mod_arith::MulOp>(v2, inv2));
  z1Base =
      b.create<mod_arith::SubOp>(z1Base, b.create<mod_arith::MulOp>(v3, inv4));
  z1Base =
      b.create<mod_arith::AddOp>(z1Base, b.create<mod_arith::MulOp>(v4, inv20));
  z1Base =
      b.create<mod_arith::AddOp>(z1Base, b.create<mod_arith::MulOp>(v5, inv30));
  z1Base = b.create<mod_arith::SubOp>(z1Base, v6x12);

  Value z1Inner = b.create<mod_arith::MulOp>(v0Minusv1, inv12);
  z1Inner = b.create<mod_arith::SubOp>(
      b.create<mod_arith::MulOp>(v2Minusv3, inv24), z1Inner);
  z1Inner = b.create<mod_arith::SubOp>(
      z1Inner, b.create<mod_arith::MulOp>(v4Minusv5, inv120));
  z1Inner = b.create<mod_arith::SubOp>(z1Inner, v6x3);
  auto z1 = b.create<mod_arith::AddOp>(
      z1Base, b.create<mod_arith::MulOp>(nonResidue, z1Inner));

  // z₂ = -5/4 * v₀ + 2/3 * (v₁ + v₂) - 1/24 * (v₃ + v₄) + 4 * v₆ + ξ * v₆
  auto v0x5Inv4 = b.create<mod_arith::MulOp>(v0x5, inv4);
  Value z2 = b.create<mod_arith::SubOp>(
      b.create<mod_arith::MulOp>(b.create<mod_arith::DoubleOp>(v1Plusv2), inv3),
      v0x5Inv4);
  z2 = b.create<mod_arith::SubOp>(z2,
                                  b.create<mod_arith::MulOp>(v3Plusv4, inv24));
  z2 = b.create<mod_arith::AddOp>(z2, v6x4);
  z2 = b.create<mod_arith::AddOp>(z2,
                                  b.create<mod_arith::MulOp>(nonResidue, v6));

  // z₃ = 1/12 * (5 * v₀ - 7 * v₁) - 1/24 * (v₂ - 7 * v₃ + v₄ + v₅) + 15 * v₆
  auto v0x5Minusv1x7 = b.create<mod_arith::SubOp>(v0x5, v1x7);
  Value z3 = b.create<mod_arith::MulOp>(v0x5Minusv1x7, inv12);
  // v₂ - 7 * v₃ + v₄ + v₅
  Value inner24 = b.create<mod_arith::SubOp>(v2, v3x7);
  inner24 = b.create<mod_arith::AddOp>(inner24, v4);
  inner24 = b.create<mod_arith::AddOp>(inner24, v5);
  z3 = b.create<mod_arith::SubOp>(z3,
                                  b.create<mod_arith::MulOp>(inner24, inv24));
  z3 = b.create<mod_arith::AddOp>(z3, v6x15);

  return fromCoeffs(b, type, {z0, z1, z2, z3});
}

Value QuarticExtensionField::square(Value x) {
  // Toom-Cook squaring algorithm for quartic extension fields.
  // Following Tachyon's quartic_extension_field.h implementation.
  // https://github.com/kroma-network/tachyon/blob/e7b1306/tachyon/math/finite_fields/quartic_extension_field.h#L495-L563
  // https://eprint.iacr.org/2006/471.pdf
  //
  // For x = x₀ + x₁u + x₂u² + x₃u³ where u⁴ = ξ:
  //
  // Evaluation points: 0, 1, -1, 2, -2, 3, ∞
  // v₀ = x₀²
  // v₁ = (x₀ + x₁ + x₂ + x₃)²
  // v₂ = (x₀ - x₁ + x₂ - x₃)²
  // v₃ = (x₀ + 2x₁ + 4x₂ + 8x₃)²
  // v₄ = (x₀ - 2x₁ + 4x₂ - 8x₃)²
  // v₅ = (x₀ + 3x₁ + 9x₂ + 27x₃)²
  // v₆ = x₃²

  auto baseField = cast<PrimeFieldType>(type.getBaseFieldType());
  auto const3 = createConst(b, baseField, converter, 3);
  auto const9 = createConst(b, baseField, converter, 9);
  auto const27 = createConst(b, baseField, converter, 27);

  auto xCoeffs = toCoeffs(b, x);
  auto x0 = xCoeffs[0];
  auto x1 = xCoeffs[1];
  auto x2 = xCoeffs[2];
  auto x3 = xCoeffs[3];

  // Compute multiples of coefficients
  // 2x₁, 3x₁
  auto x1Times2 = b.create<mod_arith::DoubleOp>(x1);
  auto x1Times3 = b.create<mod_arith::MulOp>(x1, const3);
  // 4x₂, 9x₂
  auto x2Times2 = b.create<mod_arith::DoubleOp>(x2);
  auto x2Times4 = b.create<mod_arith::DoubleOp>(x2Times2);
  auto x2Times9 = b.create<mod_arith::MulOp>(x2, const9);
  // 8x₃, 27x₃
  auto x3Times2 = b.create<mod_arith::DoubleOp>(x3);
  auto x3Times4 = b.create<mod_arith::DoubleOp>(x3Times2);
  auto x3Times8 = b.create<mod_arith::DoubleOp>(x3Times4);
  auto x3Times27 = b.create<mod_arith::MulOp>(x3, const27);

  // v₀ = x₀²
  auto v0 = b.create<mod_arith::SquareOp>(x0);

  // v₁ = (x₀ + x₁ + x₂ + x₃)²
  Value sum1 = b.create<mod_arith::AddOp>(x0, x1);
  sum1 = b.create<mod_arith::AddOp>(sum1, x2);
  sum1 = b.create<mod_arith::AddOp>(sum1, x3);
  auto v1 = b.create<mod_arith::SquareOp>(sum1);

  // v₂ = (x₀ - x₁ + x₂ - x₃)²
  Value diff1 = b.create<mod_arith::SubOp>(x0, x1);
  diff1 = b.create<mod_arith::AddOp>(diff1, x2);
  diff1 = b.create<mod_arith::SubOp>(diff1, x3);
  auto v2 = b.create<mod_arith::SquareOp>(diff1);

  // v₃ = (x₀ + 2x₁ + 4x₂ + 8x₃)²
  Value sum2 = b.create<mod_arith::AddOp>(x0, x1Times2);
  sum2 = b.create<mod_arith::AddOp>(sum2, x2Times4);
  sum2 = b.create<mod_arith::AddOp>(sum2, x3Times8);
  auto v3 = b.create<mod_arith::SquareOp>(sum2);

  // v₄ = (x₀ - 2x₁ + 4x₂ - 8x₃)²
  Value diff2 = b.create<mod_arith::SubOp>(x0, x1Times2);
  diff2 = b.create<mod_arith::AddOp>(diff2, x2Times4);
  diff2 = b.create<mod_arith::SubOp>(diff2, x3Times8);
  auto v4 = b.create<mod_arith::SquareOp>(diff2);

  // v₅ = (x₀ + 3x₁ + 9x₂ + 27x₃)²
  Value sum3 = b.create<mod_arith::AddOp>(x0, x1Times3);
  sum3 = b.create<mod_arith::AddOp>(sum3, x2Times9);
  sum3 = b.create<mod_arith::AddOp>(sum3, x3Times27);
  auto v5 = b.create<mod_arith::SquareOp>(sum3);

  // v₆ = x₃²
  auto v6 = b.create<mod_arith::SquareOp>(x3);

  return toomCookInterpolate(v0, v1, v2, v3, v4, v5, v6);
}

Value QuarticExtensionField::mul(Value x, Value y) {
  // Toom-Cook multiplication algorithm for quartic extension fields.
  // Following Tachyon's quartic_extension_field.h implementation.
  // https://github.com/kroma-network/tachyon/blob/e7b1306/tachyon/math/finite_fields/quartic_extension_field.h#L394-L493
  // https://eprint.iacr.org/2006/471.pdf
  //
  // For x = x₀ + x₁u + x₂u² + x₃u³ and y = y₀ + y₁u + y₂u² + y₃u³
  // where u⁴ = ξ:
  //
  // Evaluation points: 0, 1, -1, 2, -2, 3, ∞
  // v₀ = x₀ * y₀
  // v₁ = (x₀ + x₁ + x₂ + x₃)(y₀ + y₁ + y₂ + y₃)
  // v₂ = (x₀ - x₁ + x₂ - x₃)(y₀ - y₁ + y₂ - y₃)
  // v₃ = (x₀ + 2x₁ + 4x₂ + 8x₃)(y₀ + 2y₁ + 4y₂ + 8y₃)
  // v₄ = (x₀ - 2x₁ + 4x₂ - 8x₃)(y₀ - 2y₁ + 4y₂ - 8y₃)
  // v₅ = (x₀ + 3x₁ + 9x₂ + 27x₃)(y₀ + 3y₁ + 9y₂ + 27y₃)
  // v₆ = x₃ * y₃

  auto baseField = cast<PrimeFieldType>(type.getBaseFieldType());
  auto const3 = createConst(b, baseField, converter, 3);
  auto const9 = createConst(b, baseField, converter, 9);
  auto const27 = createConst(b, baseField, converter, 27);

  auto xCoeffs = toCoeffs(b, x);
  auto x0 = xCoeffs[0];
  auto x1 = xCoeffs[1];
  auto x2 = xCoeffs[2];
  auto x3 = xCoeffs[3];
  auto yCoeffs = toCoeffs(b, y);
  auto y0 = yCoeffs[0];
  auto y1 = yCoeffs[1];
  auto y2 = yCoeffs[2];
  auto y3 = yCoeffs[3];

  // Compute multiples of x coefficients
  auto x1Times2 = b.create<mod_arith::DoubleOp>(x1);
  auto x1Times3 = b.create<mod_arith::MulOp>(x1, const3);
  auto x2Times2 = b.create<mod_arith::DoubleOp>(x2);
  auto x2Times4 = b.create<mod_arith::DoubleOp>(x2Times2);
  auto x2Times9 = b.create<mod_arith::MulOp>(x2, const9);
  auto x3Times2 = b.create<mod_arith::DoubleOp>(x3);
  auto x3Times4 = b.create<mod_arith::DoubleOp>(x3Times2);
  auto x3Times8 = b.create<mod_arith::DoubleOp>(x3Times4);
  auto x3Times27 = b.create<mod_arith::MulOp>(x3, const27);

  // Compute multiples of y coefficients
  auto y1Times2 = b.create<mod_arith::DoubleOp>(y1);
  auto y1Times3 = b.create<mod_arith::MulOp>(y1, const3);
  auto y2Times2 = b.create<mod_arith::DoubleOp>(y2);
  auto y2Times4 = b.create<mod_arith::DoubleOp>(y2Times2);
  auto y2Times9 = b.create<mod_arith::MulOp>(y2, const9);
  auto y3Times2 = b.create<mod_arith::DoubleOp>(y3);
  auto y3Times4 = b.create<mod_arith::DoubleOp>(y3Times2);
  auto y3Times8 = b.create<mod_arith::DoubleOp>(y3Times4);
  auto y3Times27 = b.create<mod_arith::MulOp>(y3, const27);

  // v₀ = x₀ * y₀
  auto v0 = b.create<mod_arith::MulOp>(x0, y0);

  // v₁ = (x₀ + x₁ + x₂ + x₃)(y₀ + y₁ + y₂ + y₃)
  Value xSum1 = b.create<mod_arith::AddOp>(x0, x1);
  xSum1 = b.create<mod_arith::AddOp>(xSum1, x2);
  xSum1 = b.create<mod_arith::AddOp>(xSum1, x3);
  Value ySum1 = b.create<mod_arith::AddOp>(y0, y1);
  ySum1 = b.create<mod_arith::AddOp>(ySum1, y2);
  ySum1 = b.create<mod_arith::AddOp>(ySum1, y3);
  auto v1 = b.create<mod_arith::MulOp>(xSum1, ySum1);

  // v₂ = (x₀ - x₁ + x₂ - x₃)(y₀ - y₁ + y₂ - y₃)
  Value xDiff1 = b.create<mod_arith::SubOp>(x0, x1);
  xDiff1 = b.create<mod_arith::AddOp>(xDiff1, x2);
  xDiff1 = b.create<mod_arith::SubOp>(xDiff1, x3);
  Value yDiff1 = b.create<mod_arith::SubOp>(y0, y1);
  yDiff1 = b.create<mod_arith::AddOp>(yDiff1, y2);
  yDiff1 = b.create<mod_arith::SubOp>(yDiff1, y3);
  auto v2 = b.create<mod_arith::MulOp>(xDiff1, yDiff1);

  // v₃ = (x₀ + 2x₁ + 4x₂ + 8x₃)(y₀ + 2y₁ + 4y₂ + 8y₃)
  Value xSum2 = b.create<mod_arith::AddOp>(x0, x1Times2);
  xSum2 = b.create<mod_arith::AddOp>(xSum2, x2Times4);
  xSum2 = b.create<mod_arith::AddOp>(xSum2, x3Times8);
  Value ySum2 = b.create<mod_arith::AddOp>(y0, y1Times2);
  ySum2 = b.create<mod_arith::AddOp>(ySum2, y2Times4);
  ySum2 = b.create<mod_arith::AddOp>(ySum2, y3Times8);
  auto v3 = b.create<mod_arith::MulOp>(xSum2, ySum2);

  // v₄ = (x₀ - 2x₁ + 4x₂ - 8x₃)(y₀ - 2y₁ + 4y₂ - 8y₃)
  Value xDiff2 = b.create<mod_arith::SubOp>(x0, x1Times2);
  xDiff2 = b.create<mod_arith::AddOp>(xDiff2, x2Times4);
  xDiff2 = b.create<mod_arith::SubOp>(xDiff2, x3Times8);
  Value yDiff2 = b.create<mod_arith::SubOp>(y0, y1Times2);
  yDiff2 = b.create<mod_arith::AddOp>(yDiff2, y2Times4);
  yDiff2 = b.create<mod_arith::SubOp>(yDiff2, y3Times8);
  auto v4 = b.create<mod_arith::MulOp>(xDiff2, yDiff2);

  // v₅ = (x₀ + 3x₁ + 9x₂ + 27x₃)(y₀ + 3y₁ + 9y₂ + 27y₃)
  Value xSum3 = b.create<mod_arith::AddOp>(x0, x1Times3);
  xSum3 = b.create<mod_arith::AddOp>(xSum3, x2Times9);
  xSum3 = b.create<mod_arith::AddOp>(xSum3, x3Times27);
  Value ySum3 = b.create<mod_arith::AddOp>(y0, y1Times3);
  ySum3 = b.create<mod_arith::AddOp>(ySum3, y2Times9);
  ySum3 = b.create<mod_arith::AddOp>(ySum3, y3Times27);
  auto v5 = b.create<mod_arith::MulOp>(xSum3, ySum3);

  // v₆ = x₃ * y₃
  auto v6 = b.create<mod_arith::MulOp>(x3, y3);

  return toomCookInterpolate(v0, v1, v2, v3, v4, v5, v6);
}

Value QuarticExtensionField::frobeniusMap(Value x, const APInt &exponent) {
  // Frobenius map:
  //   φᵉ(x₀ + x₁v + x₂v² + x₃v³)
  //     = x₀ + x₁ * v^(pᵉ) + x₂ * v^(2 * pᵉ) + x₃ * v^(3 * pᵉ)
  // v^(k * pᵉ) = ξ^((k * pᵉ) / 4) * v^((k * pᵉ) % 4) (for k = [1, 3])

  auto baseField = cast<PrimeFieldType>(type.getBaseFieldType());
  APInt modulus = baseField.getModulus().getValue();
  unsigned bitWidth = modulus.getBitWidth();
  auto convertedType = converter->convertType(baseField);

  // Get non-residue ξ
  auto nonResidueAttr = cast<IntegerAttr>(type.getNonResidue());
  APInt xi = nonResidueAttr.getValue();

  // Compute pᵉ mod (p - 1) using Fermat's little theorem
  // We need enough bits to hold the intermediate pᵉ computation
  // exponent can be large (e.g., p³), so we compute pᵉ mod 4 and pᵉ mod (p - 1)
  APInt pMinus1 = modulus - 1;

  // Compute pᵉ mod 4 and pᵉ mod (p - 1)
  // p mod 4 and p mod (p - 1) = 1
  // pᵉ mod 4 = (p mod 4)ᵉ mod 4
  // pᵉ mod (p - 1) = 1ᵉ = 1 (by Fermat's little theorem)
  APInt pMod4 = modulus.urem(APInt(bitWidth, 4));
  APInt pEMod4 = expMod(pMod4, exponent, APInt(bitWidth, 4));

  // Create zero constant for initialization
  APInt zeroVal(bitWidth, 0);
  auto zeroConst = b.create<mod_arith::ConstantOp>(
      convertedType, IntegerAttr::get(baseField.getStorageType(), zeroVal));

  auto coeffs = toCoeffs(b, x);

  // Initialize result coefficients to zero
  SmallVector<Value, 4> result = {zeroConst, zeroConst, zeroConst, zeroConst};

  // For each input coefficient k, compute coefficient and position
  for (unsigned k = 0; k < 4; ++k) {
    // k * pᵉ mod 4 determines the output position
    APInt kPEMod4 = (APInt(bitWidth, k) * pEMod4).urem(APInt(bitWidth, 4));
    unsigned newPos = kPEMod4.getZExtValue();

    // Compute ξ^((k * pᵉ) / 4) mod p
    // Since pᵉ ≡ 1 (mod p - 1), we have ξ^(k * pᵉ / 4) = ξ^(k / 4) when
    // p ≡ 1 (mod 4). For general case, we compute more carefully.
    //
    // k * pᵉ / 4 is the integer division, which determines the exponent of ξ.
    // We use: ξ^exp where exp = (k * pᵉ) / 4 mod (p - 1)
    //
    // For k = 0: exp = 0, coeff = 1
    // For k > 0: We need to compute (k * pᵉ) / 4 mod (p - 1)
    APInt coeff;
    if (k == 0) {
      coeff = APInt(bitWidth, 1);
    } else {
      // Compute k * pᵉ with sufficient precision
      // We only need (k * pᵉ) mod 4 * (p - 1) to get both the quotient and
      // remainder
      unsigned mulBits = bitWidth * 2;
      APInt kVal(mulBits, k);

      // Compute pᵉ mod 4 * (p - 1)
      APInt fourPMinus1 = APInt(mulBits, 4) * pMinus1.zext(mulBits);
      APInt pMod = modulus.zext(mulBits);
      APInt pEMod = expMod(pMod, exponent.zext(mulBits), fourPMinus1);

      APInt kPE = kVal * pEMod;
      APInt four(mulBits, 4);
      APInt xiExp = kPE.udiv(four).urem(pMinus1.zext(mulBits));
      coeff = expMod(xi, xiExp.zextOrTrunc(bitWidth), modulus);
    }

    auto coeffConst = b.create<mod_arith::ConstantOp>(
        convertedType, IntegerAttr::get(baseField.getStorageType(), coeff));
    auto scaled = b.create<mod_arith::MulOp>(coeffs[k], coeffConst);
    result[newPos] = b.create<mod_arith::AddOp>(result[newPos], scaled);
  }

  return fromCoeffs(b, type, result);
}

Value QuarticExtensionField::inverse(Value x) {
  // Frobenius-based inverse for quartic extension fields.
  // See Tachyon's quartic_extension_field.h for reference.
  // https://github.com/kroma-network/tachyon/blob/e7b1306/tachyon/math/finite_fields/quartic_extension_field.h#L565-L596
  // Algorithm 11.3.4 in "Handbook of Elliptic and Hyperelliptic Curve
  // Cryptography"
  // http://pustaka.unp.ac.id/file/abstrak_kki/EBOOKS/Kriptografi%20dan%20Ethical%20Hacking%20B.pdf
  //
  // For Fp4, we use the formula:
  //   x⁻¹ = x^(r - 1) * (xʳ)⁻¹
  // where r = p³ + p² + p + 1 = (p⁴ - 1) / (p - 1)
  //
  // xʳ = x * φ(x) * φ²(x) * φ³(x) is the norm and lies in Fp.
  // x^(r - 1) = φ(x) * φ²(x) * φ³(x)

  auto baseField = cast<PrimeFieldType>(type.getBaseFieldType());
  unsigned bitWidth = baseField.getModulus().getValue().getBitWidth();

  // φ(x)
  auto phi1 = frobeniusMap(x, APInt(bitWidth, 1));

  // φ²(x)
  auto phi2 = frobeniusMap(x, APInt(bitWidth, 2));

  // φ³(x)
  auto phi3 = frobeniusMap(x, APInt(bitWidth, 3));

  // xʳ⁻¹ = φ(x) * φ²(x) * φ³(x)
  auto xRMinus1 = mul(phi1, phi2);
  xRMinus1 = mul(xRMinus1, phi3);

  auto xCoeffs = toCoeffs(b, x);
  auto tCoeffs = toCoeffs(b, xRMinus1);

  // x₀ * t₀
  auto x0t0 = b.create<mod_arith::MulOp>(xCoeffs[0], tCoeffs[0]);
  // x₁ * t₃
  auto x1t3 = b.create<mod_arith::MulOp>(xCoeffs[1], tCoeffs[3]);
  // x₂ * t₂
  auto x2t2 = b.create<mod_arith::MulOp>(xCoeffs[2], tCoeffs[2]);
  // x₃ * t₁
  auto x3t1 = b.create<mod_arith::MulOp>(xCoeffs[3], tCoeffs[1]);

  // x₁ * t₃ + x₂ * t₂ + x₃ * t₁
  Value crossSum = b.create<mod_arith::AddOp>(x1t3, x2t2);
  crossSum = b.create<mod_arith::AddOp>(crossSum, x3t1);

  // ξ * (x₁ * t₃ + x₂ * t₂ + x₃ * t₁)
  auto xiCross = b.create<mod_arith::MulOp>(nonResidue, crossSum);

  // (xʳ)₀ = x₀ * t₀ + ξ * (...)
  auto xR0 = b.create<mod_arith::AddOp>(x0t0, xiCross);
  auto xR0Inv = b.create<mod_arith::InverseOp>(xR0);

  // x⁻¹ = xʳ⁻¹ * (xʳ)⁻¹
  auto z0 = b.create<mod_arith::MulOp>(tCoeffs[0], xR0Inv);
  auto z1 = b.create<mod_arith::MulOp>(tCoeffs[1], xR0Inv);
  auto z2 = b.create<mod_arith::MulOp>(tCoeffs[2], xR0Inv);
  auto z3 = b.create<mod_arith::MulOp>(tCoeffs[3], xR0Inv);

  return fromCoeffs(b, type, {z0, z1, z2, z3});
}

} // namespace mlir::zkir::field
