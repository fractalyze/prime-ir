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

#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/Jacobian/Double.h"

#include "zkir/Dialect/Field/IR/FieldOps.h"

namespace mlir::zkir::elliptic_curve {
namespace {

// mdbl-2007-bl
// https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html#doubling-mdbl-2007-bl
// Cost: 1M + 5S
// Assumption: Z == 1
SmallVector<Value> affineToJacobian(ValueRange point, Value a,
                                    ImplicitLocOpBuilder &b) {
  auto x = point[0];
  auto y = point[1];

  // XX = X²
  auto xx = b.create<field::SquareOp>(x);
  // YY = Y²
  auto yy = b.create<field::SquareOp>(y);
  // YYYY = YY²
  auto yyyy = b.create<field::SquareOp>(yy);
  // S = 2*((X+YY)²-XX-YYYY)
  auto sTmp1 = b.create<field::AddOp>(x, yy);
  auto sTmp2 = b.create<field::SquareOp>(sTmp1);
  auto sTmp3 = b.create<field::SubOp>(sTmp2, xx);
  auto sTmp4 = b.create<field::SubOp>(sTmp3, yyyy);
  auto s = b.create<field::DoubleOp>(sTmp4);
  // M = 3*XX+a
  auto mTmp1 = b.create<field::DoubleOp>(xx);
  auto mTmp2 = b.create<field::AddOp>(mTmp1, xx);
  auto m = b.create<field::AddOp>(mTmp2, a);
  // X3 = M²-2*S
  auto x3Tmp1 = b.create<field::SquareOp>(m);
  auto x3Tmp2 = b.create<field::DoubleOp>(s);
  auto x3 = b.create<field::SubOp>(x3Tmp1, x3Tmp2);
  // Y3 = M*(S-T)-8*YYYY
  auto y3Tmp1 = b.create<field::SubOp>(s, x3);
  auto y3Tmp2 = b.create<field::MulOp>(m, y3Tmp1);
  auto x3Tmp3 = b.create<field::DoubleOp>(yyyy);
  auto x3Tmp4 = b.create<field::DoubleOp>(x3Tmp3);
  auto x3Tmp5 = b.create<field::DoubleOp>(x3Tmp4);
  auto y3 = b.create<field::SubOp>(y3Tmp2, x3Tmp5);
  // Z3 = 2*Y
  auto z3 = b.create<field::DoubleOp>(y);

  return {x3, y3, z3};
}

// When a == 0, canonicalize using:
// dbl-2009-l
// https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
// Cost: 2M + 5S
//
// Otherwise, apply:
// dbl-2007-bl
// https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html#doubling-dbl-2007-bl
// Cost: 1M + 8S + 1*a
SmallVector<Value> jacobianToJacobian(ValueRange point, Value a,
                                      ImplicitLocOpBuilder &b) {
  Value x = point[0];
  Value y = point[1];
  Value z = point[2];

  // XX = X²
  auto xx = b.create<field::SquareOp>(x);
  // YY = Y²
  auto yy = b.create<field::SquareOp>(y);
  // YYYY = YY²
  auto yyyy = b.create<field::SquareOp>(yy);
  // ZZ = Z²
  auto zz = b.create<field::SquareOp>(z);
  // S = 2*((X+YY)²-XX-YYYY)
  auto sTmp1 = b.create<field::AddOp>(x, yy);
  auto sTmp2 = b.create<field::SquareOp>(sTmp1);
  auto sTmp3 = b.create<field::SubOp>(sTmp2, xx);
  auto sTmp4 = b.create<field::SubOp>(sTmp3, yyyy);
  auto s = b.create<field::DoubleOp>(sTmp4);
  // M = 3*XX+a*ZZ²
  auto mTmp1 = b.create<field::DoubleOp>(xx);
  auto mTmp2 = b.create<field::AddOp>(mTmp1, xx);
  auto mTmp3 = b.create<field::SquareOp>(zz);
  auto mTmp4 = b.create<field::MulOp>(a, mTmp3);
  auto m = b.create<field::AddOp>(mTmp2, mTmp4);
  // X3 = M²-2*S
  auto x3Tmp1 = b.create<field::SquareOp>(m);
  auto x3Tmp2 = b.create<field::DoubleOp>(s);
  auto x3 = b.create<field::SubOp>(x3Tmp1, x3Tmp2);
  // Y3 = M*(S-X3)-8*YYYY
  auto y3Tmp1 = b.create<field::SubOp>(s, x3);
  auto y3Tmp2 = b.create<field::MulOp>(m, y3Tmp1);
  auto y3Tmp3 = b.create<field::DoubleOp>(yyyy);
  auto y3Tmp4 = b.create<field::DoubleOp>(y3Tmp3);
  auto y3Tmp5 = b.create<field::DoubleOp>(y3Tmp4);
  auto y3 = b.create<field::SubOp>(y3Tmp2, y3Tmp5);
  // Z3 = (Y+Z)²-YY-ZZ
  auto z3Tmp1 = b.create<field::AddOp>(y, z);
  auto z3Tmp2 = b.create<field::SquareOp>(z3Tmp1);
  auto z3Tmp3 = b.create<field::SubOp>(z3Tmp2, yy);
  auto z3 = b.create<field::SubOp>(z3Tmp3, zz);

  return {x3, y3, z3};
}

} // namespace

SmallVector<Value> jacobianDouble(ValueRange point, ShortWeierstrassAttr curve,
                                  ImplicitLocOpBuilder &b) {
  auto a = b.create<field::ConstantOp>(curve.getBaseField(), curve.getA());
  if (point.size() == 2) {
    return affineToJacobian(point, a, b);
  } else if (point.size() == 3) {
    return jacobianToJacobian(point, a, b);
  } else {
    llvm_unreachable("Unsupported point type for jacobian doubling");
    return {};
  }
}

} // namespace mlir::zkir::elliptic_curve
