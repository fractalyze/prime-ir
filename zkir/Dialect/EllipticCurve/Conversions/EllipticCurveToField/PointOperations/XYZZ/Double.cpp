#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/XYZZ/Double.h"

#include "zkir/Dialect/Field/IR/FieldOps.h"

namespace mlir::zkir::elliptic_curve {
namespace {

// mdbl-2008-s-1
// https://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-mdbl-2008-s-1
// Cost: 4M + 3S
SmallVector<Value> affineToXYZZ(ValueRange point, ShortWeierstrassAttr curve,
                                ImplicitLocOpBuilder &b) {
  Value x = point[0];
  Value y = point[1];

  // U = 2*Y
  auto u = b.create<field::DoubleOp>(y);
  // V = U²
  auto v = b.create<field::SquareOp>(u);
  // W = U*V
  auto w = b.create<field::MulOp>(u, v);
  // S = X*V
  auto s = b.create<field::MulOp>(x, v);
  // M = 3*X²+a
  auto mTmp1 = b.create<field::SquareOp>(x);
  auto mTmp2 = b.create<field::DoubleOp>(mTmp1);
  auto mTmp3 = b.create<field::AddOp>(mTmp2, mTmp1);
  auto a = b.create<field::ConstantOp>(curve.getBaseField(), curve.getA());
  auto m = b.create<field::AddOp>(mTmp3, a);
  // X3 = M²-2*S
  auto x3Tmp1 = b.create<field::SquareOp>(m);
  auto x3Tmp2 = b.create<field::DoubleOp>(s);
  auto x3 = b.create<field::SubOp>(x3Tmp1, x3Tmp2);
  // Y3 = M*(S-X3)-W*Y
  auto y3Tmp1 = b.create<field::SubOp>(s, x3);
  auto y3Tmp2 = b.create<field::MulOp>(m, y3Tmp1);
  auto y3Tmp3 = b.create<field::MulOp>(w, y);
  auto y3 = b.create<field::SubOp>(y3Tmp2, y3Tmp3);
  // ZZ3 = V
  // ZZZ3 = W

  return {x3, y3, v, w};
}

// dbl-2008-s-1
// https://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-dbl-2008-s-1
// Cost: 6M + 4S + 1*a
SmallVector<Value> xyzzToXyzz(ValueRange point, ShortWeierstrassAttr curve,
                              ImplicitLocOpBuilder &b) {
  Value x = point[0];
  Value y = point[1];
  Value zz = point[2];
  Value zzz = point[3];

  // U = 2*Y
  auto u = b.create<field::DoubleOp>(y);
  // V = U²
  auto v = b.create<field::SquareOp>(u);
  // W = U*V
  auto w = b.create<field::MulOp>(u, v);
  // S = X*V
  auto s = b.create<field::MulOp>(x, v);
  // M = 3*X²+a*ZZ²
  auto mTmp1 = b.create<field::SquareOp>(x);
  auto mTmp2 = b.create<field::DoubleOp>(mTmp1);
  auto mTmp3 = b.create<field::AddOp>(mTmp2, mTmp1);
  auto mTmp4 = b.create<field::SquareOp>(zz);
  auto a = b.create<field::ConstantOp>(curve.getBaseField(), curve.getA());
  auto mTmp5 = b.create<field::MulOp>(a, mTmp4);
  auto m = b.create<field::AddOp>(mTmp3, mTmp5);
  // X3 = M²-2*S
  auto x3Tmp1 = b.create<field::SquareOp>(m);
  auto x3Tmp2 = b.create<field::DoubleOp>(s);
  auto x3 = b.create<field::SubOp>(x3Tmp1, x3Tmp2);
  // Y3 = M*(S-X3)-W*Y
  auto y3Tmp1 = b.create<field::SubOp>(s, x3);
  auto y3Tmp2 = b.create<field::MulOp>(m, y3Tmp1);
  auto y3Tmp3 = b.create<field::MulOp>(w, y);
  auto y3 = b.create<field::SubOp>(y3Tmp2, y3Tmp3);
  // ZZ3 = V*ZZ
  auto zz3 = b.create<field::MulOp>(v, zz);
  // ZZZ3 = W*ZZZ
  auto zzz3 = b.create<field::MulOp>(w, zzz);

  return {x3, y3, zz3, zzz3};
}

} // namespace

SmallVector<Value> xyzzDouble(ValueRange point, ShortWeierstrassAttr curve,
                              ImplicitLocOpBuilder &b) {
  if (point.size() == 2) {
    return affineToXYZZ(point, curve, b);
  } else if (point.size() == 4) {
    return xyzzToXyzz(point, curve, b);
  } else {
    llvm_unreachable("Unsupported point type for xyzz doubling");
    return {};
  }
}

} // namespace mlir::zkir::elliptic_curve
