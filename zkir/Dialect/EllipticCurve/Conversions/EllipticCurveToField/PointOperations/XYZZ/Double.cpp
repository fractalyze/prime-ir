#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/XYZZ/Double.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::zkir::elliptic_curve {

// dbl-2008-s-1
// https://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-dbl-2008-s-1
// Cost: 6M + 4S + 1*a
Value xyzzDouble(const Value &point, Type inputType, ImplicitLocOpBuilder &b) {
  if (!isa<XYZZType>(inputType)) {
    assert(false && "Unsupported point types for XYZZ doubling");
  }

  Value zero = b.create<arith::ConstantIndexOp>(0);
  Value one = b.create<arith::ConstantIndexOp>(1);
  Value two = b.create<arith::ConstantIndexOp>(2);
  Value three = b.create<arith::ConstantIndexOp>(3);

  auto x = b.create<tensor::ExtractOp>(point, zero);
  auto y = b.create<tensor::ExtractOp>(point, one);
  auto zz = b.create<tensor::ExtractOp>(point, two);
  auto zzz = b.create<tensor::ExtractOp>(point, three);

  field::PrimeFieldType basefield = cast<field::PrimeFieldType>(x.getType());
  field::PrimeFieldAttr aAttr;

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
  aAttr = cast<XYZZType>(inputType).getCurve().getA();
  auto a = b.create<field::ConstantOp>(basefield, aAttr.getValue());
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

  return b.create<tensor::FromElementsOp>(
      SmallVector<Value>({x3, y3, zz3, zzz3}));
}

}  // namespace mlir::zkir::elliptic_curve
