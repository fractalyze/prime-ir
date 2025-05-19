#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/Jacobian/Double.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::zkir::elliptic_curve {

// mdbl-2007-bl
// https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html#doubling-mdbl-2007-bl
// Cost: 1M + 5S
// Assumption: Z == 1
Value affineToJacobianDouble(const Value &point, Type affineType,
                             ImplicitLocOpBuilder &b) {
  Value zero = b.create<arith::ConstantIndexOp>(0);
  Value one = b.create<arith::ConstantIndexOp>(1);

  auto x = b.create<tensor::ExtractOp>(point, zero);
  auto y = b.create<tensor::ExtractOp>(point, one);

  field::PrimeFieldType basefield = cast<field::PrimeFieldType>(x.getType());
  field::PrimeFieldAttr aAttr;

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
  aAttr = cast<AffineType>(affineType).getCurve().getA();
  auto a = b.create<field::ConstantOp>(basefield, aAttr.getValue());
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

  return b.create<tensor::FromElementsOp>(SmallVector<Value>({x3, y3, z3}));
}

// dbl-2007-bl
// https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html#doubling-dbl-2007-bl
// Cost: 1M + 8S + 1*a
Value jacobianDefaultDouble(const Value &point, Type jacobianType,
                            ImplicitLocOpBuilder &b) {
  Value zero = b.create<arith::ConstantIndexOp>(0);
  Value one = b.create<arith::ConstantIndexOp>(1);
  Value two = b.create<arith::ConstantIndexOp>(2);

  auto x = b.create<tensor::ExtractOp>(point, zero);
  auto y = b.create<tensor::ExtractOp>(point, one);
  auto z = b.create<tensor::ExtractOp>(point, two);

  field::PrimeFieldType basefield = cast<field::PrimeFieldType>(x.getType());
  field::PrimeFieldAttr aAttr;

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
  aAttr = cast<JacobianType>(jacobianType).getCurve().getA();
  auto a = b.create<field::ConstantOp>(basefield, aAttr.getValue());
  auto mTmp4 = b.create<field::MulOp>(a, mTmp3);
  auto m = b.create<field::AddOp>(mTmp2, mTmp4);
  // X3 = M²-2*S
  auto mm = b.create<field::SquareOp>(m);
  auto two_s = b.create<field::DoubleOp>(s);
  auto x3 = b.create<field::SubOp>(mm, two_s);
  // Y3 = M*(S-X3)-8*YYYY
  auto y3Tmp1 = b.create<field::SubOp>(s, x3);
  auto y3Tmp2 = b.create<field::MulOp>(m, y3Tmp1);
  auto x3Tmp3 = b.create<field::DoubleOp>(yyyy);
  auto x3Tmp4 = b.create<field::DoubleOp>(x3Tmp3);
  auto x3Tmp5 = b.create<field::DoubleOp>(x3Tmp4);
  auto y3 = b.create<field::SubOp>(y3Tmp2, x3Tmp5);
  // Z3 = (Y+Z)²-YY-ZZ
  auto z3Tmp1 = b.create<field::AddOp>(y, z);
  auto z3Tmp2 = b.create<field::SquareOp>(z3Tmp1);
  auto z3Tmp3 = b.create<field::SubOp>(z3Tmp2, yy);
  auto z3 = b.create<field::SubOp>(z3Tmp3, zz);

  return b.create<tensor::FromElementsOp>(SmallVector<Value>({x3, y3, z3}));
}

Value jacobianDouble(const Value &point, Type inputType,
                     ImplicitLocOpBuilder &b) {
  if (isa<AffineType>(inputType)) {
    return affineToJacobianDouble(point, inputType, b);
  } else if (isa<JacobianType>(inputType)) {
    return jacobianDefaultDouble(point, inputType, b);
  } else {
    assert(false && "Unsupported point type for Jacobian doubling");
  }
}

}  // namespace mlir::zkir::elliptic_curve
