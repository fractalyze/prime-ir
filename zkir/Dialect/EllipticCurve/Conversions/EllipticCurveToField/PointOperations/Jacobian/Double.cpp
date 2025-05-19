#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/Jacobian/Double.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
static Value affineToJacobianDouble(const Value &point, const Value &a,
                                    ImplicitLocOpBuilder &b) {
  Value zero = b.create<arith::ConstantIndexOp>(0);
  Value one = b.create<arith::ConstantIndexOp>(1);

  auto x = b.create<tensor::ExtractOp>(point, zero);
  auto y = b.create<tensor::ExtractOp>(point, one);

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

  return b.create<tensor::FromElementsOp>(SmallVector<Value>({x3, y3, z3}));
}

// dbl-2009-l
// https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
// Cost: 2M + 5S
static Value jacobianA0Double(const Value &point, ImplicitLocOpBuilder &b) {
  Value zero = b.create<arith::ConstantIndexOp>(0);
  Value one = b.create<arith::ConstantIndexOp>(1);
  Value two = b.create<arith::ConstantIndexOp>(2);

  auto x = b.create<tensor::ExtractOp>(point, zero);
  auto y = b.create<tensor::ExtractOp>(point, one);
  auto z = b.create<tensor::ExtractOp>(point, two);

  // A = X²
  auto a = b.create<field::SquareOp>(x);
  // B = Y²
  auto yy = b.create<field::SquareOp>(y);
  // C = B²
  auto c = b.create<field::SquareOp>(yy);
  // D = 2*((X+B)²-A-C)
  auto dTmp1 = b.create<field::AddOp>(x, yy);
  auto dTmp2 = b.create<field::SquareOp>(dTmp1);
  auto dTmp3 = b.create<field::SubOp>(dTmp2, a);
  auto dTmp4 = b.create<field::SubOp>(dTmp3, c);
  auto d = b.create<field::DoubleOp>(dTmp4);
  // E = 3*A
  auto eTmp1 = b.create<field::DoubleOp>(a);
  auto e = b.create<field::AddOp>(eTmp1, a);
  // F = E²
  auto f = b.create<field::SquareOp>(e);
  // X3 = F-2*D
  auto x3Tmp1 = b.create<field::DoubleOp>(d);
  auto x3 = b.create<field::SubOp>(f, x3Tmp1);
  // Y3 = E*(D-X3)-8*C
  auto y3Tmp1 = b.create<field::SubOp>(d, x3);
  auto y3Tmp2 = b.create<field::MulOp>(e, y3Tmp1);
  auto y3Tmp3 = b.create<field::DoubleOp>(c);
  auto y3Tmp4 = b.create<field::DoubleOp>(y3Tmp3);
  auto y3Tmp5 = b.create<field::DoubleOp>(y3Tmp4);
  auto y3 = b.create<field::SubOp>(y3Tmp2, y3Tmp5);
  // Z3 = 2*Y*Z
  auto z3Tmp1 = b.create<field::DoubleOp>(y);
  auto z3 = b.create<field::MulOp>(z3Tmp1, z);

  return b.create<tensor::FromElementsOp>(SmallVector<Value>({x3, y3, z3}));
}

// dbl-2007-bl
// https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html#doubling-dbl-2007-bl
// Cost: 1M + 8S + 1*a
static Value jacobianDefaultDouble(const Value &point, const Value &a,
                                   ImplicitLocOpBuilder &b) {
  Value zero = b.create<arith::ConstantIndexOp>(0);
  Value one = b.create<arith::ConstantIndexOp>(1);
  Value two = b.create<arith::ConstantIndexOp>(2);

  auto x = b.create<tensor::ExtractOp>(point, zero);
  auto y = b.create<tensor::ExtractOp>(point, one);
  auto z = b.create<tensor::ExtractOp>(point, two);

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

  return b.create<tensor::FromElementsOp>(SmallVector<Value>({x3, y3, z3}));
}

Value jacobianDouble(const Value &point, Type inputType,
                     ImplicitLocOpBuilder &b) {
  auto baseField = cast<field::PrimeFieldType>(
      cast<RankedTensorType>(point.getType()).getElementType());
  field::PrimeFieldAttr aAttr;

  if (auto affineType = dyn_cast<AffineType>(inputType)) {
    aAttr = affineType.getCurve().getA();
    auto a = b.create<field::ConstantOp>(baseField, aAttr.getValue());
    return affineToJacobianDouble(point, a, b);
  } else if (auto jacobianType = dyn_cast<JacobianType>(inputType)) {
    aAttr = jacobianType.getCurve().getA();
    auto a = b.create<field::ConstantOp>(baseField, aAttr.getValue());
    auto zero = b.create<field::ConstantOp>(baseField, 0);

    auto cmpEq = b.create<field::CmpOp>(arith::CmpIPredicate::eq, a, zero);
    auto ifOp = b.create<scf::IfOp>(
        cmpEq,
        /*thenBuilder=*/
        [&](OpBuilder &builder, Location loc) {
          ImplicitLocOpBuilder b(loc, builder);
          b.create<scf::YieldOp>(jacobianA0Double(point, b));
        },
        /*elseBuilder=*/
        [&](OpBuilder &builder, Location loc) {
          ImplicitLocOpBuilder b(loc, builder);
          b.create<scf::YieldOp>(jacobianDefaultDouble(point, a, b));
        });
    return ifOp.getResult(0);
  } else {
    assert(false && "Unsupported point type for Jacobian doubling");
  }
}

}  // namespace mlir::zkir::elliptic_curve
