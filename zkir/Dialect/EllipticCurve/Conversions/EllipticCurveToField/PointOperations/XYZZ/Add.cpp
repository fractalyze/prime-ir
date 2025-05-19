#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/XYZZ/Add.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/XYZZ/Double.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"

namespace mlir::zkir::elliptic_curve {

// madd-2008-s
// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-madd-2008-s
// Cost: 8M + 2S
// Assumption: ZZ2 == ZZZ2 == 1
static Value xyzzAndAffine(const Value &p1, const Value &p2, Type affineType,
                           ImplicitLocOpBuilder &b) {
  Value zero = b.create<arith::ConstantIndexOp>(0);
  Value one = b.create<arith::ConstantIndexOp>(1);
  Value two = b.create<arith::ConstantIndexOp>(2);
  Value three = b.create<arith::ConstantIndexOp>(3);

  auto x1 = b.create<tensor::ExtractOp>(p1, zero);
  auto y1 = b.create<tensor::ExtractOp>(p1, one);
  auto zz1 = b.create<tensor::ExtractOp>(p1, two);
  auto zzz1 = b.create<tensor::ExtractOp>(p1, three);

  auto x2 = b.create<tensor::ExtractOp>(p2, zero);
  auto y2 = b.create<tensor::ExtractOp>(p2, one);

  // U2 = X2*ZZ1
  auto u2 = b.create<field::MulOp>(x2, zz1);
  // S2 = Y2*ZZZ1
  auto s2 = b.create<field::MulOp>(y2, zzz1);

  // if x1 == u2 && y1 == s2
  auto cmpEq1 = b.create<field::CmpOp>(arith::CmpIPredicate::eq, x1, u2);
  auto cmpEq2 = b.create<field::CmpOp>(arith::CmpIPredicate::eq, y1, s2);
  auto combined_condition = b.create<arith::AndIOp>(cmpEq1, cmpEq2);
  auto ifOp = b.create<scf::IfOp>(
      combined_condition,
      /*thenBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder b(loc, builder);
        b.create<scf::YieldOp>(affineToXYZZDouble(p2, affineType, b));
      },
      /*elseBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder b(loc, builder);
        // P = U2-X1
        auto p = b.create<field::SubOp>(u2, x1);
        // R = S2-Y1
        auto r = b.create<field::SubOp>(s2, y1);
        // PP = P²
        auto pp = b.create<field::SquareOp>(p);
        // PPP = P*PP
        auto ppp = b.create<field::MulOp>(p, pp);
        // Q = X1*PP
        auto q = b.create<field::MulOp>(x1, pp);
        // X3 = R²-PPP-2*Q
        auto x3Tmp1 = b.create<field::SquareOp>(r);
        auto x3Tmp2 = b.create<field::SubOp>(x3Tmp1, ppp);
        auto x3Tmp3 = b.create<field::DoubleOp>(q);
        auto x3 = b.create<field::SubOp>(x3Tmp2, x3Tmp3);
        // Y3 = R*(Q-X3)-Y1*PPP
        auto y3Tmp1 = b.create<field::SubOp>(q, x3);
        auto y3Tmp2 = b.create<field::MulOp>(r, y3Tmp1);
        auto y3Tmp3 = b.create<field::MulOp>(y1, ppp);
        auto y3 = b.create<field::SubOp>(y3Tmp2, y3Tmp3);
        // ZZ3 = ZZ1*PP
        auto zz3 = b.create<field::MulOp>(zz1, pp);
        // ZZZ3 = ZZZ1*PPP
        auto zzz3 = b.create<field::MulOp>(zzz1, ppp);

        auto makePoint = b.create<tensor::FromElementsOp>(
            SmallVector<Value>({x3, y3, zz3, zzz3}));
        b.create<scf::YieldOp>(ValueRange{makePoint});
      });
  return ifOp.getResult(0);
}

// add-2008-s
// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-add-2008-s
// Cost: 12M + 2S
static Value xyzzAndXyzz(const Value &p1, const Value &p2, Type xyzzType,
                         ImplicitLocOpBuilder &b) {
  Value zero = b.create<arith::ConstantIndexOp>(0);
  Value one = b.create<arith::ConstantIndexOp>(1);
  Value two = b.create<arith::ConstantIndexOp>(2);
  Value three = b.create<arith::ConstantIndexOp>(3);

  auto x1 = b.create<tensor::ExtractOp>(p1, zero);
  auto y1 = b.create<tensor::ExtractOp>(p1, one);
  auto zz1 = b.create<tensor::ExtractOp>(p1, two);
  auto zzz1 = b.create<tensor::ExtractOp>(p1, three);

  auto x2 = b.create<tensor::ExtractOp>(p2, zero);
  auto y2 = b.create<tensor::ExtractOp>(p2, one);
  auto zz2 = b.create<tensor::ExtractOp>(p2, two);
  auto zzz2 = b.create<tensor::ExtractOp>(p2, three);

  // U1 = X1*ZZ2
  auto u1 = b.create<field::MulOp>(x1, zz2);
  // U2 = X2*ZZ1
  auto u2 = b.create<field::MulOp>(x2, zz1);
  // S1 = Y1*ZZZ2
  auto s1 = b.create<field::MulOp>(y1, zzz2);
  // S2 = Y2*ZZZ1
  auto s2 = b.create<field::MulOp>(y2, zzz1);

  // if u1 == u2 && s1 == s2
  auto cmpEq1 = b.create<field::CmpOp>(arith::CmpIPredicate::eq, u1, u2);
  auto cmpEq2 = b.create<field::CmpOp>(arith::CmpIPredicate::eq, s1, s2);
  auto combined_condition = b.create<arith::AndIOp>(cmpEq1, cmpEq2);
  auto ifOp = b.create<scf::IfOp>(
      combined_condition,
      /*thenBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder b(loc, builder);
        b.create<scf::YieldOp>(xyzzDouble(p1, xyzzType, b));
      },
      /*elseBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder b(loc, builder);
        // P = U2-U1
        auto p = b.create<field::SubOp>(u2, u1);
        // R = S2-S1
        auto r = b.create<field::SubOp>(s2, s1);
        // PP = P²
        auto pp = b.create<field::SquareOp>(p);
        // PPP = P*PP
        auto ppp = b.create<field::MulOp>(p, pp);
        // Q = U1*PP
        auto q = b.create<field::MulOp>(u1, pp);
        // X3 = R²-PPP-2*Q
        auto x3Tmp1 = b.create<field::SquareOp>(r);
        auto x3Tmp2 = b.create<field::SubOp>(x3Tmp1, ppp);
        auto x3Tmp3 = b.create<field::DoubleOp>(q);
        auto x3 = b.create<field::SubOp>(x3Tmp2, x3Tmp3);
        // Y3 = R*(Q-X3)-S1*PPP
        auto y3Tmp1 = b.create<field::SubOp>(q, x3);
        auto y3Tmp2 = b.create<field::MulOp>(r, y3Tmp1);
        auto y3Tmp3 = b.create<field::MulOp>(s1, ppp);
        auto y3 = b.create<field::SubOp>(y3Tmp2, y3Tmp3);
        // ZZ3 = ZZ1*ZZ2*PP
        auto zz3Tmp = b.create<field::MulOp>(zz1, zz2);
        auto zz3 = b.create<field::MulOp>(zz3Tmp, pp);
        // ZZZ3 = ZZZ1*ZZZ2*PPP
        auto zzz3Tmp = b.create<field::MulOp>(zzz1, zzz2);
        auto zzz3 = b.create<field::MulOp>(zzz3Tmp, ppp);
        auto makePoint = b.create<tensor::FromElementsOp>(
            SmallVector<Value>({x3, y3, zz3, zzz3}));
        b.create<scf::YieldOp>(ValueRange{makePoint});
      });
  return ifOp.getResult(0);
}

Value xyzzAdd(const Value &p1, const Value &p2, Type p1Type, Type p2Type,
              ImplicitLocOpBuilder &b) {
  if (isa<AffineType>(p1Type)) {
    return xyzzAndAffine(p2, p1, p1Type, b);
  } else if (isa<AffineType>(p2Type)) {
    return xyzzAndAffine(p1, p2, p2Type, b);
  } else if (isa<XYZZType>(p1Type) && isa<XYZZType>(p2Type)) {
    return xyzzAndXyzz(p1, p2, p1Type, b);
  } else {
    assert(false && "Unsupported point types for XYZZ addition");
  }
}

}  // namespace mlir::zkir::elliptic_curve
