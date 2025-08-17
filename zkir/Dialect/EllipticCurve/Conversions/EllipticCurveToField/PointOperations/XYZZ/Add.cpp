#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/XYZZ/Add.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/XYZZ/Double.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"

namespace mlir::zkir::elliptic_curve {
namespace {

// madd-2008-s
// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-mmadd-2008-s
// Cost: 4M + 2S
// Assumption: ZZ1 == ZZZ1 == ZZ2 == ZZZ2 == 1
SmallVector<Value> affineAndAffine(ValueRange p1, ValueRange p2,
                                   ShortWeierstrassAttr curve,
                                   ImplicitLocOpBuilder &b) {
  auto x1 = p1[0];
  auto y1 = p1[1];

  auto x2 = p2[0];
  auto y2 = p2[1];

  // if x1 == x2 && y1 == y2
  auto cmpEq1 = b.create<field::CmpOp>(arith::CmpIPredicate::eq, x1, x2);
  auto cmpEq2 = b.create<field::CmpOp>(arith::CmpIPredicate::eq, y1, y2);
  auto combined_condition = b.create<arith::AndIOp>(cmpEq1, cmpEq2);
  auto ifOp = b.create<scf::IfOp>(
      combined_condition,
      /*thenBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder b(loc, builder);
        b.create<scf::YieldOp>(xyzzDouble(p1, curve, b));
      },
      /*elseBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder b(loc, builder);
        // P = X2-X1
        auto p = b.create<field::SubOp>(x2, x1);
        // R = Y2-Y1
        auto r = b.create<field::SubOp>(y2, y1);
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

        b.create<scf::YieldOp>(ValueRange{x3, y3, pp, ppp});
      });
  return ifOp.getResults();
}

// madd-2008-s
// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-madd-2008-s
// Cost: 8M + 2S
// Assumption: ZZ2 == ZZZ2 == 1
SmallVector<Value> xyzzAndAffine(ValueRange p1, ValueRange p2,
                                 ShortWeierstrassAttr curve,
                                 ImplicitLocOpBuilder &b) {
  auto x1 = p1[0];
  auto y1 = p1[1];
  auto zz1 = p1[2];
  auto zzz1 = p1[3];

  auto x2 = p2[0];
  auto y2 = p2[1];

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
        b.create<scf::YieldOp>(xyzzDouble(p2, curve, b));
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

        b.create<scf::YieldOp>(ValueRange{x3, y3, zz3, zzz3});
      });
  return ifOp.getResults();
}

// add-2008-s
// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-add-2008-s
// Cost: 12M + 2S
SmallVector<Value> xyzzAndXyzz(ValueRange p1, ValueRange p2,
                               ShortWeierstrassAttr curve,
                               ImplicitLocOpBuilder &b) {
  auto x1 = p1[0];
  auto y1 = p1[1];
  auto zz1 = p1[2];
  auto zzz1 = p1[3];

  auto x2 = p2[0];
  auto y2 = p2[1];
  auto zz2 = p2[2];
  auto zzz2 = p2[3];

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
        b.create<scf::YieldOp>(xyzzDouble(p1, curve, b));
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
        b.create<scf::YieldOp>(ValueRange{x3, y3, zz3, zzz3});
      });
  return ifOp.getResults();
}

} // namespace

SmallVector<Value> xyzzAdd(ValueRange p1, ValueRange p2,
                           ShortWeierstrassAttr curve,
                           ImplicitLocOpBuilder &b) {
  if (p1.size() == 2) {
    if (p2.size() == 2) {
      return affineAndAffine(p2, p1, curve, b);
    }
    return xyzzAndAffine(p2, p1, curve, b);
  } else if (p2.size() == 2) {
    return xyzzAndAffine(p1, p2, curve, b);
  } else if (p1.size() == 4 && p2.size() == 4) {
    return xyzzAndXyzz(p1, p2, curve, b);
  } else {
    llvm_unreachable("Unsupported point types for xyzz addition");
    return {};
  }
}

} // namespace mlir::zkir::elliptic_curve
