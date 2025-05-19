#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/Jacobian/Add.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/Jacobian/Double.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"

namespace mlir::zkir::elliptic_curve {

// mmadd-2007-bl
// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-mmadd-2007-bl
// Cost: 4M + 2S
// Assumption: Z1 == Z2 == 1
static Value affineAndAffine(const Value &p1, const Value &p2, Type affineType,
                             ImplicitLocOpBuilder &b) {
  Value zero = b.create<arith::ConstantIndexOp>(0);
  Value one = b.create<arith::ConstantIndexOp>(1);

  auto x1 = b.create<tensor::ExtractOp>(p1, zero);
  auto y1 = b.create<tensor::ExtractOp>(p1, one);

  auto x2 = b.create<tensor::ExtractOp>(p2, zero);
  auto y2 = b.create<tensor::ExtractOp>(p2, one);

  // if x1 == x2 && y1 == y2
  auto cmpEq1 = b.create<field::CmpOp>(arith::CmpIPredicate::eq, x1, x2);
  auto cmpEq2 = b.create<field::CmpOp>(arith::CmpIPredicate::eq, y1, y2);
  auto combined_condition = b.create<arith::AndIOp>(cmpEq1, cmpEq2);
  auto ifOp = b.create<scf::IfOp>(
      combined_condition,
      /*thenBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder b(loc, builder);
        b.create<scf::YieldOp>(jacobianDouble(p1, affineType, b));
      },
      /*elseBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder b(loc, builder);
        // H = X2-X1
        auto h = b.create<field::SubOp>(x2, x1);
        // HH = H²
        auto hh = b.create<field::SquareOp>(h);
        // I = 4*HH
        auto iTmp = b.create<field::DoubleOp>(hh);
        auto i = b.create<field::DoubleOp>(iTmp);
        // J = H*I
        auto j = b.create<field::MulOp>(h, i);
        // r = 2*(Y2-Y1)
        auto rTmp = b.create<field::SubOp>(y2, y1);
        auto r = b.create<field::DoubleOp>(rTmp);
        // V = X1*I
        auto v = b.create<field::MulOp>(x1, i);
        // X3 = r²-J-2*V
        auto x3Tmp1 = b.create<field::SquareOp>(r);
        auto x3Tmp2 = b.create<field::DoubleOp>(v);
        auto x3Tmp3 = b.create<field::SubOp>(x3Tmp1, j);
        auto x3 = b.create<field::SubOp>(x3Tmp3, x3Tmp2);
        // Y3 = r*(V-X3)-2*Y1*J
        auto y3Tmp1 = b.create<field::SubOp>(v, x3);
        auto y3Tmp2 = b.create<field::MulOp>(r, y3Tmp1);
        auto y3Tmp3 = b.create<field::DoubleOp>(y1);
        auto y3Tmp4 = b.create<field::MulOp>(y3Tmp3, j);
        auto y3 = b.create<field::SubOp>(y3Tmp2, y3Tmp4);
        // Z3 = 2*H
        auto z3 = b.create<field::DoubleOp>(h);

        auto makePoint =
            b.create<tensor::FromElementsOp>(SmallVector<Value>({x3, y3, z3}));
        b.create<scf::YieldOp>(ValueRange{makePoint});
      });
  return ifOp.getResult(0);
}

// madd-2007-bl
// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
// Cost: 7M + 4S
// Assumption: Z2 == 1
static Value jacobianAndAffine(const Value &p1, const Value &p2,
                               Type affineType, ImplicitLocOpBuilder &b) {
  Value zero = b.create<arith::ConstantIndexOp>(0);
  Value one = b.create<arith::ConstantIndexOp>(1);
  Value two = b.create<arith::ConstantIndexOp>(2);

  auto x1 = b.create<tensor::ExtractOp>(p1, zero);
  auto y1 = b.create<tensor::ExtractOp>(p1, one);
  auto z1 = b.create<tensor::ExtractOp>(p1, two);

  auto x2 = b.create<tensor::ExtractOp>(p2, zero);
  auto y2 = b.create<tensor::ExtractOp>(p2, one);

  // Z1Z1 = Z1²
  auto z1z1 = b.create<field::SquareOp>(z1);
  // U2 = X2*Z1Z1
  auto u2 = b.create<field::MulOp>(x2, z1z1);
  // S2 = Y2*Z1*Z1Z1
  auto s2Tmp = b.create<field::MulOp>(y2, z1);
  auto s2 = b.create<field::MulOp>(s2Tmp, z1z1);

  // if x1 == u2 && y1 == s2
  auto cmpEq1 = b.create<field::CmpOp>(arith::CmpIPredicate::eq, x1, u2);
  auto cmpEq2 = b.create<field::CmpOp>(arith::CmpIPredicate::eq, y1, s2);
  auto combined_condition = b.create<arith::AndIOp>(cmpEq1, cmpEq2);
  auto ifOp = b.create<scf::IfOp>(
      combined_condition,
      /*thenBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder b(loc, builder);
        b.create<scf::YieldOp>(jacobianDouble(p2, affineType, b));
      },
      /*elseBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder b(loc, builder);
        // H = U2-X1
        auto h = b.create<field::SubOp>(u2, x1);
        // HH = H²
        auto hh = b.create<field::SquareOp>(h);
        // I = 4*HH
        auto iTmp = b.create<field::DoubleOp>(hh);
        auto i = b.create<field::DoubleOp>(iTmp);
        // J = H*I
        auto j = b.create<field::MulOp>(h, i);
        // r = 2*(S2-Y1)
        auto rTmp = b.create<field::SubOp>(s2, y1);
        auto r = b.create<field::DoubleOp>(rTmp);
        // V = X1*I
        auto v = b.create<field::MulOp>(x1, i);
        // X3 = r²-J-2*V
        auto x3Tmp1 = b.create<field::SquareOp>(r);
        auto x3Tmp2 = b.create<field::DoubleOp>(v);
        auto x3Tmp3 = b.create<field::SubOp>(x3Tmp1, j);
        auto x3 = b.create<field::SubOp>(x3Tmp3, x3Tmp2);
        // Y3 = r*(V-X3)-2*Y1*J
        auto y3Tmp1 = b.create<field::SubOp>(v, x3);
        auto y3Tmp2 = b.create<field::MulOp>(r, y3Tmp1);
        auto y3Tmp3 = b.create<field::DoubleOp>(y1);
        auto y3Tmp4 = b.create<field::MulOp>(y3Tmp3, j);
        auto y3 = b.create<field::SubOp>(y3Tmp2, y3Tmp4);
        // Z3 = (Z1+H)²-Z1Z1-HH
        auto z3Tmp1 = b.create<field::AddOp>(z1, h);
        auto z3Tmp2 = b.create<field::SquareOp>(z3Tmp1);
        auto z3Tmp3 = b.create<field::SubOp>(z3Tmp2, z1z1);
        auto z3 = b.create<field::SubOp>(z3Tmp3, hh);

        auto makePoint =
            b.create<tensor::FromElementsOp>(SmallVector<Value>({x3, y3, z3}));
        b.create<scf::YieldOp>(ValueRange{makePoint});
      });
  return ifOp.getResult(0);
}

// add-2007-bl
// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
// Cost: 11M + 5S
static Value jacobianAndJacobian(const Value &p1, const Value &p2,
                                 Type jacobianType, ImplicitLocOpBuilder &b) {
  Value zero = b.create<arith::ConstantIndexOp>(0);
  Value one = b.create<arith::ConstantIndexOp>(1);
  Value two = b.create<arith::ConstantIndexOp>(2);

  auto x1 = b.create<tensor::ExtractOp>(p1, zero);
  auto y1 = b.create<tensor::ExtractOp>(p1, one);
  auto z1 = b.create<tensor::ExtractOp>(p1, two);

  auto x2 = b.create<tensor::ExtractOp>(p2, zero);
  auto y2 = b.create<tensor::ExtractOp>(p2, one);
  auto z2 = b.create<tensor::ExtractOp>(p2, two);

  // Z1Z1 = Z1²
  auto z1z1 = b.create<field::SquareOp>(z1);
  // Z2Z2 = Z2²
  auto z2z2 = b.create<field::SquareOp>(z2);
  // U1 = X1*Z2Z2
  auto u1 = b.create<field::MulOp>(x1, z2z2);
  // U2 = X2*Z1Z1
  auto u2 = b.create<field::MulOp>(x2, z1z1);
  // S1 = Y1*Z2*Z2Z2
  auto s1Tmp = b.create<field::MulOp>(y1, z2);
  auto s1 = b.create<field::MulOp>(s1Tmp, z2z2);
  // S2 = Y2*Z1*Z1Z1
  auto s2Tmp = b.create<field::MulOp>(y2, z1);
  auto s2 = b.create<field::MulOp>(s2Tmp, z1z1);

  // if u1 == u2 && s1 == s2
  auto cmpEq1 = b.create<field::CmpOp>(arith::CmpIPredicate::eq, u1, u2);
  auto cmpEq2 = b.create<field::CmpOp>(arith::CmpIPredicate::eq, s1, s2);
  auto combined_condition = b.create<arith::AndIOp>(cmpEq1, cmpEq2);
  auto ifOp = b.create<scf::IfOp>(
      combined_condition,
      /*thenBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder b(loc, builder);
        b.create<scf::YieldOp>(jacobianDouble(p1, jacobianType, b));
      },
      /*elseBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder b(loc, builder);
        // H = U2-U1
        auto h = b.create<field::SubOp>(u2, u1);
        // I = (2*H)²
        auto iTmp = b.create<field::DoubleOp>(h);
        auto i = b.create<field::SquareOp>(iTmp);
        // J = -H*I
        auto jTmp = b.create<field::NegateOp>(h);
        auto j = b.create<field::MulOp>(jTmp, i);
        // r = 2*(S2-S1)
        auto rTmp = b.create<field::SubOp>(s2, s1);
        auto r = b.create<field::DoubleOp>(rTmp);
        // V = U1*I
        auto v = b.create<field::MulOp>(u1, i);
        // X3 = r²+J-2*V
        auto x3Tmp1 = b.create<field::SquareOp>(r);
        auto x3Tmp2 = b.create<field::AddOp>(x3Tmp1, j);
        auto x3Tmp3 = b.create<field::DoubleOp>(v);
        auto x3 = b.create<field::SubOp>(x3Tmp2, x3Tmp3);
        // Y3 = r*(V-X3)+2*S1*J
        auto y3Tmp1 = b.create<field::SubOp>(v, x3);
        auto y3Tmp2 = b.create<field::MulOp>(r, y3Tmp1);
        auto y3Tmp3 = b.create<field::DoubleOp>(s1);
        auto y3Tmp4 = b.create<field::MulOp>(y3Tmp3, j);
        auto y3 = b.create<field::AddOp>(y3Tmp2, y3Tmp4);
        // Z3 = ((Z1+Z2)²-Z1Z1-Z2Z2)*H
        auto z3Tmp1 = b.create<field::AddOp>(z1, z2);
        auto z3Tmp2 = b.create<field::SquareOp>(z3Tmp1);
        auto z3Tmp3 = b.create<field::SubOp>(z3Tmp2, z1z1);
        auto z3Tmp4 = b.create<field::SubOp>(z3Tmp3, z2z2);
        auto z3 = b.create<field::MulOp>(z3Tmp4, h);

        auto makePoint =
            b.create<tensor::FromElementsOp>(SmallVector<Value>({x3, y3, z3}));
        b.create<scf::YieldOp>(ValueRange{makePoint});
      });
  return ifOp.getResult(0);
}

Value jacobianAdd(const Value &p1, const Value &p2, Type p1Type, Type p2Type,
                  ImplicitLocOpBuilder &b) {
  if (isa<AffineType>(p1Type)) {
    if (isa<AffineType>(p2Type)) {
      return affineAndAffine(p1, p2, p1Type, b);
    }
    return jacobianAndAffine(p2, p1, p1Type, b);
  } else if (isa<AffineType>(p2Type)) {
    return jacobianAndAffine(p1, p2, p2Type, b);
  } else if (isa<JacobianType>(p1Type) && isa<JacobianType>(p2Type)) {
    return jacobianAndJacobian(p1, p2, p1Type, b);
  } else {
    assert(false && "Unsupported point types for Jacobian addition");
  }
}

}  // namespace mlir::zkir::elliptic_curve
