/* Copyright 2025 The PrimeIR Authors.

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

#include "prime_ir/Dialect/ModArith/Conversions/ModArithToArith/Inverter/BYInverter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithAttributes.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithOps.h"

// This implements the Bernstein-Yang modular inverse algorithm, which is a
// variant of the binary GCD algorithm optimized for constant-time
// implementation.

// Reference:
// https://github.com/bitcoin-core/secp256k1/blob/master/doc/safegcd_implementation.md
namespace mlir::prime_ir::mod_arith {

BYInverter::BYInverter(ImplicitLocOpBuilder &b, Type inputType)
    : b(b), modArithType(cast<ModArithType>(getElementTypeOrSelf(inputType))) {
  IntegerAttr modulus = modArithType.getModulus();
  BYAttr byAttr = modArithType.getBYAttr();
  unsigned extModBitWidth = byAttr.getNewBitWidth().getValue().getZExtValue();
  unsigned modBitWidth = modulus.getValue().getBitWidth();
  unsigned n = byAttr.getDivsteps().getValue().getZExtValue();
  unsigned limbBitWidth = n > 64 ? n : 64;

  intType = IntegerType::get(b.getContext(), modBitWidth);
  extIntType = IntegerType::get(b.getContext(), extModBitWidth);
  limbType = IntegerType::get(b.getContext(), limbBitWidth);

  maskN = arith::ConstantIntOp::create(
      b, limbType, APInt::getAllOnes(n).zextOrTrunc(limbBitWidth));
  APInt mInt = modulus.getValue().zext(extModBitWidth);
  APInt mInvInt = byAttr.getMInv().getValue();
  m = arith::ConstantIntOp::create(b, extIntType, mInt);
  mInv = arith::ConstantIntOp::create(b, limbType,
                                      mInvInt.zextOrTrunc(limbBitWidth));

  limbTypeOne = arith::ConstantIntOp::create(b, limbType, 1);
  limbTypeZero = arith::ConstantIntOp::create(b, limbType, 0);
  extIntTypeOne = arith::ConstantIntOp::create(b, extIntType, 1);
  extIntTypeZero = arith::ConstantIntOp::create(b, extIntType, 0);

  extIntTypeN = arith::ConstantIntOp::create(b, extIntType, n);
  limbTypeN = arith::ConstantIntOp::create(b, limbType, n);
}

BYInverter::JumpResult BYInverter::GenerateJump(Value f, Value g, Value eta) {
  TMatrix t = {limbTypeOne, limbTypeZero, limbTypeZero, limbTypeOne};
  Value steps = limbTypeN;

  f = arith::AndIOp::create(b, arith::TruncIOp::create(b, limbType, f), maskN);
  g = arith::AndIOp::create(b, arith::TruncIOp::create(b, limbType, g), maskN);

  SmallVector<Value, 8> initValues = {steps, f,     g,     eta,
                                      t.t00, t.t01, t.t10, t.t11};
  SmallVector<Type, 8> types(8, limbType);

  auto whileOp = scf::WhileOp::create(
      b, types, initValues,
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        ImplicitLocOpBuilder b(loc, builder);
        // While loop condition: steps != 0
        auto [steps, f, g, eta, t] =
            std::make_tuple(args[0], args[1], args[2], args[3],
                            TMatrix{args[4], args[5], args[6], args[7]});

        Value zeros = math::CountTrailingZerosOp::create(b, g);
        zeros = arith::MinUIOp::create(b, zeros, steps);
        steps = arith::SubIOp::create(b, steps, zeros);
        eta = arith::AddIOp::create(b, eta, zeros);
        g = arith::ShRUIOp::create(b, g, zeros);
        t.t00 = arith::ShLIOp::create(b, t.t00, zeros);
        t.t01 = arith::ShLIOp::create(b, t.t01, zeros);

        Value isContinue = arith::CmpIOp::create(b, arith::CmpIPredicate::ne,
                                                 steps, limbTypeZero);
        scf::ConditionOp::create(
            b, isContinue,
            ValueRange{steps, f, g, eta, t.t00, t.t01, t.t10, t.t11});
      },
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        ImplicitLocOpBuilder b(loc, builder);
        auto [steps, f, g, eta, t] =
            std::make_tuple(args[0], args[1], args[2], args[3],
                            TMatrix{args[4], args[5], args[6], args[7]});

        Value deltaPos = arith::CmpIOp::create(b, arith::CmpIPredicate::sgt,
                                               eta, limbTypeZero);

        Value negEta = arith::SubIOp::create(b, limbTypeZero, eta);
        Value negT00 = arith::SubIOp::create(b, limbTypeZero, t.t00);
        Value negT01 = arith::SubIOp::create(b, limbTypeZero, t.t01);
        Value negF = arith::SubIOp::create(b, limbTypeZero, f);

        eta = arith::SelectOp::create(b, deltaPos, negEta, eta);
        t.t00 = arith::SelectOp::create(b, deltaPos, t.t10, t.t00);
        t.t01 = arith::SelectOp::create(b, deltaPos, t.t11, t.t01);
        t.t10 = arith::SelectOp::create(b, deltaPos, negT00, t.t10);
        t.t11 = arith::SelectOp::create(b, deltaPos, negT01, t.t11);
        f = arith::SelectOp::create(b, deltaPos, g, f);
        g = arith::SelectOp::create(b, deltaPos, negF, g);

        Value five = arith::ConstantIntOp::create(b, limbType, 5);
        Value oneMinusEta = arith::SubIOp::create(b, limbTypeOne, eta);
        Value shift = arith::MinSIOp::create(
            b, arith::MinSIOp::create(b, steps, oneMinusEta), five);
        Value mask = arith::SubIOp::create(
            b, arith::ShLIOp::create(b, limbTypeOne, shift), limbTypeOne);

        Value threeF = arith::MulIOp::create(
            b, arith::ConstantIntOp::create(b, limbType, 3), f);
        Value twentyEight = arith::ConstantIntOp::create(b, limbType, 28);
        Value w = arith::AndIOp::create(
            b,
            arith::MulIOp::create(
                b, g, arith::XOrIOp::create(b, threeF, twentyEight)),
            mask);

        t.t10 =
            arith::AddIOp::create(b, arith::MulIOp::create(b, t.t00, w), t.t10);
        t.t11 =
            arith::AddIOp::create(b, arith::MulIOp::create(b, t.t01, w), t.t11);

        g = arith::AddIOp::create(b, g, arith::MulIOp::create(b, w, f));

        scf::YieldOp::create(
            b, ValueRange{steps, f, g, eta, t.t00, t.t01, t.t10, t.t11});
      });

  return JumpResult{TMatrix{whileOp.getResult(4), whileOp.getResult(5),
                            whileOp.getResult(6), whileOp.getResult(7)},
                    whileOp.getResult(3)};
}

BYInverter::FGResult BYInverter::GenerateFG(Value f, Value g, TMatrix t) {
  Value extT00 = arith::ExtSIOp::create(b, extIntType, t.t00);
  Value extT01 = arith::ExtSIOp::create(b, extIntType, t.t01);
  Value extT10 = arith::ExtSIOp::create(b, extIntType, t.t10);
  Value extT11 = arith::ExtSIOp::create(b, extIntType, t.t11);

  Value newF = arith::AddIOp::create(b, arith::MulIOp::create(b, f, extT00),
                                     arith::MulIOp::create(b, g, extT01));

  Value newG = arith::AddIOp::create(b, arith::MulIOp::create(b, f, extT10),
                                     arith::MulIOp::create(b, g, extT11));
  newF = arith::ShRSIOp::create(b, newF, extIntTypeN);
  newG = arith::ShRSIOp::create(b, newG, extIntTypeN);

  return {newF, newG};
}

BYInverter::DEResult BYInverter::GenerateDE(Value d, Value e, TMatrix t) {
  Value isNegD = arith::ExtUIOp::create(
      b, limbType,
      arith::CmpIOp::create(b, arith::CmpIPredicate::slt, d, extIntTypeZero));
  Value isNegE = arith::ExtUIOp::create(
      b, limbType,
      arith::CmpIOp::create(b, arith::CmpIPredicate::slt, e, extIntTypeZero));
  Value md = arith::AddIOp::create(b, arith::MulIOp::create(b, t.t00, isNegD),
                                   arith::MulIOp::create(b, t.t01, isNegE));
  Value me = arith::AddIOp::create(b, arith::MulIOp::create(b, t.t10, isNegD),
                                   arith::MulIOp::create(b, t.t11, isNegE));

  // Calculate cd and ce using lowest bits
  Value dLow =
      arith::AndIOp::create(b, arith::TruncIOp::create(b, limbType, d), maskN);
  Value eLow =
      arith::AndIOp::create(b, arith::TruncIOp::create(b, limbType, e), maskN);
  Value cd = arith::AndIOp::create(
      b,
      arith::AddIOp::create(b, arith::MulIOp::create(b, t.t00, dLow),
                            arith::MulIOp::create(b, t.t01, eLow)),
      maskN);
  Value ce = arith::AndIOp::create(
      b,
      arith::AddIOp::create(b, arith::MulIOp::create(b, t.t10, dLow),
                            arith::MulIOp::create(b, t.t11, eLow)),
      maskN);

  md = arith::SubIOp::create(
      b, md,
      arith::AndIOp::create(
          b, arith::AddIOp::create(b, arith::MulIOp::create(b, cd, mInv), md),
          maskN));
  me = arith::SubIOp::create(
      b, me,
      arith::AndIOp::create(
          b, arith::AddIOp::create(b, arith::MulIOp::create(b, ce, mInv), me),
          maskN));

  Value extT00 = arith::ExtSIOp::create(b, extIntType, t.t00);
  Value extT01 = arith::ExtSIOp::create(b, extIntType, t.t01);
  Value extT10 = arith::ExtSIOp::create(b, extIntType, t.t10);
  Value extT11 = arith::ExtSIOp::create(b, extIntType, t.t11);

  cd = arith::AddIOp::create(
      b,
      arith::AddIOp::create(b, arith::MulIOp::create(b, d, extT00),
                            arith::MulIOp::create(b, e, extT01)),
      arith::MulIOp::create(b, m, arith::ExtSIOp::create(b, extIntType, md)));
  ce = arith::AddIOp::create(
      b,
      arith::AddIOp::create(b, arith::MulIOp::create(b, d, extT10),
                            arith::MulIOp::create(b, e, extT11)),
      arith::MulIOp::create(b, m, arith::ExtSIOp::create(b, extIntType, me)));

  cd = arith::ShRSIOp::create(b, cd, extIntTypeN);
  ce = arith::ShRSIOp::create(b, ce, extIntTypeN);
  return {cd, ce};
}

Value BYInverter::GenerateNorm(Value value, Value antiunit) {
  Value isNeg = arith::CmpIOp::create(b, arith::CmpIPredicate::slt, value,
                                      extIntTypeZero);
  Value result = arith::SelectOp::create(
      b, isNeg, arith::AddIOp::create(b, value, m), value);

  result = arith::SelectOp::create(
      b, antiunit, arith::SubIOp::create(b, extIntTypeZero, result), result);

  result = arith::SelectOp::create(
      b,
      arith::CmpIOp::create(b, arith::CmpIPredicate::slt, result,
                            extIntTypeZero),
      arith::AddIOp::create(b, result, m), result);

  return result;
}

Value BYInverter::Generate(Value input, bool isMont) {
  Value f = m;
  Value g = arith::ExtUIOp::create(b, extIntType, input);
  Value d = extIntTypeZero;
  Value e;
  if (isMont) {
    MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
    e = arith::ConstantOp::create(b, montAttr.getRSquared());
    e = arith::ExtUIOp::create(b, extIntType, e);
  } else {
    e = extIntTypeOne;
  }
  Value eta = limbTypeOne;

  SmallVector<Value, 5> initValues = {f, g, d, e, eta};
  SmallVector<Type, 5> types;
  for (auto &v : initValues) {
    types.push_back(v.getType());
  }

  auto whileOp = scf::WhileOp::create(
      b, types, initValues,
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        ImplicitLocOpBuilder b(loc, builder);
        Value g = args[1];
        Value cond = arith::CmpIOp::create(b, arith::CmpIPredicate::ne, g,
                                           extIntTypeZero);
        scf::ConditionOp::create(b, cond, args);
      },
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        ImplicitLocOpBuilder b(loc, builder);
        Value f = args[0];
        Value g = args[1];
        Value d = args[2];
        Value e = args[3];
        Value eta = args[4];
        BYInverter::JumpResult jumpResult = GenerateJump(f, g, eta);
        BYInverter::FGResult fgResult = GenerateFG(f, g, jumpResult.t);
        BYInverter::DEResult deResult = GenerateDE(d, e, jumpResult.t);

        scf::YieldOp::create(b, ValueRange{fgResult.f, fgResult.g, deResult.d,
                                           deResult.e, jumpResult.eta});
      });
  f = whileOp.getResult(0);
  d = whileOp.getResult(2);

  Value minusOneIntType = arith::ConstantIntOp::create(
      b, extIntType, APInt::getAllOnes(extIntType.getWidth()));

  Value antiunit =
      arith::CmpIOp::create(b, arith::CmpIPredicate::eq, f, minusOneIntType);
  Value invertible = arith::OrIOp::create(
      b, arith::CmpIOp::create(b, arith::CmpIPredicate::eq, f, extIntTypeOne),
      antiunit);

  d = GenerateNorm(d, antiunit);
  // return zero for non-invertible input
  Value result = arith::SelectOp::create(b, invertible, d, extIntTypeZero);
  result = arith::TruncIOp::create(b, intType, result);
  return result;
}

} // namespace mlir::prime_ir::mod_arith
