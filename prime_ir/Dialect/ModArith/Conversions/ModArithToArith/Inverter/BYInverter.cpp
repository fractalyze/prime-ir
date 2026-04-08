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
  unsigned limbBitWidth = byAttr.getLimbBitWidth();

  intType = IntegerType::get(b.getContext(), modBitWidth);
  extIntType = IntegerType::get(b.getContext(), extModBitWidth);
  limbType = IntegerType::get(b.getContext(), limbBitWidth);

  maskN = b.create<arith::ConstantIntOp>(
      limbType, APInt::getAllOnes(n).zextOrTrunc(limbBitWidth));
  APInt mInt = modulus.getValue().zext(extModBitWidth);
  APInt mInvInt = byAttr.getMInv().getValue();
  m = b.create<arith::ConstantIntOp>(extIntType, mInt);
  mInv = b.create<arith::ConstantIntOp>(limbType,
                                        mInvInt.zextOrTrunc(limbBitWidth));

  limbTypeOne = b.create<arith::ConstantIntOp>(limbType, 1);
  limbTypeZero = b.create<arith::ConstantIntOp>(limbType, 0);
  extIntTypeOne = b.create<arith::ConstantIntOp>(extIntType, 1);
  extIntTypeZero = b.create<arith::ConstantIntOp>(extIntType, 0);

  extIntTypeN = b.create<arith::ConstantIntOp>(extIntType, n);
  limbTypeN = b.create<arith::ConstantIntOp>(limbType, n);
}

BYInverter::JumpResult BYInverter::GenerateJump(Value f, Value g, Value eta) {
  TMatrix t = {limbTypeOne, limbTypeZero, limbTypeZero, limbTypeOne};
  Value steps = limbTypeN;

  f = b.create<arith::AndIOp>(b.create<arith::TruncIOp>(limbType, f), maskN);
  g = b.create<arith::AndIOp>(b.create<arith::TruncIOp>(limbType, g), maskN);

  SmallVector<Value, 8> initValues = {steps, f,     g,     eta,
                                      t.t00, t.t01, t.t10, t.t11};
  SmallVector<Type, 8> types(8, limbType);

  auto whileOp = b.create<scf::WhileOp>(
      types, initValues,
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        ImplicitLocOpBuilder b(loc, builder);
        // While loop condition: steps != 0
        auto [steps, f, g, eta, t] =
            std::make_tuple(args[0], args[1], args[2], args[3],
                            TMatrix{args[4], args[5], args[6], args[7]});

        Value zeros = b.create<math::CountTrailingZerosOp>(g);
        zeros = b.create<arith::MinUIOp>(zeros, steps);
        steps = b.create<arith::SubIOp>(steps, zeros);
        eta = b.create<arith::AddIOp>(eta, zeros);
        g = b.create<arith::ShRUIOp>(g, zeros);
        t.t00 = b.create<arith::ShLIOp>(t.t00, zeros);
        t.t01 = b.create<arith::ShLIOp>(t.t01, zeros);

        Value isContinue = b.create<arith::CmpIOp>(arith::CmpIPredicate::ne,
                                                   steps, limbTypeZero);
        b.create<scf::ConditionOp>(
            isContinue,
            ValueRange{steps, f, g, eta, t.t00, t.t01, t.t10, t.t11});
      },
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        ImplicitLocOpBuilder b(loc, builder);
        auto [steps, f, g, eta, t] =
            std::make_tuple(args[0], args[1], args[2], args[3],
                            TMatrix{args[4], args[5], args[6], args[7]});

        Value deltaPos = b.create<arith::CmpIOp>(arith::CmpIPredicate::sgt, eta,
                                                 limbTypeZero);

        Value negEta = b.create<arith::SubIOp>(limbTypeZero, eta);
        Value negT00 = b.create<arith::SubIOp>(limbTypeZero, t.t00);
        Value negT01 = b.create<arith::SubIOp>(limbTypeZero, t.t01);
        Value negF = b.create<arith::SubIOp>(limbTypeZero, f);

        eta = b.create<arith::SelectOp>(deltaPos, negEta, eta);
        t.t00 = b.create<arith::SelectOp>(deltaPos, t.t10, t.t00);
        t.t01 = b.create<arith::SelectOp>(deltaPos, t.t11, t.t01);
        t.t10 = b.create<arith::SelectOp>(deltaPos, negT00, t.t10);
        t.t11 = b.create<arith::SelectOp>(deltaPos, negT01, t.t11);
        f = b.create<arith::SelectOp>(deltaPos, g, f);
        g = b.create<arith::SelectOp>(deltaPos, negF, g);

        Value five = b.create<arith::ConstantIntOp>(limbType, 5);
        Value oneMinusEta = b.create<arith::SubIOp>(limbTypeOne, eta);
        Value shift = b.create<arith::MinSIOp>(
            b.create<arith::MinSIOp>(steps, oneMinusEta), five);
        Value mask = b.create<arith::SubIOp>(
            b.create<arith::ShLIOp>(limbTypeOne, shift), limbTypeOne);

        Value threeF = b.create<arith::MulIOp>(
            b.create<arith::ConstantIntOp>(limbType, 3), f);
        Value twentyEight = b.create<arith::ConstantIntOp>(limbType, 28);
        Value w = b.create<arith::AndIOp>(
            b.create<arith::MulIOp>(
                g, b.create<arith::XOrIOp>(threeF, twentyEight)),
            mask);

        t.t10 =
            b.create<arith::AddIOp>(b.create<arith::MulIOp>(t.t00, w), t.t10);
        t.t11 =
            b.create<arith::AddIOp>(b.create<arith::MulIOp>(t.t01, w), t.t11);

        g = b.create<arith::AddIOp>(g, b.create<arith::MulIOp>(w, f));

        b.create<scf::YieldOp>(
            ValueRange{steps, f, g, eta, t.t00, t.t01, t.t10, t.t11});
      });

  return JumpResult{TMatrix{whileOp.getResult(4), whileOp.getResult(5),
                            whileOp.getResult(6), whileOp.getResult(7)},
                    whileOp.getResult(3)};
}

BYInverter::FGResult BYInverter::GenerateFG(Value f, Value g, TMatrix t) {
  Value extT00 = b.create<arith::ExtSIOp>(extIntType, t.t00);
  Value extT01 = b.create<arith::ExtSIOp>(extIntType, t.t01);
  Value extT10 = b.create<arith::ExtSIOp>(extIntType, t.t10);
  Value extT11 = b.create<arith::ExtSIOp>(extIntType, t.t11);

  Value newF = b.create<arith::AddIOp>(b.create<arith::MulIOp>(f, extT00),
                                       b.create<arith::MulIOp>(g, extT01));

  Value newG = b.create<arith::AddIOp>(b.create<arith::MulIOp>(f, extT10),
                                       b.create<arith::MulIOp>(g, extT11));
  newF = b.create<arith::ShRSIOp>(newF, extIntTypeN);
  newG = b.create<arith::ShRSIOp>(newG, extIntTypeN);

  return {newF, newG};
}

BYInverter::DEResult BYInverter::GenerateDE(Value d, Value e, TMatrix t) {
  Value isNegD = b.create<arith::ExtUIOp>(
      limbType,
      b.create<arith::CmpIOp>(arith::CmpIPredicate::slt, d, extIntTypeZero));
  Value isNegE = b.create<arith::ExtUIOp>(
      limbType,
      b.create<arith::CmpIOp>(arith::CmpIPredicate::slt, e, extIntTypeZero));
  Value md = b.create<arith::AddIOp>(b.create<arith::MulIOp>(t.t00, isNegD),
                                     b.create<arith::MulIOp>(t.t01, isNegE));
  Value me = b.create<arith::AddIOp>(b.create<arith::MulIOp>(t.t10, isNegD),
                                     b.create<arith::MulIOp>(t.t11, isNegE));

  // Calculate cd and ce using lowest bits
  Value dLow =
      b.create<arith::AndIOp>(b.create<arith::TruncIOp>(limbType, d), maskN);
  Value eLow =
      b.create<arith::AndIOp>(b.create<arith::TruncIOp>(limbType, e), maskN);
  Value cd = b.create<arith::AndIOp>(
      b.create<arith::AddIOp>(b.create<arith::MulIOp>(t.t00, dLow),
                              b.create<arith::MulIOp>(t.t01, eLow)),
      maskN);
  Value ce = b.create<arith::AndIOp>(
      b.create<arith::AddIOp>(b.create<arith::MulIOp>(t.t10, dLow),
                              b.create<arith::MulIOp>(t.t11, eLow)),
      maskN);

  md = b.create<arith::SubIOp>(
      md, b.create<arith::AndIOp>(
              b.create<arith::AddIOp>(b.create<arith::MulIOp>(cd, mInv), md),
              maskN));
  me = b.create<arith::SubIOp>(
      me, b.create<arith::AndIOp>(
              b.create<arith::AddIOp>(b.create<arith::MulIOp>(ce, mInv), me),
              maskN));

  Value extT00 = b.create<arith::ExtSIOp>(extIntType, t.t00);
  Value extT01 = b.create<arith::ExtSIOp>(extIntType, t.t01);
  Value extT10 = b.create<arith::ExtSIOp>(extIntType, t.t10);
  Value extT11 = b.create<arith::ExtSIOp>(extIntType, t.t11);

  cd = b.create<arith::AddIOp>(
      b.create<arith::AddIOp>(b.create<arith::MulIOp>(d, extT00),
                              b.create<arith::MulIOp>(e, extT01)),
      b.create<arith::MulIOp>(m, b.create<arith::ExtSIOp>(extIntType, md)));
  ce = b.create<arith::AddIOp>(
      b.create<arith::AddIOp>(b.create<arith::MulIOp>(d, extT10),
                              b.create<arith::MulIOp>(e, extT11)),
      b.create<arith::MulIOp>(m, b.create<arith::ExtSIOp>(extIntType, me)));

  cd = b.create<arith::ShRSIOp>(cd, extIntTypeN);
  ce = b.create<arith::ShRSIOp>(ce, extIntTypeN);
  return {cd, ce};
}

Value BYInverter::GenerateNorm(Value value, Value antiunit) {
  Value isNeg =
      b.create<arith::CmpIOp>(arith::CmpIPredicate::slt, value, extIntTypeZero);
  Value result = b.create<arith::SelectOp>(
      isNeg, b.create<arith::AddIOp>(value, m), value);

  result = b.create<arith::SelectOp>(
      antiunit, b.create<arith::SubIOp>(extIntTypeZero, result), result);

  result = b.create<arith::SelectOp>(
      b.create<arith::CmpIOp>(arith::CmpIPredicate::slt, result,
                              extIntTypeZero),
      b.create<arith::AddIOp>(result, m), result);

  return result;
}

Value BYInverter::Generate(Value input, bool isMont) {
  Value f = m;
  Value g = b.create<arith::ExtUIOp>(extIntType, input);
  Value d = extIntTypeZero;
  Value e;
  if (isMont) {
    MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
    e = b.create<arith::ConstantOp>(montAttr.getRSquared());
    e = b.create<arith::ExtUIOp>(extIntType, e);
  } else {
    e = extIntTypeOne;
  }
  Value eta = limbTypeOne;

  SmallVector<Value, 5> initValues = {f, g, d, e, eta};
  SmallVector<Type, 5> types;
  for (auto &v : initValues) {
    types.push_back(v.getType());
  }

  auto whileOp = b.create<scf::WhileOp>(
      types, initValues,
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        ImplicitLocOpBuilder b(loc, builder);
        Value g = args[1];
        Value cond = b.create<arith::CmpIOp>(arith::CmpIPredicate::ne, g,
                                             extIntTypeZero);
        b.create<scf::ConditionOp>(cond, args);
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

        b.create<scf::YieldOp>(ValueRange{fgResult.f, fgResult.g, deResult.d,
                                          deResult.e, jumpResult.eta});
      });
  f = whileOp.getResult(0);
  d = whileOp.getResult(2);

  Value minusOneIntType = b.create<arith::ConstantIntOp>(
      extIntType, APInt::getAllOnes(extIntType.getWidth()));

  Value antiunit =
      b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, f, minusOneIntType);
  Value invertible = b.create<arith::OrIOp>(
      b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, f, extIntTypeOne),
      antiunit);

  d = GenerateNorm(d, antiunit);
  // return zero for non-invertible input
  Value result = b.create<arith::SelectOp>(invertible, d, extIntTypeZero);
  result = b.create<arith::TruncIOp>(intType, result);
  return result;
}

} // namespace mlir::prime_ir::mod_arith
