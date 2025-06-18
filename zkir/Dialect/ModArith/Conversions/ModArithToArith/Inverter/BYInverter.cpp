#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/Inverter/BYInverter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "zkir/Dialect/ModArith/IR/ModArithAttributes.h"
#include "zkir/Dialect/ModArith/IR/ModArithOps.h"

// This implements the Bernstein-Yang modular inverse algorithm, which is a
// variant of the binary GCD algorithm optimized for constant-time
// implementation.

// Reference:
// https://github.com/bitcoin-core/secp256k1/blob/master/doc/safegcd_implementation.md
namespace mlir::zkir::mod_arith {

BYInverter::BYInverter(ImplicitLocOpBuilder &b, Type inputType)
    : b_(b),
      modArithType_(cast<ModArithType>(getElementTypeOrSelf(inputType))) {
  IntegerAttr modulus = modArithType_.getModulus();
  BYAttr byAttr = modArithType_.getBYAttr();
  unsigned extModBitWidth = byAttr.getNewBitWidth().getValue().getZExtValue();
  unsigned modBitWidth = modulus.getValue().getBitWidth();
  unsigned n = byAttr.getDivsteps().getValue().getZExtValue();
  unsigned limbBitWidth = n > 64 ? n : 64;

  intType_ = IntegerType::get(b.getContext(), modBitWidth);
  extIntType_ = IntegerType::get(b.getContext(), extModBitWidth);
  limbType_ = IntegerType::get(b.getContext(), limbBitWidth);

  maskN_ = b.create<arith::ConstantIntOp>(
      limbType_, APInt::getAllOnes(n).zextOrTrunc(limbBitWidth));
  APInt m = modulus.getValue().zext(extModBitWidth);
  APInt mInv = byAttr.getMInv().getValue();
  m_ = b.create<arith::ConstantIntOp>(extIntType_, m);
  mInv_ =
      b.create<arith::ConstantIntOp>(limbType_, mInv.zextOrTrunc(limbBitWidth));

  limbTypeOne_ = b.create<arith::ConstantIntOp>(limbType_, 1);
  limbTypeZero_ = b.create<arith::ConstantIntOp>(limbType_, 0);
  extIntTypeOne_ = b.create<arith::ConstantIntOp>(extIntType_, 1);
  extIntTypeZero_ = b.create<arith::ConstantIntOp>(extIntType_, 0);

  extIntTypeN_ = b.create<arith::ConstantIntOp>(extIntType_, n);
  limbTypeN_ = b.create<arith::ConstantIntOp>(limbType_, n);
}

BYInverter::JumpResult BYInverter::GenerateJump(Value f, Value g, Value eta) {
  TMatrix t = {limbTypeOne_, limbTypeZero_, limbTypeZero_, limbTypeOne_};
  Value steps = limbTypeN_;

  f = b_.create<arith::AndIOp>(b_.create<arith::TruncIOp>(limbType_, f),
                               maskN_);
  g = b_.create<arith::AndIOp>(b_.create<arith::TruncIOp>(limbType_, g),
                               maskN_);

  SmallVector<Value, 8> initValues = {steps, f,     g,     eta,
                                      t.t00, t.t01, t.t10, t.t11};
  SmallVector<Type, 8> types(8, limbType_);

  auto whileOp = b_.create<scf::WhileOp>(
      types, initValues,
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        ImplicitLocOpBuilder b(loc, builder);
        // While loop condition: steps != 0
        auto [steps, f, g, eta, t] =
            std::make_tuple(args[0], args[1], args[2], args[3],
                            TMatrix{args[4], args[5], args[6], args[7]});

        Value zeros = b_.create<math::CountTrailingZerosOp>(g);
        zeros = b_.create<arith::MinUIOp>(zeros, steps);
        steps = b_.create<arith::SubIOp>(steps, zeros);
        eta = b_.create<arith::AddIOp>(eta, zeros);
        g = b_.create<arith::ShRUIOp>(g, zeros);
        t.t00 = b_.create<arith::ShLIOp>(t.t00, zeros);
        t.t01 = b_.create<arith::ShLIOp>(t.t01, zeros);

        Value isContinue = b_.create<arith::CmpIOp>(arith::CmpIPredicate::ne,
                                                    steps, limbTypeZero_);
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
                                                 limbTypeZero_);

        Value negEta = b.create<arith::SubIOp>(limbTypeZero_, eta);
        Value negT00 = b.create<arith::SubIOp>(limbTypeZero_, t.t00);
        Value negT01 = b.create<arith::SubIOp>(limbTypeZero_, t.t01);
        Value negF = b.create<arith::SubIOp>(limbTypeZero_, f);

        eta = b.create<arith::SelectOp>(deltaPos, negEta, eta);
        t.t00 = b.create<arith::SelectOp>(deltaPos, t.t10, t.t00);
        t.t01 = b.create<arith::SelectOp>(deltaPos, t.t11, t.t01);
        t.t10 = b.create<arith::SelectOp>(deltaPos, negT00, t.t10);
        t.t11 = b.create<arith::SelectOp>(deltaPos, negT01, t.t11);
        f = b.create<arith::SelectOp>(deltaPos, g, f);
        g = b.create<arith::SelectOp>(deltaPos, negF, g);

        Value five = b.create<arith::ConstantIntOp>(limbType_, 5);
        Value oneMinusEta = b.create<arith::SubIOp>(limbTypeOne_, eta);
        Value shift = b.create<arith::MinSIOp>(
            b.create<arith::MinSIOp>(steps, oneMinusEta), five);
        Value mask = b.create<arith::SubIOp>(
            b.create<arith::ShLIOp>(limbTypeOne_, shift), limbTypeOne_);

        Value threeF = b.create<arith::MulIOp>(
            b.create<arith::ConstantIntOp>(limbType_, 3), f);
        Value twentyEight = b.create<arith::ConstantIntOp>(limbType_, 28);
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
  Value extT00 = b_.create<arith::ExtSIOp>(extIntType_, t.t00);
  Value extT01 = b_.create<arith::ExtSIOp>(extIntType_, t.t01);
  Value extT10 = b_.create<arith::ExtSIOp>(extIntType_, t.t10);
  Value extT11 = b_.create<arith::ExtSIOp>(extIntType_, t.t11);

  Value newF = b_.create<arith::AddIOp>(b_.create<arith::MulIOp>(f, extT00),
                                        b_.create<arith::MulIOp>(g, extT01));

  Value newG = b_.create<arith::AddIOp>(b_.create<arith::MulIOp>(f, extT10),
                                        b_.create<arith::MulIOp>(g, extT11));
  newF = b_.create<arith::ShRSIOp>(newF, extIntTypeN_);
  newG = b_.create<arith::ShRSIOp>(newG, extIntTypeN_);

  return {newF, newG};
}

BYInverter::DEResult BYInverter::GenerateDE(Value d, Value e, TMatrix t) {
  Value isNegD = b_.create<arith::ExtUIOp>(
      limbType_,
      b_.create<arith::CmpIOp>(arith::CmpIPredicate::slt, d, extIntTypeZero_));
  Value isNegE = b_.create<arith::ExtUIOp>(
      limbType_,
      b_.create<arith::CmpIOp>(arith::CmpIPredicate::slt, e, extIntTypeZero_));
  Value md = b_.create<arith::AddIOp>(b_.create<arith::MulIOp>(t.t00, isNegD),
                                      b_.create<arith::MulIOp>(t.t01, isNegE));
  Value me = b_.create<arith::AddIOp>(b_.create<arith::MulIOp>(t.t10, isNegD),
                                      b_.create<arith::MulIOp>(t.t11, isNegE));

  // Calculate cd and ce using lowest bits
  Value dLow = b_.create<arith::AndIOp>(
      b_.create<arith::TruncIOp>(limbType_, d), maskN_);
  Value eLow = b_.create<arith::AndIOp>(
      b_.create<arith::TruncIOp>(limbType_, e), maskN_);
  Value cd = b_.create<arith::AndIOp>(
      b_.create<arith::AddIOp>(b_.create<arith::MulIOp>(t.t00, dLow),
                               b_.create<arith::MulIOp>(t.t01, eLow)),
      maskN_);
  Value ce = b_.create<arith::AndIOp>(
      b_.create<arith::AddIOp>(b_.create<arith::MulIOp>(t.t10, dLow),
                               b_.create<arith::MulIOp>(t.t11, eLow)),
      maskN_);

  md = b_.create<arith::SubIOp>(
      md, b_.create<arith::AndIOp>(
              b_.create<arith::AddIOp>(b_.create<arith::MulIOp>(cd, mInv_), md),
              maskN_));
  me = b_.create<arith::SubIOp>(
      me, b_.create<arith::AndIOp>(
              b_.create<arith::AddIOp>(b_.create<arith::MulIOp>(ce, mInv_), me),
              maskN_));

  Value extT00 = b_.create<arith::ExtSIOp>(extIntType_, t.t00);
  Value extT01 = b_.create<arith::ExtSIOp>(extIntType_, t.t01);
  Value extT10 = b_.create<arith::ExtSIOp>(extIntType_, t.t10);
  Value extT11 = b_.create<arith::ExtSIOp>(extIntType_, t.t11);

  cd = b_.create<arith::AddIOp>(
      b_.create<arith::AddIOp>(b_.create<arith::MulIOp>(d, extT00),
                               b_.create<arith::MulIOp>(e, extT01)),
      b_.create<arith::MulIOp>(m_, b_.create<arith::ExtSIOp>(extIntType_, md)));
  ce = b_.create<arith::AddIOp>(
      b_.create<arith::AddIOp>(b_.create<arith::MulIOp>(d, extT10),
                               b_.create<arith::MulIOp>(e, extT11)),
      b_.create<arith::MulIOp>(m_, b_.create<arith::ExtSIOp>(extIntType_, me)));

  cd = b_.create<arith::ShRSIOp>(cd, extIntTypeN_);
  ce = b_.create<arith::ShRSIOp>(ce, extIntTypeN_);
  return {cd, ce};
}

Value BYInverter::GenerateNorm(Value value, Value antiunit) {
  Value isNeg = b_.create<arith::CmpIOp>(arith::CmpIPredicate::slt, value,
                                         extIntTypeZero_);
  Value result = b_.create<arith::SelectOp>(
      isNeg, b_.create<arith::AddIOp>(value, m_), value);

  result = b_.create<arith::SelectOp>(
      antiunit, b_.create<arith::SubIOp>(extIntTypeZero_, result), result);

  result = b_.create<arith::SelectOp>(
      b_.create<arith::CmpIOp>(arith::CmpIPredicate::slt, result,
                               extIntTypeZero_),
      b_.create<arith::AddIOp>(result, m_), result);

  return result;
}

Value BYInverter::Generate(Value input, bool isMont) {
  Value f = m_;
  Value g = b_.create<arith::ExtUIOp>(extIntType_, input);
  Value d = extIntTypeZero_;
  Value e;
  if (isMont) {
    MontgomeryAttr montAttr = modArithType_.getMontgomeryAttr();
    e = b_.create<arith::ConstantOp>(montAttr.getRSquared());
    e = b_.create<arith::ExtUIOp>(extIntType_, e);
  } else {
    e = extIntTypeOne_;
  }
  Value eta = limbTypeOne_;

  SmallVector<Value, 5> initValues = {f, g, d, e, eta};
  SmallVector<Type, 5> types;
  for (auto &v : initValues) {
    types.push_back(v.getType());
  }

  auto whileOp = b_.create<scf::WhileOp>(
      types, initValues,
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        ImplicitLocOpBuilder b(loc, builder);
        Value g = args[1];
        Value cond = b_.create<arith::CmpIOp>(arith::CmpIPredicate::ne, g,
                                              extIntTypeZero_);
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

  Value minusOneIntType = b_.create<arith::ConstantIntOp>(
      extIntType_, APInt::getAllOnes(extIntType_.getWidth()));

  Value antiunit =
      b_.create<arith::CmpIOp>(arith::CmpIPredicate::eq, f, minusOneIntType);
  Value invertible = b_.create<arith::OrIOp>(
      b_.create<arith::CmpIOp>(arith::CmpIPredicate::eq, f, extIntTypeOne_),
      antiunit);

  d = GenerateNorm(d, antiunit);
  // return zero for non-invertible input
  Value result = b_.create<arith::SelectOp>(invertible, d, extIntTypeZero_);
  result = b_.create<arith::TruncIOp>(intType_, result);
  return result;
}

Value BYInverter::BatchGenerate(Value input, bool isMont,
                                ShapedType shapedType) {
  Value oneIndex = b_.create<arith::ConstantIndexOp>(1);
  Value zeroIndex = b_.create<arith::ConstantIndexOp>(0);
  Value sizeIndex =
      b_.create<arith::ConstantIndexOp>(shapedType.getNumElements());

  Value one =
      b_.create<ConstantOp>(modArithType_, IntegerAttr::get(intType_, 1));
  if (isMont) {
    one = b_.create<ToMontOp>(modArithType_, one);
  }
  Value zero =
      b_.create<ConstantOp>(modArithType_, IntegerAttr::get(intType_, 0));
  Value productions =
      b_.create<tensor::EmptyOp>(shapedType.getShape(), modArithType_);
  productions = b_.create<tensor::InsertOp>(one, productions, zeroIndex);
  Value product = one;

  // calculate [a₁, a₁*a₂, ..., a₁*a₂*...*aₙ]
  // TODO(quanxi1): try parallelizing the reduction
  auto forOp = b_.create<scf::ForOp>(
      /*lb=*/zeroIndex,
      /*ub=*/sizeIndex,
      /*step=*/oneIndex,
      /*iterArgs=*/ValueRange{productions, product},
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange iterArgs) {
        ImplicitLocOpBuilder b(loc, builder);
        Value element = b.create<tensor::ExtractOp>(modArithType_, input, iv);
        Value productions = iterArgs[0];
        Value product = iterArgs[1];

        Value isNotZero =
            b.create<CmpOp>(arith::CmpIPredicate::ne, element, zero);
        auto ifOp = b.create<scf::IfOp>(
            isNotZero,
            [&](OpBuilder &builder, Location loc) {
              ImplicitLocOpBuilder b(loc, builder);
              Value newProduct = b.create<MulOp>(product, element);
              b.create<scf::YieldOp>(ValueRange{
                  b.create<tensor::InsertOp>(newProduct, productions, iv),
                  newProduct});
            },
            [&](OpBuilder &builder, Location loc) {
              ImplicitLocOpBuilder b(loc, builder);
              b.create<scf::YieldOp>(ValueRange{productions, product});
            });
        b.create<scf::YieldOp>(ifOp.getResults());
      });

  // [a₁, a₁*a₂, ..., a₁*a₂*...*aₙ]
  productions = forOp.getResult(0);
  // a₁*a₂*...*aₙ
  product = forOp.getResult(1);

  // (a₁*a₂* ... *aᵢ)⁻¹
  Value invertedProduct =
      Generate(b_.create<ExtractOp>(intType_, product), isMont);
  invertedProduct = b_.create<EncapsulateOp>(modArithType_, invertedProduct);

  // calculate [a₁⁻¹, a₂⁻¹, ..., aₙ⁻¹]
  // TODO(quanxi1): Currently this is lowered to allocating a new buffer. Change
  // this to in-place operation reusing the input buffer
  Value result =
      b_.create<tensor::EmptyOp>(shapedType.getShape(), modArithType_);
  auto forOp2 = b_.create<scf::ForOp>(
      /*lb=*/zeroIndex,
      /*ub=*/b_.create<arith::SubIOp>(sizeIndex, oneIndex),
      /*step=*/oneIndex,
      /*iterArgs=*/ValueRange{result, invertedProduct},
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange iterArgs) {
        ImplicitLocOpBuilder b(loc, builder);
        Value currIndex = b.create<arith::SubIOp>(
            sizeIndex, b.create<arith::AddIOp>(iv, oneIndex));
        // (a₁*a₂*...*aᵢ)⁻¹
        Value result = iterArgs[0];
        Value invertedProduct = iterArgs[1];
        Value element =
            b.create<tensor::ExtractOp>(modArithType_, input, currIndex);

        Value isNotZero =
            b.create<CmpOp>(arith::CmpIPredicate::ne, element, zero);
        auto ifOp = b.create<scf::IfOp>(
            isNotZero,
            [&](OpBuilder &builder, Location loc) {
              ImplicitLocOpBuilder b(loc, builder);
              Value prevIndex = b.create<arith::SubIOp>(currIndex, oneIndex);
              // a₁*a₂*...*aᵢ₋₁
              Value prevProd = b.create<tensor::ExtractOp>(
                  modArithType_, productions, prevIndex);
              // aᵢ⁻¹
              Value elementInv = b.create<MulOp>(prevProd, invertedProduct);
              Value newResult =
                  b.create<tensor::InsertOp>(elementInv, result, currIndex);
              // newInvertedProduct := invertedProduct * aᵢ = (a₁*a₂*...*aᵢ₋₁)⁻¹
              Value newInvertedProduct =
                  b.create<MulOp>(invertedProduct, element);
              b.create<scf::YieldOp>(ValueRange{newResult, newInvertedProduct});
            },
            [&](OpBuilder &builder, Location loc) {
              ImplicitLocOpBuilder b(loc, builder);
              b.create<scf::YieldOp>(ValueRange{result, invertedProduct});
            });

        b.create<scf::YieldOp>(ifOp.getResults());
      });

  // [0, a₂⁻¹, ..., aₙ⁻¹]
  result = forOp2.getResult(0);
  // a₁⁻¹
  invertedProduct = forOp2.getResult(1);
  return b_.create<tensor::InsertOp>(invertedProduct, result, zeroIndex);
}

}  // namespace mlir::zkir::mod_arith
