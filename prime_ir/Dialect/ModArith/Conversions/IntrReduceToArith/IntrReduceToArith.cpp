/* Copyright 2026 The PrimeIR Authors.

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

#include "prime_ir/Dialect/ModArith/Conversions/IntrReduceToArith/IntrReduceToArith.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithOps.h"

namespace mlir::prime_ir::mod_arith {

#define GEN_PASS_DEF_INTRREDUCETOARITH
#include "prime_ir/Dialect/ModArith/Conversions/IntrReduceToArith/IntrReduceToArith.h.inc"

namespace {

// Helper to create modulus constant, handling shaped types
Value createModulusConstant(ImplicitLocOpBuilder &b, IntegerAttr modAttr,
                            Type inputType) {
  TypedAttr attr = modAttr;
  if (auto shapedType = dyn_cast<ShapedType>(inputType)) {
    attr = SplatElementsAttr::get(shapedType, modAttr);
  }
  return b.create<arith::ConstantOp>(attr);
}

// Implements conditional subtract for range [0, 2*mod)
// if (input >= mod) input -= mod
Value reduceFromExtended(ImplicitLocOpBuilder &b, Value input,
                         IntegerAttr modAttr) {
  Value cmod = createModulusConstant(b, modAttr, input.getType());
  if (isa<VectorType>(input.getType())) {
    auto sub = b.create<arith::SubIOp>(input, cmod);
    auto min = b.create<arith::MinUIOp>(sub, input);
    return min.getResult();
  } else {
    auto iflt = b.create<arith::CmpIOp>(arith::CmpIPredicate::ult, input, cmod);
    auto sub = b.create<arith::SubIOp>(input, cmod);
    auto select = b.create<arith::SelectOp>(iflt, input, sub);
    return select.getResult();
  }
}

// Implements conditional add for range [-mod+1, mod)
// if (input < 0) input += mod
Value reduceFromNegative(ImplicitLocOpBuilder &b, Value input,
                         IntegerAttr modAttr) {
  Type intType = input.getType();
  Value cmod = createModulusConstant(b, modAttr, intType);
  Value zero;
  if (isa<ShapedType>(intType)) {
    zero = b.create<arith::ConstantOp>(SplatElementsAttr::get(
        cast<ShapedType>(intType),
        IntegerAttr::get(getElementTypeOrSelf(intType), 0)));
  } else {
    zero = b.create<arith::ConstantIntOp>(intType, 0);
  }

  Value isNegative =
      b.create<arith::CmpIOp>(arith::CmpIPredicate::slt, input, zero);
  Value added = b.create<arith::AddIOp>(input, cmod);
  return b.create<arith::SelectOp>(isNegative, added, input);
}

// Gets the canonical form from an input value in [0, 2n).
Value getCanonicalFromExtended(ImplicitLocOpBuilder &b, Value input,
                               IntegerAttr modAttr) {
  Value cmod = createModulusConstant(b, modAttr, input.getType());
  if (isa<VectorType>(input.getType())) {
    auto sub = b.create<arith::SubIOp>(input, cmod);
    auto min = b.create<arith::MinUIOp>(sub, input);
    return min.getResult();
  } else {
    auto iflt = b.create<arith::CmpIOp>(arith::CmpIPredicate::ult, input, cmod);
    auto sub = b.create<arith::SubIOp>(input, cmod);
    auto select = b.create<arith::SelectOp>(iflt, input, sub);
    return select.getResult();
  }
}

// Gets the canonical difference of two values in modular arithmetic.
// Computes (lhs - rhs) mod n, returning a value in [0, n).
Value getCanonicalDiff(ImplicitLocOpBuilder &b, Value lhs, Value rhs,
                       IntegerAttr modAttr) {
  Value cmod = createModulusConstant(b, modAttr, lhs.getType());
  auto sub = b.create<arith::SubIOp>(lhs, rhs);
  if (isa<VectorType>(lhs.getType())) {
    auto add = b.create<arith::AddIOp>(sub, cmod);
    auto min = b.create<arith::MinUIOp>(sub, add);
    return min.getResult();
  } else {
    auto underflowed =
        b.create<arith::CmpIOp>(arith::CmpIPredicate::ult, lhs, rhs);
    auto add = b.create<arith::AddIOp>(sub, cmod);
    auto select = b.create<arith::SelectOp>(underflowed, add, sub);
    return select.getResult();
  }
}

// Checks if the input is from a signed multiplication.
bool isFromSignedMul(Value input) {
  auto signedOp = input.getDefiningOp<arith::MulSIExtendedOp>();
  return signedOp && signedOp.getLhs() != signedOp.getRhs();
}

// Performs single-limb Montgomery reduction.
Value reduceSingleLimb(ImplicitLocOpBuilder &b, Value tLow, Value tHigh,
                       IntegerAttr modAttr, IntegerAttr nInvAttr) {
  // Prepare nInv constant.
  TypedAttr nInvTyped = nInvAttr;
  Type limbType = nInvTyped.getType();
  if (auto shapedType = dyn_cast<ShapedType>(tLow.getType())) {
    limbType = shapedType.cloneWith(std::nullopt, limbType);
    nInvTyped = SplatElementsAttr::get(cast<ShapedType>(limbType), nInvTyped);
  }
  auto nInvConst = b.create<arith::ConstantOp>(nInvTyped);

  // Prepare modulus constant.
  auto modConst = createModulusConstant(b, modAttr, tLow.getType());

  // Compute `m` = `tLow` * `nInv` (mod `base`)
  auto m = b.create<arith::MulIOp>(tLow, nInvConst);
  // Compute `m` * `n`
  Value mNHigh;
  if (isFromSignedMul(tLow)) {
    auto mN = b.create<arith::MulSIExtendedOp>(m, modConst);
    mNHigh = mN.getHigh();
  } else {
    auto mN = b.create<arith::MulUIExtendedOp>(m, modConst);
    mNHigh = mN.getHigh();
  }

  // Calculate `T` - `mN`, which should result in zeroed low limb.
  // The reduction result will be just `tHigh` - `mN.getHigh()` mod n.
  auto sub = getCanonicalDiff(b, tHigh, mNHigh, modAttr);
  return sub;
}

// Performs multi-limb Montgomery reduction.
Value reduceMultiLimb(ImplicitLocOpBuilder &b, Value tLow, Value tHigh,
                      IntegerAttr modAttr, IntegerAttr nPrimeAttr,
                      IntegerAttr bInvAttr) {
  // Retrieve the modulus bitwidth.
  const unsigned modBitWidth =
      cast<IntegerType>(getElementTypeOrSelf(modAttr.getType())).getWidth();

  // Compute number of limbs.
  const unsigned limbWidth = nPrimeAttr.getType().getIntOrFloatBitWidth();
  const unsigned numLimbs = (modBitWidth + limbWidth - 1) / limbWidth;

  // Prepare constants for limb operations.
  Type limbType = nPrimeAttr.getType();
  TypedAttr limbWidthAttr =
      b.getIntegerAttr(getElementTypeOrSelf(tLow), limbWidth);
  TypedAttr limbMaskAttr =
      b.getIntegerAttr(getElementTypeOrSelf(tLow),
                       APInt::getAllOnes(limbWidth).zext(modBitWidth));
  TypedAttr limbShiftAttr =
      b.getIntegerAttr(getElementTypeOrSelf(tLow), (numLimbs - 1) * limbWidth);
  TypedAttr oneAttr = b.getIntegerAttr(getElementTypeOrSelf(tLow), 1);

  TypedAttr modAttrLocal = modAttr;
  TypedAttr nPrimeAttrLocal = nPrimeAttr;
  TypedAttr bInvAttrLocal = bInvAttr;

  // Splat the attributes to match the shape of `tLow`.
  if (auto shapedType = dyn_cast<ShapedType>(tLow.getType())) {
    limbType = shapedType.cloneWith(std::nullopt, limbType);
    nPrimeAttrLocal =
        SplatElementsAttr::get(cast<ShapedType>(limbType), nPrimeAttrLocal);
    limbWidthAttr = SplatElementsAttr::get(shapedType, limbWidthAttr);
    limbMaskAttr = SplatElementsAttr::get(shapedType, limbMaskAttr);
    limbShiftAttr = SplatElementsAttr::get(shapedType, limbShiftAttr);
    modAttrLocal = isa<VectorType>(modAttrLocal.getType())
                       ? modAttrLocal
                       : SplatElementsAttr::get(shapedType, modAttrLocal);
    bInvAttrLocal = SplatElementsAttr::get(shapedType, bInvAttrLocal);
    oneAttr = SplatElementsAttr::get(shapedType, oneAttr);
  }

  // Create constants for the Montgomery reduction.
  auto nPrimeConst = b.create<arith::ConstantOp>(nPrimeAttrLocal);
  auto limbWidthConst = b.create<arith::ConstantOp>(limbWidthAttr);
  auto limbMaskConst = b.create<arith::ConstantOp>(limbMaskAttr);
  auto limbShiftConst = b.create<arith::ConstantOp>(limbShiftAttr);
  auto modConst = b.create<arith::ConstantOp>(modAttrLocal);
  auto bInvConst = b.create<arith::ConstantOp>(bInvAttrLocal);
  auto oneConst = b.create<arith::ConstantOp>(oneAttr);

  auto noOverflow = arith::IntegerOverflowFlagsAttr::get(
      b.getContext(),
      arith::IntegerOverflowFlags::nuw | arith::IntegerOverflowFlags::nsw);

  // Because the number of limbs (`numLimbs`) is known at compile time, we can
  // unroll the loop as a straight-line chain of operations.
  for (unsigned i = 0; i < numLimbs - 1; ++i) {
    // Shift `T` right by `limbWidth`.
    Value freeCoeff = b.create<arith::AndIOp>(tLow, limbMaskConst);
    tLow = b.create<arith::ShRUIOp>(tLow, limbWidthConst);
    Value tHighLowerLimb = b.create<arith::ShLIOp>(tHigh, limbShiftConst);
    tLow = b.create<arith::OrIOp>(tLow, tHighLowerLimb);
    tHigh = b.create<arith::ShRUIOp>(tHigh, limbWidthConst);

    // Compute `m` = `freeCoeff` * (b^-1 mod n) and add to `T`.
    auto m = b.create<arith::MulUIExtendedOp>(freeCoeff, bInvConst);
    tLow = b.create<arith::AddIOp>(tLow, m.getLow());
    Value carry =
        b.create<arith::CmpIOp>(arith::CmpIPredicate::ult, tLow, m.getLow());
    tHigh = b.create<arith::AddIOp>(tHigh, m.getHigh(), noOverflow);
    auto tHighPlusOne = b.create<arith::AddIOp>(tHigh, oneConst, noOverflow);
    tHigh = b.create<arith::SelectOp>(carry, tHighPlusOne, tHigh);
  }

  // The last iteration is the same as normal Montgomery reduction.
  Value freeCoeff = b.create<arith::TruncIOp>(limbType, tLow);
  // Compute `m` = `freeCoeff` * `nPrime` (mod `base`)
  auto m = b.create<arith::MulIOp>(freeCoeff, nPrimeConst);
  // Compute `m` * `n`
  Value mExt = b.create<arith::ExtUIOp>(tLow.getType(), m);
  auto mN = b.create<arith::MulUIExtendedOp>(modConst, mExt);
  // Add the product to `T`.
  tLow = b.create<arith::AddIOp>(tLow, mN.getLow());
  auto carry =
      b.create<arith::CmpIOp>(arith::CmpIPredicate::ult, tLow, mN.getLow());
  tHigh = b.create<arith::AddIOp>(tHigh, mN.getHigh(), noOverflow);
  auto tHighPlusOne = b.create<arith::AddIOp>(tHigh, oneConst, noOverflow);
  tHigh = b.create<arith::SelectOp>(carry, tHighPlusOne, tHigh);
  // Shift right `T` by `limbWidth` to discard the zeroed limb.
  tLow = b.create<arith::ShRUIOp>(tLow, limbWidthConst);
  tHigh = b.create<arith::ShLIOp>(tHigh, limbShiftConst);
  tLow = b.create<arith::OrIOp>(tLow, tHigh);

  // Final conditional subtraction: if (`tLow` >= `modulus`) then subtract.
  auto result = getCanonicalFromExtended(b, tLow, modAttr);
  return result;
}

// Conversion pattern for IntrReduceOp (single-operand range reduction).
// Dispatches to conditional add or subtract based on input_range.
struct ConvertIntrReduce : public OpRewritePattern<IntrReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IntrReduceOp op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    IntegerAttr modulusAttr = op.getModulusAttr();
    LLVM::ConstantRangeAttr rangeAttr = op.getInputRangeAttr();
    Value input = op.getInput();

    // Determine strategy based on range
    APInt lower = rangeAttr.getLower();

    Value result;
    if (lower.isNegative()) {
      // Range includes negative values: conditional add (after subtraction)
      result = reduceFromNegative(b, input, modulusAttr);
    } else {
      // Range is non-negative: conditional subtract (after addition)
      result = reduceFromExtended(b, input, modulusAttr);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Conversion pattern for IntrMontReduceOp (Montgomery reduction).
// Handles both single-limb and multi-limb cases.
struct ConvertIntrMontReduce : public OpRewritePattern<IntrMontReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IntrMontReduceOp op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    IntegerAttr modulusAttr = op.getModulusAttr();
    Value low = op.getLow();
    Value high = op.getHigh();

    Value result;

    // Determine if single-limb or multi-limb based on attributes
    if (auto nInvAttr = op.getNInvAttr()) {
      // Single-limb Montgomery reduction
      result = reduceSingleLimb(b, low, high, modulusAttr, nInvAttr);
    } else {
      // Multi-limb Montgomery reduction
      result = reduceMultiLimb(b, low, high, modulusAttr, op.getNPrimeAttr(),
                               op.getBInvAttr());
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct IntrReduceToArith : impl::IntrReduceToArithBase<IntrReduceToArith> {
  using IntrReduceToArithBase::IntrReduceToArithBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<ConvertIntrReduce, ConvertIntrMontReduce>(context);

    // Convert both IntrReduceOp and IntrMontReduceOp
    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalOp<IntrReduceOp, IntrMontReduceOp>();

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::prime_ir::mod_arith
