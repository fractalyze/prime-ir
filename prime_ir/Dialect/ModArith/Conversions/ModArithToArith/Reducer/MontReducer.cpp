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

#include "prime_ir/Dialect/ModArith/Conversions/ModArithToArith/Reducer/MontReducer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h" // IWYU pragma: keep

namespace mlir::prime_ir::mod_arith {

MontReducer::MontReducer(ImplicitLocOpBuilder &b, ModArithType modArithType)
    : b(b), modAttr(modArithType.getModulus()),
      montAttr(modArithType.getMontgomeryAttr()) {}

namespace {

// Create a splat constant that works for both static and dynamic tensor shapes.
// For static shapes, uses SplatElementsAttr (compile-time constant).
// For dynamic shapes, uses linalg.fill(scalar, tensor.empty(dims)).
Value createSplatConst(ImplicitLocOpBuilder &b, TypedAttr scalarAttr,
                       ShapedType shapedType, Value shapeRef) {
  if (shapedType.hasStaticShape()) {
    return b.create<arith::ConstantOp>(
        SplatElementsAttr::get(shapedType, scalarAttr));
  }
  assert(shapeRef &&
         "A shape reference value must be provided for dynamic shapes.");
  Value scalar = b.create<arith::ConstantOp>(scalarAttr);
  SmallVector<Value> dynamicDims;
  for (int64_t i = 0; i < shapedType.getRank(); ++i) {
    if (shapedType.isDynamicDim(i)) {
      auto idx = b.create<arith::ConstantIndexOp>(i);
      dynamicDims.push_back(b.create<tensor::DimOp>(shapeRef, idx));
    }
  }
  Value empty = b.create<tensor::EmptyOp>(shapedType, dynamicDims);
  return b.create<linalg::FillOp>(scalar, empty).getResult(0);
}

} // namespace

Value MontReducer::createModulusConst(Type inputType, Value inputValue) {
  if (auto shapedType = dyn_cast<ShapedType>(inputType)) {
    if (!isa<VectorType>(this->modAttr.getType())) {
      return createSplatConst(b, this->modAttr, shapedType, inputValue);
    }
  }
  return b.create<arith::ConstantOp>(this->modAttr);
}

Value MontReducer::getCanonicalFromExtended(Value input, uint64_t bound) {
  if (bound <= 1)
    return input;

  // Binary reduction: ceil(log₂(bound)) conditional subtractions.
  // For [0, k * p), iterate from i = ceil(log₂(k)) - 1 down to 0:
  //   if (v >= 2ⁱ * p) v -= 2ⁱ * p
  // Each step halves the worst-case range. For bound == 2 this is a single
  // conditional subtraction of p (equivalent to the old special case).
  APInt mod = cast<IntegerAttr>(modAttr).getValue();
  unsigned w = mod.getBitWidth();
  unsigned m = 0;
  for (uint64_t k = bound - 1; k > 0; k >>= 1)
    ++m;
  for (int i = m - 1; i >= 0; --i) {
    APInt multiple = mod.zext(w) * APInt(w, uint64_t{1} << i);
    TypedAttr multipleAttr = IntegerAttr::get(modAttr.getType(), multiple);
    Value threshConst;
    if (auto shapedType = dyn_cast<ShapedType>(input.getType()))
      threshConst = createSplatConst(b, multipleAttr, shapedType, input);
    else
      threshConst = b.create<arith::ConstantOp>(multipleAttr);

    auto sub = b.create<arith::SubIOp>(input, threshConst);
    input = b.create<arith::MinUIOp>(sub, input).getResult();
  }
  return input;
}

Value MontReducer::getCanonicalFromExtended(Value input, Value overflow) {
  auto cmod = createModulusConst(input.getType(), input);
  // NOTE(chokobole): 'ult' is generally preferred over 'uge' for
  // better performance. However, using 'ult' would require inverting the
  // overflow check logic. Currently, 'uge' is used for clearer logic, but
  // this choice should be re-evaluated after benchmarking. See
  // https://github.com/fractalyze/prime-ir/pull/86/commits/10d3807
  auto ifge = b.create<arith::CmpIOp>(arith::CmpIPredicate::uge, input, cmod);
  auto or_ = b.create<arith::OrIOp>(ifge, overflow);
  auto sub = b.create<arith::SubIOp>(input, cmod);
  auto select = b.create<arith::SelectOp>(or_, sub, input);
  return select.getResult();
}

Value MontReducer::getCanonicalDiff(Value lhs, Value rhs) {
  auto cmod = createModulusConst(lhs.getType(), lhs);
  auto sub = b.create<arith::SubIOp>(lhs, rhs);
  auto add = b.create<arith::AddIOp>(sub, cmod);
  APInt mod = cast<IntegerAttr>(modAttr).getValue();
  if (mod.isSignBitSet()) {
    // When p > 2^(w-1), diff + p can overflow, so minui gives wrong results.
    // Fall back to cmpi + select.
    auto underflowed =
        b.create<arith::CmpIOp>(arith::CmpIPredicate::ult, lhs, rhs);
    return b.create<arith::SelectOp>(underflowed, add, sub).getResult();
  }
  return b.create<arith::MinUIOp>(sub, add).getResult();
}

bool MontReducer::isFromSignedMul(Value input) {
  auto signedOp = input.getDefiningOp<arith::MulSIExtendedOp>();
  return signedOp && signedOp.getLhs() != signedOp.getRhs();
}

Value MontReducer::reduceSingleLimb(Value tLow, Value tHigh, bool lazy) {
  TypedAttr nInvAttr = montAttr.getNInv();
  Value nInvConst;
  if (auto shapedType = dyn_cast<ShapedType>(tLow.getType())) {
    auto nInvShaped = shapedType.cloneWith(std::nullopt, nInvAttr.getType());
    nInvConst =
        createSplatConst(b, nInvAttr, cast<ShapedType>(nInvShaped), tLow);
  } else {
    nInvConst = b.create<arith::ConstantOp>(nInvAttr);
  }
  auto modConst = createModulusConst(tLow.getType(), tLow);

  // Compute m = tLow * nInv (mod base).
  auto m = b.create<arith::MulIOp>(tLow, nInvConst);
  // Compute m * n.
  Value mNHigh;
  if (isFromSignedMul(tLow)) {
    auto mN = b.create<arith::MulSIExtendedOp>(m, modConst);
    mNHigh = mN.getHigh();
  } else {
    auto mN = b.create<arith::MulUIExtendedOp>(m, modConst);
    mNHigh = mN.getHigh();
  }

  // The low part of T - mN is always zero (divisible by base), so the
  // result is just tHigh - mNHigh mod n.
  if (lazy) {
    // tHigh - mNHigh can underflow, so unconditionally add p.
    // Result is in [0, 2p).
    auto sub = b.create<arith::SubIOp>(tHigh, mNHigh);
    return b.create<arith::AddIOp>(sub, modConst).getResult();
  }
  return getCanonicalDiff(tHigh, mNHigh);
}

Value MontReducer::reduceMultiLimb(Value tLow, Value tHigh, bool lazy) {
  TypedAttr nPrimeAttr = montAttr.getNPrime();

  // Retrieve the modulus bitwidth.
  const unsigned modBitWidth =
      cast<IntegerType>(getElementTypeOrSelf(modAttr.getType())).getWidth();

  // Compute number of limbs.
  const unsigned limbWidth = montAttr.getLimbWidth();
  const unsigned numLimbs = montAttr.getNumLimbs();

  TypedAttr bInvAttr = montAttr.getBInv();
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

  // Splat the attributes to match the shape of `tLow`.
  if (auto shapedType = dyn_cast<ShapedType>(tLow.getType())) {
    limbType = shapedType.cloneWith(std::nullopt, limbType);
    nPrimeAttr = SplatElementsAttr::get(cast<ShapedType>(limbType), nPrimeAttr);
    limbWidthAttr = SplatElementsAttr::get(shapedType, limbWidthAttr);
    limbMaskAttr = SplatElementsAttr::get(shapedType, limbMaskAttr);
    limbShiftAttr = SplatElementsAttr::get(shapedType, limbShiftAttr);
    modAttrLocal = isa<VectorType>(modAttrLocal.getType())
                       ? modAttrLocal
                       : SplatElementsAttr::get(shapedType, modAttrLocal);
    bInvAttr = SplatElementsAttr::get(shapedType, bInvAttr);
    oneAttr = SplatElementsAttr::get(shapedType, oneAttr);
  }

  // Create constants for the Montgomery reduction.
  auto nPrimeConst = b.create<arith::ConstantOp>(nPrimeAttr);
  auto limbWidthConst = b.create<arith::ConstantOp>(limbWidthAttr);
  auto limbMaskConst = b.create<arith::ConstantOp>(limbMaskAttr);
  auto limbShiftConst = b.create<arith::ConstantOp>(limbShiftAttr);
  auto modConst = b.create<arith::ConstantOp>(modAttrLocal);
  auto bInvConst = b.create<arith::ConstantOp>(bInvAttr);
  auto oneConst = b.create<arith::ConstantOp>(oneAttr);

  auto noOverflow = arith::IntegerOverflowFlagsAttr::get(
      b.getContext(),
      arith::IntegerOverflowFlags::nuw | arith::IntegerOverflowFlags::nsw);

  // Because the number of limbs (`numLimbs`) is known at compile time, we can
  // unroll the loop as a straight-line chain of operations.
  // The result of the `i`th iteration is `T` * b⁻¹ mod n.
  for (unsigned i = 0; i < numLimbs - 1; ++i) {
    // Shift `T` right by `limbWidth`.
    Value freeCoeff = b.create<arith::AndIOp>(tLow, limbMaskConst);
    tLow = b.create<arith::ShRUIOp>(tLow, limbWidthConst);
    Value tHighLowerLimb = b.create<arith::ShLIOp>(tHigh, limbShiftConst);
    tLow = b.create<arith::OrIOp>(tLow, tHighLowerLimb);
    tHigh = b.create<arith::ShRUIOp>(tHigh, limbWidthConst);

    // Compute `m` = `freeCoeff` * (b⁻¹ mod n) and add to `T`.
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

  // When 2p > 2^w (modulus uses all w bits), the REDC result can land in
  // [2^w, 2p) which overflows w-bit storage. Detect this BEFORE the left
  // shift discards the evidence: tHigh bits above the lowest limb mean the
  // combined result would exceed 2^w.
  APInt mod = cast<IntegerAttr>(modAttr).getValue();
  Value overflow;
  if (mod.isSignBitSet()) {
    Value tHighUpper = b.create<arith::ShRUIOp>(tHigh, limbWidthConst);
    TypedAttr zeroAttr = b.getIntegerAttr(getElementTypeOrSelf(tLow), 0);
    Value zero;
    if (auto shapedType = dyn_cast<ShapedType>(tLow.getType()))
      zero = createSplatConst(b, zeroAttr, shapedType, tLow);
    else
      zero = b.create<arith::ConstantOp>(zeroAttr);
    overflow = b.create<arith::CmpIOp>(arith::CmpIPredicate::ne, tHighUpper,
                                       zero);
  }

  tHigh = b.create<arith::ShLIOp>(tHigh, limbShiftConst);
  tLow = b.create<arith::OrIOp>(tLow, tHigh);

  if (lazy)
    return tLow;
  if (overflow)
    return getCanonicalFromExtended(tLow, overflow);
  return getCanonicalFromExtended(tLow);
}

Value MontReducer::reduce(Value tLow, Value tHigh, bool lazy) {
  const unsigned numLimbs = montAttr.getNumLimbs();
  return numLimbs == 1 ? reduceSingleLimb(tLow, tHigh, lazy)
                       : reduceMultiLimb(tLow, tHigh, lazy);
}

} // namespace mlir::prime_ir::mod_arith
