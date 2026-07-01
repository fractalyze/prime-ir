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
    return arith::ConstantOp::create(
        b, SplatElementsAttr::get(shapedType, scalarAttr));
  }
  assert(shapeRef &&
         "A shape reference value must be provided for dynamic shapes.");
  Value scalar = arith::ConstantOp::create(b, scalarAttr);
  SmallVector<Value> dynamicDims;
  for (int64_t i = 0; i < shapedType.getRank(); ++i) {
    if (shapedType.isDynamicDim(i)) {
      auto idx = arith::ConstantIndexOp::create(b, i);
      dynamicDims.push_back(tensor::DimOp::create(b, shapeRef, idx));
    }
  }
  Value empty = tensor::EmptyOp::create(b, shapedType, dynamicDims);
  return linalg::FillOp::create(b, scalar, empty).getResult(0);
}

} // namespace

Value MontReducer::createModulusConst(Type inputType, Value inputValue) {
  if (auto shapedType = dyn_cast<ShapedType>(inputType)) {
    if (!isa<VectorType>(this->modAttr.getType())) {
      return createSplatConst(b, this->modAttr, shapedType, inputValue);
    }
  }
  return arith::ConstantOp::create(b, this->modAttr);
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
      threshConst = arith::ConstantOp::create(b, multipleAttr);

    auto sub = arith::SubIOp::create(b, input, threshConst);
    input = arith::MinUIOp::create(b, sub, input).getResult();
  }
  return input;
}

Value MontReducer::getCanonicalFromExtended(Value input, Value overflow) {
  auto cmod = createModulusConst(input.getType(), input);
  // Canonicalize the value `overflow·2^w + input` (input ∈ [0, 2p), so one
  // subtract of p suffices) in 3 ALU ops instead of 4.
  //
  // `min(input - p, input)` folds the `input ≥ p` case without a compare: when
  // input < p, `input - p` wraps to `input + (2^w - p) > input` (and < 2^w, no
  // double wrap), so min keeps input; when input ≥ p, `input - p < input`, so
  // min picks it. A carry into bit w (overflow) always means subtract p, so the
  // select forces that branch. Byte-identical to the prior
  // `(input >= p || overflow) ? input - p : input` for every input and any
  // modulus: overflow ⟹ both yield `input - p`; otherwise min reproduces the
  // compare. Drops the `cmpi` + `ori` for a single `minui`. The subtract form
  // is what makes min safe here — getCanonicalDiff adds p and so cannot.
  auto sub = arith::SubIOp::create(b, input, cmod);
  auto min = arith::MinUIOp::create(b, sub, input);
  auto select = arith::SelectOp::create(b, overflow, sub, min);
  return select.getResult();
}

Value MontReducer::getCanonicalDiff(Value lhs, Value rhs) {
  auto cmod = createModulusConst(lhs.getType(), lhs);
  auto sub = arith::SubIOp::create(b, lhs, rhs);
  auto add = arith::AddIOp::create(b, sub, cmod);
  APInt mod = cast<IntegerAttr>(modAttr).getValue();
  if (mod.isSignBitSet()) {
    // When p > 2ʷ⁻¹, diff + p can overflow, so minui gives wrong results.
    // Fall back to cmpi + select.
    auto underflowed =
        arith::CmpIOp::create(b, arith::CmpIPredicate::ult, lhs, rhs);
    return arith::SelectOp::create(b, underflowed, add, sub).getResult();
  }
  return arith::MinUIOp::create(b, sub, add).getResult();
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
    nInvConst = arith::ConstantOp::create(b, nInvAttr);
  }
  auto modConst = createModulusConst(tLow.getType(), tLow);

  // Compute m = tLow * nInv (mod base).
  auto m = arith::MulIOp::create(b, tLow, nInvConst);
  // Compute m * n.
  Value mNHigh;
  if (isFromSignedMul(tLow)) {
    auto mN = arith::MulSIExtendedOp::create(b, m, modConst);
    mNHigh = mN.getHigh();
  } else {
    auto mN = arith::MulUIExtendedOp::create(b, m, modConst);
    mNHigh = mN.getHigh();
  }

  // The low part of T - mN is always zero (divisible by base), so the
  // result is just tHigh - mNHigh mod n.
  if (lazy) {
    // tHigh - mNHigh can underflow, so unconditionally add p.
    // Result is in [0, 2p).
    auto sub = arith::SubIOp::create(b, tHigh, mNHigh);
    return arith::AddIOp::create(b, sub, modConst).getResult();
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
  auto nPrimeConst = arith::ConstantOp::create(b, nPrimeAttr);
  auto limbWidthConst = arith::ConstantOp::create(b, limbWidthAttr);
  auto limbMaskConst = arith::ConstantOp::create(b, limbMaskAttr);
  auto limbShiftConst = arith::ConstantOp::create(b, limbShiftAttr);
  auto modConst = arith::ConstantOp::create(b, modAttrLocal);
  auto bInvConst = arith::ConstantOp::create(b, bInvAttr);
  auto oneConst = arith::ConstantOp::create(b, oneAttr);

  auto noOverflow = arith::IntegerOverflowFlagsAttr::get(
      b.getContext(),
      arith::IntegerOverflowFlags::nuw | arith::IntegerOverflowFlags::nsw);

  // Because the number of limbs (`numLimbs`) is known at compile time, we can
  // unroll the loop as a straight-line chain of operations.
  // The result of the `i`th iteration is `T` * b⁻¹ mod n.
  for (unsigned i = 0; i < numLimbs - 1; ++i) {
    // Shift `T` right by `limbWidth`.
    Value freeCoeff = arith::AndIOp::create(b, tLow, limbMaskConst);
    tLow = arith::ShRUIOp::create(b, tLow, limbWidthConst);
    Value tHighLowerLimb = arith::ShLIOp::create(b, tHigh, limbShiftConst);
    tLow = arith::OrIOp::create(b, tLow, tHighLowerLimb);
    tHigh = arith::ShRUIOp::create(b, tHigh, limbWidthConst);

    // Compute `m` = `freeCoeff` * (b⁻¹ mod n) and add to `T`.
    auto m = arith::MulUIExtendedOp::create(b, freeCoeff, bInvConst);
    tLow = arith::AddIOp::create(b, tLow, m.getLow());
    Value carry =
        arith::CmpIOp::create(b, arith::CmpIPredicate::ult, tLow, m.getLow());
    tHigh = arith::AddIOp::create(b, tHigh, m.getHigh(), noOverflow);
    auto tHighPlusOne = arith::AddIOp::create(b, tHigh, oneConst, noOverflow);
    tHigh = arith::SelectOp::create(b, carry, tHighPlusOne, tHigh);
  }
  // The last iteration is the same as normal Montgomery reduction.
  Value freeCoeff = arith::TruncIOp::create(b, limbType, tLow);
  // Compute `m` = `freeCoeff` * `nPrime` (mod `base`)
  auto m = arith::MulIOp::create(b, freeCoeff, nPrimeConst);
  // Compute `m` * `n`
  Value mExt = arith::ExtUIOp::create(b, tLow.getType(), m);
  auto mN = arith::MulUIExtendedOp::create(b, modConst, mExt);
  // Add the product to `T`.
  tLow = arith::AddIOp::create(b, tLow, mN.getLow());
  auto carry =
      arith::CmpIOp::create(b, arith::CmpIPredicate::ult, tLow, mN.getLow());
  tHigh = arith::AddIOp::create(b, tHigh, mN.getHigh(), noOverflow);
  auto tHighPlusOne = arith::AddIOp::create(b, tHigh, oneConst, noOverflow);
  tHigh = arith::SelectOp::create(b, carry, tHighPlusOne, tHigh);
  // Shift right `T` by `limbWidth` to discard the zeroed limb.
  tLow = arith::ShRUIOp::create(b, tLow, limbWidthConst);

  // When p > 2ʷ⁻¹ (modulus uses all w bits), the REDC result can exceed
  // 2ʷ because 2p > 2ʷ. The upper bits of tHigh (above the lowest limb)
  // represent the overflow: real_value = tLow_combined + tHighUpper * 2ʷ.
  // Since 2ʷ ≡ R (mod p) where R = 2ʷ mod p, we recover the correct value
  // by adding tHighUpper * R to tLow, then conditionally subtracting p.
  APInt mod = cast<IntegerAttr>(modAttr).getValue();
  if (mod.isSignBitSet()) {
    Value tHighUpper = arith::ShRUIOp::create(b, tHigh, limbWidthConst);
    // Mask tHigh to only the lowest limb before the left-shift.
    tHigh = arith::AndIOp::create(b, tHigh, limbMaskConst);
    tHigh = arith::ShLIOp::create(b, tHigh, limbShiftConst);
    tLow = arith::OrIOp::create(b, tLow, tHigh);

    // Add overflow * R to recover the truncated value mod p.
    // R = 2ʷ mod p is small, so tHighUpper * R fits in w bits.
    // R = 2ʷ mod p = 2ʷ - p. Compute without overflow: -p in w-bit wraps
    // to 2ʷ - p since p < 2ʷ.
    APInt rVal = -mod; // R = 2ʷ - p (unsigned wrap in w bits)
    TypedAttr rAttr = b.getIntegerAttr(getElementTypeOrSelf(tLow), rVal);
    Value rConst;
    if (auto shapedType = dyn_cast<ShapedType>(tLow.getType()))
      rConst = createSplatConst(b, rAttr, shapedType, tLow);
    else
      rConst = arith::ConstantOp::create(b, rAttr);
    Value correction = arith::MulIOp::create(b, tHighUpper, rConst);
    tLow = arith::AddIOp::create(b, tLow, correction);

    return getCanonicalFromExtended(tLow);
  }

  tHigh = arith::ShLIOp::create(b, tHigh, limbShiftConst);
  tLow = arith::OrIOp::create(b, tLow, tHigh);

  if (lazy)
    return tLow;
  // Final conditional subtraction: if (`tLow` >= `modulus`) then subtract
  // `modulus`.
  return getCanonicalFromExtended(tLow);
}

Value MontReducer::reduce(Value tLow, Value tHigh, bool lazy) {
  // Lazy REDC returns [0, 2p), which requires 2p ≤ 2ʷ. When the modulus
  // uses all w bits (p > 2ʷ⁻¹), 2p > 2ʷ and lazy is not representable.
  APInt mod = cast<IntegerAttr>(modAttr).getValue();
  if (mod.isSignBitSet()) {
    // Lazy is not possible when 2p > 2ʷ because [0, 2p) does not fit in
    // w bits. Always reduce to [0, p).
    assert(!lazy &&
           "lazy REDC not supported for primes using all w bits (2p > 2ʷ)");
  }
  const unsigned numLimbs = montAttr.getNumLimbs();
  return numLimbs == 1 ? reduceSingleLimb(tLow, tHigh, lazy)
                       : reduceMultiLimb(tLow, tHigh, lazy);
}

} // namespace mlir::prime_ir::mod_arith
