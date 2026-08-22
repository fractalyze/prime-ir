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
  // Canonicalize `overflow·2^w + input` (input ∈ [0, 2p)) in 3 ALU ops, not 4.
  // `min(input - p, input)` picks input when input < p (the subtract wraps up)
  // and input - p when input ≥ p — folding the compare into a minui; overflow
  // forces the subtract branch. Byte-identical to the old
  // `(input >= p || overflow) ? input - p : input`, minus the cmpi+ori. Uses
  // subtract-of-p, not getCanonicalDiff's add-of-p, so min stays safe.
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

  // Compute number of limbs.
  const unsigned limbWidth = montAttr.getLimbWidth();
  const unsigned numLimbs = montAttr.getNumLimbs();

  Type limbType = nPrimeAttr.getType();
  TypedAttr limbWidthAttr =
      b.getIntegerAttr(getElementTypeOrSelf(tLow), limbWidth);
  TypedAttr limbShiftAttr =
      b.getIntegerAttr(getElementTypeOrSelf(tLow), (numLimbs - 1) * limbWidth);

  TypedAttr modAttrLocal = modAttr;

  // Splat the attributes to match the shape of `tLow`.
  if (auto shapedType = dyn_cast<ShapedType>(tLow.getType())) {
    limbType = shapedType.cloneWith(std::nullopt, limbType);
    nPrimeAttr = SplatElementsAttr::get(cast<ShapedType>(limbType), nPrimeAttr);
    limbWidthAttr = SplatElementsAttr::get(shapedType, limbWidthAttr);
    limbShiftAttr = SplatElementsAttr::get(shapedType, limbShiftAttr);
    modAttrLocal = isa<VectorType>(modAttrLocal.getType())
                       ? modAttrLocal
                       : SplatElementsAttr::get(shapedType, modAttrLocal);
  }

  // Create constants for the Montgomery reduction.
  auto nPrimeConst = arith::ConstantOp::create(b, nPrimeAttr);
  auto limbWidthConst = arith::ConstantOp::create(b, limbWidthAttr);
  auto limbShiftConst = arith::ConstantOp::create(b, limbShiftAttr);
  auto modConst = arith::ConstantOp::create(b, modAttrLocal);

  auto noOverflow = arith::IntegerOverflowFlagsAttr::get(
      b.getContext(),
      arith::IntegerOverflowFlags::nuw | arith::IntegerOverflowFlags::nsw);

  // Standard per-limb REDC, unrolled (`numLimbs` is a compile-time constant).
  // Each iteration adds `m·n` with `m = (T mod b)·nPrime mod b` — the unique
  // multiple of `n` that zeroes T's lowest limb — then shifts T right one
  // limb. The result is V = (T + n·Σᵢ mᵢbⁱ)/2ʷ < T/2ʷ + n ≤ 2n under the
  // REDC precondition T < n·2ʷ (callers pre-reduce operands to guarantee it).
  //
  // Do NOT replace the per-iteration `m·n` with the historical shift-then-add
  // `t₀·(b⁻¹ mod n)` variant: its result bound is V < T/2ʷ + bInv + n, and
  // bInv is an arbitrary constant in [0, n). For moduli where bInv is large
  // (e.g. the BLS12-381 scalar field, bInv ≈ n) V exceeds both 2n and 2ʷ,
  // which silently truncates and breaks the [0, 2n) result contract that the
  // lazy-reduction bound tracking and the final conditional subtraction rely
  // on. That variant shipped and miscompiled (p-1)² to 0 instead of 1 for
  // every full-width modulus (fractalyze/xla#542); the case is pinned by
  // tests/Dialect/ModArith/mod_arith_secp256k1_runner.mlir.
  for (unsigned i = 0; i < numLimbs; ++i) {
    // m = (T mod b) * nPrime (mod b).
    Value freeCoeff = arith::TruncIOp::create(b, limbType, tLow);
    auto m = arith::MulIOp::create(b, freeCoeff, nPrimeConst);
    Value mExt = arith::ExtUIOp::create(b, tLow.getType(), m);
    auto mN = arith::MulUIExtendedOp::create(b, modConst, mExt);

    // T += m * n. Fold the carry out of tLow into mN's high limb first:
    // mN.high = floor(m·n / 2ʷ) ≤ b - 2 (m ≤ b-1, n < 2ʷ), so adding the carry
    // cannot wrap.
    auto add = arith::AddUIExtendedOp::create(b, tLow, mN.getLow());
    tLow = add.getSum();
    Value carryExt =
        arith::ExtUIOp::create(b, tLow.getType(), add.getOverflow());
    Value mNHigh = arith::AddIOp::create(b, mN.getHigh(), carryExt, noOverflow);

    Value carryOut; // Bit 2w of T; reachable only in the first iteration.
    if (i == 0) {
      // In the first iteration tHigh can be as large as n - 1, so adding
      // mN.high can carry out of the w-bit register when n is within
      // 2^limbWidth of 2ʷ (e.g. secp256k1). Capture the carry-out; it is
      // folded back in after the limb shift below. From the second iteration
      // on, T < n·2ʷ/b + n·b implies tHigh < 2^(w-limbWidth) + 1 and the add
      // cannot wrap.
      auto sum = arith::AddUIExtendedOp::create(b, tHigh, mNHigh);
      tHigh = sum.getSum();
      carryOut = sum.getOverflow();
    } else {
      tHigh = arith::AddIOp::create(b, tHigh, mNHigh, noOverflow);
    }

    // T's lowest limb is now zero; shift T right by one limb.
    tLow = arith::ShRUIOp::create(b, tLow, limbWidthConst);
    Value tHighLowerLimb = arith::ShLIOp::create(b, tHigh, limbShiftConst);
    tLow = arith::OrIOp::create(b, tLow, tHighLowerLimb);
    tHigh = arith::ShRUIOp::create(b, tHigh, limbWidthConst);
    if (i == 0) {
      Value carryExt = arith::ExtUIOp::create(b, tLow.getType(), carryOut);
      Value carryShifted =
          arith::ShLIOp::create(b, carryExt, limbShiftConst, noOverflow);
      tHigh = arith::OrIOp::create(b, tHigh, carryShifted);
    }
  }

  // Here V = tLow + tHigh·2ʷ with V < 2n (classical REDC bound above).
  APInt mod = cast<IntegerAttr>(modAttr).getValue();
  if (mod.isSignBitSet()) {
    // Full-width modulus (n > 2ʷ⁻¹): V < 2n can exceed 2ʷ, so tHigh holds
    // V's 2ʷ bit (0 or 1). [0, 2n) does not fit in w bits, so a lazy result
    // is not representable; always canonicalize, folding the overflow bit
    // (the wrapped subtract in getCanonicalFromExtended yields
    // tLow + 2ʷ - n = V - n < n exactly when the bit is set).
    // tHigh is 0 or 1, so its low bit *is* the overflow bit — a truncation
    // rather than a compare against a materialized w-bit zero.
    Type overflowType = b.getI1Type();
    if (auto shapedType = dyn_cast<ShapedType>(tHigh.getType()))
      overflowType = shapedType.cloneWith(std::nullopt, overflowType);
    Value overflow = arith::TruncIOp::create(b, overflowType, tHigh);
    return getCanonicalFromExtended(tLow, overflow);
  }

  // Narrow modulus (n < 2ʷ⁻¹): V < 2n < 2ʷ, so tHigh == 0 and tLow already
  // holds V in the lazy range [0, 2n).
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
