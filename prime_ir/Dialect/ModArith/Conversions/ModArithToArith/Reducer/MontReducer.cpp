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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h" // IWYU pragma: keep

namespace mlir::prime_ir::mod_arith {

MontReducer::MontReducer(ImplicitLocOpBuilder &b, ModArithType modArithType)
    : b(b), modAttr(modArithType.getModulus()),
      montAttr(modArithType.getMontgomeryAttr()) {}

Value MontReducer::createModulusConst(Type inputType) {
  TypedAttr modAttr = this->modAttr;
  if (auto shapedType = dyn_cast<ShapedType>(inputType)) {
    if (!isa<VectorType>(this->modAttr.getType())) {
      modAttr = SplatElementsAttr::get(shapedType, this->modAttr);
    }
  }
  return b.create<arith::ConstantOp>(modAttr);
}

Value MontReducer::getCanonicalFromExtended(Value input) {
  auto cmod = createModulusConst(input.getType());
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

Value MontReducer::getCanonicalFromExtended(Value input, Value overflow) {
  auto cmod = createModulusConst(input.getType());
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
  auto cmod = createModulusConst(lhs.getType());
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

bool MontReducer::isFromSignedMul(Value input) {
  auto signedOp = input.getDefiningOp<arith::MulSIExtendedOp>();
  return signedOp && signedOp.getLhs() != signedOp.getRhs();
}

Value MontReducer::reduceSingleLimb(Value tLow, Value tHigh) {
  // Prepare nInv constant.
  TypedAttr nInvAttr = montAttr.getNInv();
  Type limbType = nInvAttr.getType();
  if (auto shapedType = dyn_cast<ShapedType>(tLow.getType())) {
    limbType = shapedType.cloneWith(std::nullopt, limbType);
    nInvAttr = SplatElementsAttr::get(cast<ShapedType>(limbType), nInvAttr);
  }
  auto nInvConst = b.create<arith::ConstantOp>(nInvAttr);

  // Prepare modulus constant.
  auto modConst = createModulusConst(tLow.getType());

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

  // Calculate `T` - `mN`, which should result in zeroed low limb since it
  // should be divisible by `base`. The low part subtraction cannot
  // underflow since if `tLow` < `mN.getLow()`, then `tLow` -
  // `mN.getLow()` cannot result in zero low limb. This means, `tLow` is
  // always equal to `mN.getLow()` so we can skip the low subtractions.
  // The reduction result will be just `tHigh` - `mN.getHigh()` mod n.
  auto sub = getCanonicalDiff(tHigh, mNHigh);
  return sub;
}

Value MontReducer::reduceMultiLimb(Value tLow, Value tHigh) {
  // Extract Montgomery constants.
  TypedAttr nPrimeAttr = montAttr.getNPrime();
  TypedAttr bInvAttr = montAttr.getBInv();

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
  tHigh = b.create<arith::ShLIOp>(tHigh, limbShiftConst);
  tLow = b.create<arith::OrIOp>(tLow, tHigh);

  // Final conditional subtraction: if (`tLow` >= `modulus`) then subtract
  // `modulus`.
  auto result = getCanonicalFromExtended(tLow);
  return result;
}

Value MontReducer::reduce(Value tLow, Value tHigh) {
  // Determine the number of limbs to choose the appropriate reduction strategy.
  TypedAttr nPrimeAttr = montAttr.getNPrime();
  const unsigned modBitWidth =
      cast<IntegerType>(getElementTypeOrSelf(modAttr.getType())).getWidth();
  const unsigned limbWidth = nPrimeAttr.getType().getIntOrFloatBitWidth();
  const unsigned numLimbs = (modBitWidth + limbWidth - 1) / limbWidth;

  // If the number of limbs is 1, the 2ʷ is larger than the modulus, so we
  // can use `nInv` instead of `nPrime` and avoid carry check.
  if (numLimbs == 1) {
    return reduceSingleLimb(tLow, tHigh);
  }

  return reduceMultiLimb(tLow, tHigh);
}

} // namespace mlir::prime_ir::mod_arith
