#include "zkir/Dialect/ModArith/IR/ModArithAttributes.h"

#include <utility>

#include "mlir/IR/BuiltinTypes.h"
#include "zkir/Utils/APIntUtils.h"

namespace mlir::zkir::mod_arith {

ModArithType MontgomeryAttr::getModType() const { return getImpl()->modType; }
IntegerAttr MontgomeryAttr::getNPrime() const { return getImpl()->nPrime; }
IntegerAttr MontgomeryAttr::getR() const { return getImpl()->r; }
IntegerAttr MontgomeryAttr::getRInv() const { return getImpl()->rInv; }
IntegerAttr MontgomeryAttr::getBInv() const { return getImpl()->bInv; }
IntegerAttr MontgomeryAttr::getRSquared() const { return getImpl()->rSquared; }

namespace detail {

MontgomeryAttrStorage *MontgomeryAttrStorage::construct(
    AttributeStorageAllocator &allocator, KeyTy &&key) {
  // Extract the `modType` and `modulus` from the key
  ModArithType modType = key;
  APInt modulus = modType.getModulus().getValue();

  // `w` is single limb size when `modulus` is a multi-precision
  // NOTE(batzor): In the single-precision case, we use modulus.getBitWidth() -
  // 1 to have `b` with the same bitwidth as the `modulus`. This is fine since
  // we already use 1 extra bit for the `modulus` storage to handle overflows
  // in addition operation.
  size_t numWords = modulus.getNumWords();
  size_t w = numWords > 1 ? APInt::APINT_BITS_PER_WORD : modulus.getBitWidth();

  // `b` = 2^`w`
  APInt b = APInt::getOneBitSet(w + 1, w);

  // `bReduced` = `b` (mod `modulus`)
  // `bInv` = `b`⁻¹ mod `modulus`
  APInt modExt = modulus.zextOrTrunc(b.getBitWidth());
  APInt bReduced = b.urem(modExt);
  bReduced = bReduced.zextOrTrunc(modulus.getBitWidth());
  APInt bInv = multiplicativeInverse(bReduced, modulus);

  // Compute `R` = `b^l` (mod `modulus`) where `l` is the number of limbs
  APInt r = expMod(bReduced, numWords, modulus);
  APInt rInv = multiplicativeInverse(r, modulus);
  APInt rSquared = mulMod(r, r, modulus);

  // `modulusModB` = `modulus` (mod `b`)
  APInt modulusModB = modulus.zextOrTrunc(b.getBitWidth());
  if (modulus.getBitWidth() > w) {
    modulusModB.clearBit(w);
  }
  // Compute the multiplicative inverse of `modulus` (mod `b`)
  APInt invN = multiplicativeInverse(modulusModB, b);

  // Compute `nPrime` = -`invN` (mod `b`) = `b` - `invN`
  APInt nPrime = b - invN;
  // Truncate from `w + 1` bits to `w` bits
  nPrime = nPrime.trunc(w);

  // Construct the `rAttr` with the bitwidth of the modulus
  IntegerAttr rAttr = IntegerAttr::get(modType.getModulus().getType(), r);

  // Construct the `rInvAttr` with the bitwidth of the modulus
  IntegerAttr rInvAttr = IntegerAttr::get(modType.getModulus().getType(), rInv);

  // Construct the `bInvAttr` with the bitwidth of the modulus
  IntegerAttr bInvAttr = IntegerAttr::get(modType.getModulus().getType(), bInv);

  // Construct the `rSquaredAttr` with the bitwidth of the modulus
  IntegerAttr rSquaredAttr =
      IntegerAttr::get(modType.getModulus().getType(), rSquared);

  // Construct the `nPrimeAttr` with the bitwidth `w`
  IntegerAttr nPrimeAttr = IntegerAttr::get(
      IntegerType::get(modType.getContext(), nPrime.getBitWidth()), nPrime);

  return new (allocator.allocate<MontgomeryAttrStorage>())
      MontgomeryAttrStorage(std::move(modType), std::move(nPrimeAttr),
                            std::move(rAttr), std::move(rInvAttr),
                            std::move(bInvAttr), std::move(rSquaredAttr));
}

}  // namespace detail

ModArithType BYAttr::getModType() const { return getImpl()->modType; }
IntegerAttr BYAttr::getDivsteps() const { return getImpl()->divsteps; }
IntegerAttr BYAttr::getMInv() const { return getImpl()->mInv; }
IntegerAttr BYAttr::getNewBitWidth() const { return getImpl()->newBitWidth; }

namespace detail {

// static
BYAttrStorage *BYAttrStorage::construct(AttributeStorageAllocator &allocator,
                                        KeyTy &&key) {
  ModArithType modType = key;
  APInt modulus = modType.getModulus().getValue();
  unsigned bitWidth = modulus.getBitWidth();
  // `divsteps` determine the number of steps we will batch into one jump step.
  // Assuming 12 batches, divsteps * 12 should exceed the bound, which is a
  // sufficient number of total iterations to calculate GCD.
  // Refer to Section 12 of Bernstein and Yang's Fast constant-time gcd
  // computation and modular inversion
  // (https://gcd.cr.yp.to/safegcd-20190305.pdf).
  unsigned bound =
      bitWidth < 46 ? (49 * bitWidth + 80) / 17 : (49 * bitWidth + 57) / 17;
  unsigned divsteps = 0;
  while (12 * divsteps <= bound) {
    divsteps++;
  }

  // need one extra limb to properly store the intermediate results
  bitWidth += APInt::APINT_BITS_PER_WORD;

  APInt mask = APInt::getOneBitSet(bitWidth, divsteps);
  APInt mInv =
      multiplicativeInverse(modulus.zextOrTrunc(bitWidth).urem(mask), mask);

  auto intType = IntegerType::get(modType.getContext(), bitWidth);
  auto divstepsAttr = IntegerAttr::get(intType, divsteps);
  auto mInvAttr = IntegerAttr::get(intType, mInv);
  auto newBitWidthAttr = IntegerAttr::get(intType, bitWidth);
  return new (allocator.allocate<BYAttrStorage>())
      BYAttrStorage(std::move(modType), std::move(divstepsAttr),
                    std::move(mInvAttr), std::move(newBitWidthAttr));
}

}  // namespace detail
}  // namespace mlir::zkir::mod_arith

#include "zkir/Dialect/ModArith/IR/ModArithAttributes.cpp.inc"
