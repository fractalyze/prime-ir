#include "zkir/Dialect/ModArith/IR/ModArithAttributes.h"

#include <utility>

#include "mlir/IR/BuiltinTypes.h"
#include "zkir/Utils/APIntUtils.h"

namespace mlir::zkir::mod_arith {

ModArithType MontgomeryAttr::getModType() const { return getImpl()->modType; }
IntegerAttr MontgomeryAttr::getNPrime() const { return getImpl()->nPrime; }
IntegerAttr MontgomeryAttr::getR() const { return getImpl()->r; }
IntegerAttr MontgomeryAttr::getRInv() const { return getImpl()->rInv; }
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
  size_t w = modulus.getBitWidth() - 1;
  if (modulus.getNumWords() > 1) {
    w = APInt::APINT_BITS_PER_WORD;
  }

  // `b` = 2^`w`
  APInt b = APInt::getOneBitSet(modulus.getBitWidth(), w);

  // Compute `r` = `b^l` (mod `modulus`) where `l` is the number of limbs
  APInt r = expMod(b, modulus.getNumWords(), modulus);
  APInt rInv = multiplicativeInverse(r, modulus);
  APInt rSquared = mulMod(r, r, modulus);

  // Now, we can truncate and do operations in `w + 1` bits for the `nPrime`
  // calculation
  b = b.trunc(w + 1);

  // `modulusModB` = `modulus` (mod `b`)
  APInt modulusModB = modulus.trunc(w + 1);
  modulusModB.clearHighBits(modulusModB.getBitWidth() - w);

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

  // Construct the `rSquaredAttr` with 2x the bitwidth of the modulus
  // NOTE(batzor): It is currently 2x bitwidth due to how the `ToMontOp` works
  // but should be later changed.
  IntegerAttr rSquaredAttr = IntegerAttr::get(
      IntegerType::get(modType.getContext(), modulus.getBitWidth() * 2),
      rSquared.zext(modulus.getBitWidth() * 2));

  // Construct the `nPrimeAttr` with the bitwidth `w`
  IntegerAttr nPrimeAttr = IntegerAttr::get(
      IntegerType::get(modType.getContext(), nPrime.getBitWidth()), nPrime);

  return new (allocator.allocate<MontgomeryAttrStorage>())
      MontgomeryAttrStorage(std::move(modType), std::move(nPrimeAttr),
                            std::move(rAttr), std::move(rInvAttr),
                            std::move(rSquaredAttr));
}

}  // namespace detail

}  // namespace mlir::zkir::mod_arith

#include "zkir/Dialect/ModArith/IR/ModArithAttributes.cpp.inc"
