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

#include "prime_ir/Dialect/ModArith/IR/ModArithAttributes.h"

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinTypes.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithOperation.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"
#include "prime_ir/Utils/APIntUtils.h"

namespace mlir::prime_ir::mod_arith {

IntegerAttr MontgomeryAttr::getModulus() const { return getImpl()->modulus; }
IntegerAttr MontgomeryAttr::getNPrime() const { return getImpl()->nPrime; }
IntegerAttr MontgomeryAttr::getNInv() const { return getImpl()->nInv; }
IntegerAttr MontgomeryAttr::getR() const { return getImpl()->r; }
IntegerAttr MontgomeryAttr::getRInv() const { return getImpl()->rInv; }
IntegerAttr MontgomeryAttr::getBInv() const { return getImpl()->bInv; }
IntegerAttr MontgomeryAttr::getRSquared() const { return getImpl()->rSquared; }
const SmallVector<IntegerAttr> &MontgomeryAttr::getInvTwoPowers() const {
  return getImpl()->invTwoPowers;
}

namespace detail {

// static
MontgomeryAttrStorage *
MontgomeryAttrStorage::construct(AttributeStorageAllocator &allocator,
                                 KeyTy &&key) {
  // Extract `modulus` from the key
  IntegerAttr modAttr = key;
  APInt modulus = modAttr.getValue();

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
  APInt modExt = modulus.zextOrTrunc(b.getBitWidth());
  APInt bReduced = b.urem(modExt);
  bReduced = bReduced.zextOrTrunc(modulus.getBitWidth());
  ModArithType modType = ModArithType::get(modAttr.getContext(), modAttr);
  ModArithOperation bReducedOp(bReduced, modType);

  // Compute `R` = `b^l` (mod `modulus`) where `l` is the number of limbs
  ModArithOperation rOp =
      bReducedOp.power(APInt(modulus.getBitWidth(), numWords));

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
  IntegerAttr rAttr = rOp.getIntegerAttr();

  // Construct the `rInvAttr` with the bitwidth of the modulus
  IntegerAttr rInvAttr = rOp.inverse().getIntegerAttr();

  // Construct the `bInvAttr` with the bitwidth of the modulus
  IntegerAttr bInvAttr = bReducedOp.inverse().getIntegerAttr();

  // Construct the `rSquaredAttr` with the bitwidth of the modulus
  IntegerAttr rSquaredAttr = rOp.square().getIntegerAttr();

  // Construct the `nPrimeAttr` with the bitwidth `w`
  IntegerAttr nPrimeAttr = IntegerAttr::get(
      IntegerType::get(modAttr.getContext(), nPrime.getBitWidth()), nPrime);

  // Construct the `nInvAttr` with the bitwidth `w`
  IntegerAttr nInvAttr = IntegerAttr::get(
      IntegerType::get(modAttr.getContext(), w), invN.trunc(w));

  // Construct the `invTwoPowersAttr` with the bitwidth `w`
  SmallVector<IntegerAttr> invTwoPowers;
  invTwoPowers.reserve(modulus.getBitWidth());
  auto invTwo =
      ModArithOperation(APInt::getOneBitSet(modulus.getBitWidth(), 1), modType)
          .inverse();
  auto cur = invTwo;
  invTwoPowers.push_back(invTwo.getIntegerAttr());
  size_t twoAdicity = (modulus - 1).countTrailingZeros();
  for (size_t i = 1; i < twoAdicity; i++) {
    cur *= invTwo;
    invTwoPowers.push_back(cur.getIntegerAttr());
  }

  return new (allocator.allocate<MontgomeryAttrStorage>())
      MontgomeryAttrStorage(std::move(modAttr), std::move(nPrimeAttr),
                            std::move(nInvAttr), std::move(rAttr),
                            std::move(rInvAttr), std::move(bInvAttr),
                            std::move(rSquaredAttr), std::move(invTwoPowers));
}

} // namespace detail

IntegerAttr BYAttr::getModulus() const { return getImpl()->modulus; }
IntegerAttr BYAttr::getDivsteps() const { return getImpl()->divsteps; }
IntegerAttr BYAttr::getMInv() const { return getImpl()->mInv; }
IntegerAttr BYAttr::getNewBitWidth() const { return getImpl()->newBitWidth; }

namespace detail {

// static
BYAttrStorage *BYAttrStorage::construct(AttributeStorageAllocator &allocator,
                                        KeyTy &&key) {
  IntegerAttr modAttr = key;
  APInt modulus = modAttr.getValue();
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

  auto intType = IntegerType::get(modAttr.getContext(), bitWidth);
  auto divstepsAttr = IntegerAttr::get(intType, divsteps);
  auto mInvAttr = IntegerAttr::get(intType, mInv);
  auto newBitWidthAttr = IntegerAttr::get(intType, bitWidth);
  return new (allocator.allocate<BYAttrStorage>())
      BYAttrStorage(std::move(modAttr), std::move(divstepsAttr),
                    std::move(mInvAttr), std::move(newBitWidthAttr));
}

} // namespace detail
} // namespace mlir::prime_ir::mod_arith

#include "prime_ir/Dialect/ModArith/IR/ModArithAttributes.cpp.inc"
