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

#ifndef PRIME_IR_DIALECT_MODARITH_IR_MODARITHATTRIBUTES_H_
#define PRIME_IR_DIALECT_MODARITH_IR_MODARITHATTRIBUTES_H_

#include <utility>

#include "llvm/ADT/Hashing.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::prime_ir::mod_arith::detail {

struct MontgomeryAttrStorage : public AttributeStorage {
  using KeyTy = IntegerAttr;

  MontgomeryAttrStorage(IntegerAttr modulus, IntegerAttr nPrime,
                        IntegerAttr nInv, IntegerAttr r, IntegerAttr rInv,
                        IntegerAttr bInv, IntegerAttr rSquared,
                        SmallVector<IntegerAttr> invTwoPowers)
      : modulus(std::move(modulus)), nPrime(std::move(nPrime)),
        nInv(std::move(nInv)), r(std::move(r)), rInv(std::move(rInv)),
        bInv(std::move(bInv)), rSquared(std::move(rSquared)),
        invTwoPowers(std::move(invTwoPowers)) {}

  KeyTy getAsKey() const { return KeyTy(modulus); }

  bool operator==(const KeyTy &key) const { return key == KeyTy(modulus); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key);
  }

  static MontgomeryAttrStorage *construct(AttributeStorageAllocator &allocator,
                                          KeyTy &&key);

  IntegerAttr modulus;
  IntegerAttr nPrime;
  IntegerAttr nInv;
  IntegerAttr r;
  IntegerAttr rInv;
  IntegerAttr bInv;
  IntegerAttr rSquared;
  SmallVector<IntegerAttr> invTwoPowers;
};

struct BYAttrStorage : public AttributeStorage {
  using KeyTy = IntegerAttr;

  BYAttrStorage(IntegerAttr modulus, IntegerAttr divsteps, IntegerAttr mInv,
                IntegerAttr newBitWidth)
      : modulus(std::move(modulus)), divsteps(std::move(divsteps)),
        mInv(std::move(mInv)), newBitWidth(std::move(newBitWidth)) {}

  KeyTy getAsKey() const { return KeyTy(modulus); }

  bool operator==(const KeyTy &key) const { return key == KeyTy(modulus); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key);
  }

  static BYAttrStorage *construct(AttributeStorageAllocator &allocator,
                                  KeyTy &&key);

  IntegerAttr modulus;
  IntegerAttr divsteps;
  IntegerAttr mInv;
  IntegerAttr newBitWidth;
};

} // namespace mlir::prime_ir::mod_arith::detail

#define GET_ATTRDEF_CLASSES
#include "prime_ir/Dialect/ModArith/IR/ModArithAttributes.h.inc"

#endif // PRIME_IR_DIALECT_MODARITH_IR_MODARITHATTRIBUTES_H_
