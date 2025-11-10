#ifndef ZKIR_DIALECT_MODARITH_IR_MODARITHATTRIBUTES_H_
#define ZKIR_DIALECT_MODARITH_IR_MODARITHATTRIBUTES_H_

#include <utility>

#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::zkir::mod_arith::detail {

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

} // namespace mlir::zkir::mod_arith::detail

#define GET_ATTRDEF_CLASSES
#include "zkir/Dialect/ModArith/IR/ModArithAttributes.h.inc"

#endif // ZKIR_DIALECT_MODARITH_IR_MODARITHATTRIBUTES_H_
