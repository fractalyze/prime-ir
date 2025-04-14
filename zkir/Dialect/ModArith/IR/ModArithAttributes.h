#ifndef ZKIR_DIALECT_MODARITH_IR_MODARITHATTRIBUTES_H_
#define ZKIR_DIALECT_MODARITH_IR_MODARITHATTRIBUTES_H_

#include <utility>

#include "mlir/IR/AttributeSupport.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"

namespace mlir::zkir::mod_arith::detail {

struct MontgomeryAttrStorage : public AttributeStorage {
  using KeyTy = ModArithType;

  MontgomeryAttrStorage(ModArithType modType, IntegerAttr nPrime, IntegerAttr r,
                        IntegerAttr rInv, IntegerAttr rSquared)
      : modType(std::move(modType)),
        nPrime(std::move(nPrime)),
        r(std::move(r)),
        rInv(std::move(rInv)),
        rSquared(std::move(rSquared)) {}

  KeyTy getAsKey() const { return KeyTy(modType); }

  bool operator==(const KeyTy &key) const { return key == KeyTy(modType); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key);
  }

  static MontgomeryAttrStorage *construct(AttributeStorageAllocator &allocator,
                                          KeyTy &&key);

  ModArithType modType;
  IntegerAttr nPrime;
  IntegerAttr r;
  IntegerAttr rInv;
  IntegerAttr rSquared;
};

}  // namespace mlir::zkir::mod_arith::detail

#define GET_ATTRDEF_CLASSES
#include "zkir/Dialect/ModArith/IR/ModArithAttributes.h.inc"

#endif  // ZKIR_DIALECT_MODARITH_IR_MODARITHATTRIBUTES_H_
