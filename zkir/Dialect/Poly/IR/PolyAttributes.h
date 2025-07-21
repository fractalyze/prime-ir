#ifndef ZKIR_DIALECT_POLY_IR_POLYATTRIBUTES_H_
#define ZKIR_DIALECT_POLY_IR_POLYATTRIBUTES_H_

#include <tuple>
#include <utility>

#include "zkir/Dialect/Field/IR/FieldAttributes.h"
#include "zkir/Dialect/ModArith/IR/ModArithAttributes.h"

namespace mlir::zkir::poly::detail {

struct PrimitiveRootAttrStorage : public AttributeStorage {
  using KeyTy = std::tuple<field::RootOfUnityAttr, mod_arith::MontgomeryAttr>;
  PrimitiveRootAttrStorage(field::RootOfUnityAttr rootOfUnity,
                           field::PrimeFieldAttr root,
                           field::PrimeFieldAttr invDegree,
                           field::PrimeFieldAttr invRoot,
                           DenseElementsAttr roots, DenseElementsAttr invRoots,
                           mod_arith::MontgomeryAttr montgomery)
      : rootOfUnity(std::move(rootOfUnity)),
        root(std::move(root)),
        invDegree(std::move(invDegree)),
        invRoot(std::move(invRoot)),
        roots(std::move(roots)),
        invRoots(std::move(invRoots)),
        montgomery(std::move(montgomery)) {}

  KeyTy getAsKey() const { return KeyTy(rootOfUnity, montgomery); }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(rootOfUnity, montgomery);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
  }

  static PrimitiveRootAttrStorage *construct(
      AttributeStorageAllocator &allocator, KeyTy &&key);

  field::RootOfUnityAttr rootOfUnity;
  field::PrimeFieldAttr root;
  field::PrimeFieldAttr invDegree;
  field::PrimeFieldAttr invRoot;
  DenseElementsAttr roots;
  DenseElementsAttr invRoots;
  mod_arith::MontgomeryAttr montgomery;
};

}  // namespace mlir::zkir::poly::detail

#define GET_ATTRDEF_CLASSES
#include "zkir/Dialect/Poly/IR/PolyAttributes.h.inc"

#endif  // ZKIR_DIALECT_POLY_IR_POLYATTRIBUTES_H_
