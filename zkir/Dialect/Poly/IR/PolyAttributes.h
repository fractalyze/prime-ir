#ifndef ZKIR_DIALECT_POLY_IR_POLYATTRIBUTES_H_
#define ZKIR_DIALECT_POLY_IR_POLYATTRIBUTES_H_

#include <tuple>
#include <utility>

#include "mlir/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "zkir/Dialect/Field/IR/FieldAttributes.h"
#include "zkir/Dialect/ModArith/IR/ModArithAttributes.h"
#include "zkir/Dialect/Poly/IR/PolyDialect.h"
#include "zkir/Dialect/Poly/IR/PolyTypes.h"

namespace mlir::zkir::poly::detail {

struct PrimitiveRootAttrStorage : public AttributeStorage {
  using KeyTy = std::tuple<zkir::field::PrimeFieldAttr, IntegerAttr,
                           mod_arith::MontgomeryAttr>;
  PrimitiveRootAttrStorage(IntegerAttr degree,
                           zkir::field::PrimeFieldAttr invDegree,
                           zkir::field::PrimeFieldAttr root,
                           zkir::field::PrimeFieldAttr invRoot,
                           DenseElementsAttr roots, DenseElementsAttr invRoots,
                           zkir::mod_arith::MontgomeryAttr montgomery)
      : degree(std::move(degree)),
        invDegree(std::move(invDegree)),
        root(std::move(root)),
        invRoot(std::move(invRoot)),
        roots(std::move(roots)),
        invRoots(std::move(invRoots)),
        montgomery(std::move(montgomery)) {}

  KeyTy getAsKey() const { return KeyTy(root, degree, montgomery); }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(root, degree, montgomery);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key),
                              std::get<2>(key));
  }

  static PrimitiveRootAttrStorage *construct(
      AttributeStorageAllocator &allocator, KeyTy &&key);

  IntegerAttr degree;
  zkir::field::PrimeFieldAttr invDegree;
  zkir::field::PrimeFieldAttr root;
  zkir::field::PrimeFieldAttr invRoot;
  DenseElementsAttr roots;
  DenseElementsAttr invRoots;
  zkir::mod_arith::MontgomeryAttr montgomery;
};

}  // namespace mlir::zkir::poly::detail

#define GET_ATTRDEF_CLASSES
#include "zkir/Dialect/Poly/IR/PolyAttributes.h.inc"

#endif  // ZKIR_DIALECT_POLY_IR_POLYATTRIBUTES_H_
