#ifndef ZKIR_DIALECT_TENSOREXT_IR_TENSOREXTATTRIBUTES_H_
#define ZKIR_DIALECT_TENSOREXT_IR_TENSOREXTATTRIBUTES_H_

#include <tuple>
#include <utility>

#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::zkir::tensor_ext::detail {

struct BitReverseIndicesAttrStorage : public AttributeStorage {
  using KeyTy = std::tuple<IntegerAttr>;
  BitReverseIndicesAttrStorage(IntegerAttr bitWidth,
                               DenseIntElementsAttr indices)
      : bitWidth(std::move(bitWidth)), indices(std::move(indices)) {}

  KeyTy getAsKey() const { return KeyTy(bitWidth); }

  bool operator==(const KeyTy &key) const { return key == KeyTy(bitWidth); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key));
  }

  static BitReverseIndicesAttrStorage *construct(
      AttributeStorageAllocator &allocator, KeyTy &&key);

  IntegerAttr bitWidth;
  DenseIntElementsAttr indices;
};

}  // namespace mlir::zkir::tensor_ext::detail

#define GET_ATTRDEF_CLASSES
#include "zkir/Dialect/TensorExt/IR/TensorExtAttributes.h.inc"

#endif  // ZKIR_DIALECT_TENSOREXT_IR_TENSOREXTATTRIBUTES_H_
