#include "zkir/Dialect/TensorExt/IR/TensorExtAttributes.h"

#include <utility>

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::zkir::tensor_ext {

IntegerAttr BitReverseIndicesAttr::getBitWidth() const {
  return getImpl()->bitWidth;
}
DenseIntElementsAttr BitReverseIndicesAttr::getIndices() const {
  return getImpl()->indices;
}
Type BitReverseIndicesAttr::getIndicesType() const {
  return getImpl()->indices.getType();
}

namespace detail {

BitReverseIndicesAttrStorage *BitReverseIndicesAttrStorage::construct(
    AttributeStorageAllocator &allocator, KeyTy &&key) {
  IntegerAttr bitWidthAttr = std::get<0>(key);
  unsigned indexBitWidth = bitWidthAttr.getValue().getZExtValue();
  unsigned numCoeffs = 1 << indexBitWidth;
  SmallVector<APInt> indices;
  indices.reserve((numCoeffs - (1 << (indexBitWidth / 2))) / 2);
  for (unsigned index = 0; index < numCoeffs; index++) {
    APInt idx = APInt(indexBitWidth, index);
    APInt ridx = idx.reverseBits();
    if (idx.ult(ridx)) {
      indices.push_back(idx);
      indices.push_back(ridx);
    }
  }
  llvm::SmallVector<int64_t> indicesShape = {
      static_cast<int64_t>(indices.size())};
  auto indicesType = RankedTensorType::get(
      indicesShape, IndexType::get(bitWidthAttr.getContext()));
  auto indicesAttr = DenseIntElementsAttr::get(indicesType, indices);
  return new (allocator.allocate<BitReverseIndicesAttrStorage>())
      BitReverseIndicesAttrStorage(std::move(bitWidthAttr),
                                   std::move(indicesAttr));
}

}  // namespace detail
}  // namespace mlir::zkir::tensor_ext

#define GET_ATTRDEF_CLASSES
#include "zkir/Dialect/TensorExt/IR/TensorExtAttributes.cpp.inc"
