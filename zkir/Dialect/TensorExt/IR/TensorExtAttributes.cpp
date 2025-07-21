#include "zkir/Dialect/TensorExt/IR/TensorExtAttributes.h"

#include <utility>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

// IWYU pragma: begin_keep
// Headers needed for TensorExtAttributes.cpp.inc
#include "llvm/ADT/TypeSwitch.h"
#include "zkir/Dialect/TensorExt/IR/TensorExtDialect.h"
// IWYU pragma: end_keep

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
  // clang-format off
  // NOTE(batzor): Number of identity mapped indices is 2^(indexBitWidth / 2) / 2.  // NOLINT(whitespace/line_length)
  // Since these indices are inserted twice, we add it to the reserved size.
  // clang-format on
  indices.reserve((numCoeffs + (1 << (indexBitWidth / 2))) / 2);
  for (unsigned index = 0; index < numCoeffs; index++) {
    APInt idx = APInt(indexBitWidth, index);
    APInt ridx = idx.reverseBits();
    if (idx.ule(ridx)) {
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
