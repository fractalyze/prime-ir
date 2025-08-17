#include "zkir/Utils/ShapedTypeConverter.h"

#include <assert.h>

namespace mlir::zkir {

// static
Type ShapedTypeConverter::convertShapedType(ShapedType oldType,
                                            ArrayRef<int64_t> shape,
                                            Type elementType) {
  if (auto memrefType = dyn_cast<UnrankedMemRefType>(oldType)) {
    return UnrankedMemRefType::get(elementType, memrefType.getMemorySpace());
  } else if (auto tensorType = dyn_cast<UnrankedTensorType>(oldType)) {
    return UnrankedTensorType::get(elementType);
  } else if (auto memrefType = dyn_cast<MemRefType>(oldType)) {
    if (memrefType.getShape().size() != shape.size()) {
      assert(memrefType.getShape().size() + 1 == shape.size());
      int64_t newDimension = shape.back();
      SmallVector<int64_t> strides;
      int64_t offset = 0;
      bool result = succeeded(memrefType.getStridesAndOffset(strides, offset));
      assert(result);
      for (int64_t &stride : strides) {
        stride *= newDimension;
      }
      strides.push_back(1);
      auto layout = StridedLayoutAttr::get(memrefType.getContext(),
                                           offset * newDimension, strides);
      return MemRefType::get(shape, elementType, layout,
                             memrefType.getMemorySpace())
          .canonicalizeStridedLayout();
    } else {
      return MemRefType::get(shape, elementType, memrefType.getLayout(),
                             memrefType.getMemorySpace());
    }
  } else if (auto tensorType = dyn_cast<RankedTensorType>(oldType)) {
    return tensorType.cloneWith(shape, elementType);
  }
  llvm_unreachable("Unsupported shaped type");
  return oldType;
}

} // namespace mlir::zkir
