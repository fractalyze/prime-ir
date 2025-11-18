/* Copyright 2025 The ZKIR Authors.

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

      // If the offset is dynamic, we don't need to multiply by the new
      // dimension.
      if (offset != ShapedType::kDynamic) {
        offset *= newDimension;
      }
      auto layout =
          StridedLayoutAttr::get(memrefType.getContext(), offset, strides);
      return MemRefType::get(shape, elementType, layout,
                             memrefType.getMemorySpace())
          .canonicalizeStridedLayout();
    } else {
      return MemRefType::get(shape, elementType, memrefType.getLayout(),
                             memrefType.getMemorySpace());
    }
  } else if (auto tensorType = dyn_cast<RankedTensorType>(oldType)) {
    return tensorType.cloneWith(shape, elementType);
  } else if (auto vectorType = dyn_cast<VectorType>(oldType)) {
    return vectorType.cloneWith(shape, elementType);
  }
  llvm_unreachable("Unsupported shaped type");
  return oldType;
}

} // namespace mlir::zkir
