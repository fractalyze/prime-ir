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

#include "prime_ir/Dialect/TensorExt/IR/TensorExtOps.h"

#include "llvm/ADT/bit.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/TensorExt/IR/TensorExtDialect.h"

namespace mlir::prime_ir::tensor_ext {

Operation *TensorExtDialect::materializeConstant(OpBuilder &builder,
                                                 Attribute value, Type type,
                                                 Location loc) {
  // TODO(batzor): Allow for other constant types.
  if (auto op = arith::ConstantOp::materialize(builder, value, type, loc))
    return op;
  return nullptr;
}

LogicalResult BitReverseOp::verify() {
  auto tensorType = cast<RankedTensorType>(getSource().getType());
  int64_t rank = tensorType.getRank();
  int64_t dim = getDimension();

  if (dim < 0 || dim >= rank) {
    return emitOpError("dimension ")
           << dim << " is out of range for tensor of rank " << rank;
  }

  int64_t dimSize = tensorType.getShape()[dim];
  if (!llvm::has_single_bit<uint64_t>(dimSize)) {
    return emitOpError("dimension size ") << dimSize << " is not a power of 2";
  }

  return success();
}

OpFoldResult BitReverseOp::fold(FoldAdaptor adaptor) {
  auto constTensor =
      dyn_cast_if_present<DenseIntElementsAttr>(adaptor.getSource());
  if (!constTensor)
    return {};

  auto tensorType = cast<RankedTensorType>(constTensor.getType());
  int64_t rank = tensorType.getRank();
  int64_t dim = getDimension();
  int64_t dimSize = tensorType.getShape()[dim];

  SmallVector<APInt> values(constTensor.getValues<APInt>());
  unsigned bitWidth = llvm::countr_zero<uint64_t>(dimSize);

  SmallVector<int64_t> strides(rank);
  strides[rank - 1] = 1;
  for (int i = rank - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * tensorType.getShape()[i + 1];
  }

  int64_t numElements = tensorType.getNumElements();
  for (int64_t linearIndex = 0; linearIndex < numElements; ++linearIndex) {
    // Derive multi-dimensional index from linear index.
    int64_t dimIndex = (linearIndex / strides[dim]) % dimSize;

    APInt fromIndex(bitWidth, dimIndex);
    APInt toIndex = fromIndex.reverseBits();
    int64_t revDimIndex = toIndex.getZExtValue();

    if (revDimIndex > dimIndex) {
      // Calculate linear index of the element to swap with.
      int64_t linearRevIndex =
          linearIndex - dimIndex * strides[dim] + revDimIndex * strides[dim];
      std::swap(values[linearIndex], values[linearRevIndex]);
    }
  }

  return DenseElementsAttr::get(tensorType, values);
}

namespace {
#include "prime_ir/Dialect/TensorExt/IR/TensorExtCanonicalization.cpp.inc"
}

void BitReverseOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                               MLIRContext *context) {
  patterns.add<BitReverseIsInvolution>(context);
  patterns.add<BitReverseMulBitReverse>(context);
}
} // namespace mlir::prime_ir::tensor_ext
