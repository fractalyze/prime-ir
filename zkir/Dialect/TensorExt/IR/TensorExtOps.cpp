#include "zkir/Dialect/TensorExt/IR/TensorExtOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "zkir/Dialect/TensorExt/IR/TensorExtDialect.h"

namespace mlir::zkir::tensor_ext {

Operation *TensorExtDialect::materializeConstant(OpBuilder &builder,
                                                 Attribute value, Type type,
                                                 Location loc) {
  // TODO(batzor): Allow for other constant types.
  if (auto op = arith::ConstantOp::materialize(builder, value, type, loc))
    return op;
  return nullptr;
}

OpFoldResult BitReverseOp::fold(FoldAdaptor adaptor) {
  if (auto constTensor =
          dyn_cast_if_present<DenseIntElementsAttr>(adaptor.getSource())) {
    SmallVector<APInt> reversed(constTensor.begin(), constTensor.end());
    unsigned bitWidth = llvm::countr_zero(reversed.size());

    // Apply the bit reversal mapping
    for (size_t i = 0; i < reversed.size(); ++i) {
      APInt fromIndex(bitWidth, i);
      APInt toIndex = fromIndex.reverseBits();
      if (toIndex.ugt(fromIndex)) {
        size_t j = toIndex.getZExtValue();
        APInt tmp = reversed[i];
        reversed[i] = reversed[j];
        reversed[j] = tmp;
      }
    }

    return DenseElementsAttr::get(constTensor.getType(), reversed);
  }
  return {};
}

namespace {
#include "zkir/Dialect/TensorExt/IR/TensorExtCanonicalization.cpp.inc"
}

void BitReverseOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                               MLIRContext *context) {
  patterns.add<BitReverseIsInvolution>(context);
}
}  // namespace mlir::zkir::tensor_ext
