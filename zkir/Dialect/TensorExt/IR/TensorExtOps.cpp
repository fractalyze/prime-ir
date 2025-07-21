#include "zkir/Dialect/TensorExt/IR/TensorExtOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "zkir/Dialect/TensorExt/IR/TensorExtAttributes.h"
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
  auto shapedType = cast<ShapedType>(getSource().getType());
  if (auto constTensor =
          dyn_cast_if_present<DenseIntElementsAttr>(adaptor.getSource())) {
    unsigned bitWidth =
        llvm::countr_zero(static_cast<unsigned>(shapedType.getShape()[0]));

    BitReverseIndicesAttr bitReverseIndicesAttr = BitReverseIndicesAttr::get(
        getContext(),
        IntegerAttr::get(IntegerType::get(getContext(), 64), bitWidth));
    DenseIntElementsAttr indices = bitReverseIndicesAttr.getIndices();

    SmallVector<APInt> reversed(constTensor.begin(), constTensor.end());

    // Apply the bit reversal mapping
    for (size_t i = 0; i < indices.getNumElements(); i += 2) {
      size_t fromIdx = (*(indices.begin() + i)).getZExtValue();
      size_t toIdx = (*(indices.begin() + i + 1)).getZExtValue();
      APInt tmp = reversed[fromIdx];
      reversed[fromIdx] = reversed[toIdx];
      reversed[toIdx] = tmp;
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
