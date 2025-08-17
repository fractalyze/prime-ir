#include "zkir/Dialect/Poly/IR/PolyOps.h"

#include "mlir/include/mlir/IR/PatternMatch.h"

namespace mlir::zkir::poly {

namespace {
#include "zkir/Dialect/Poly/IR/PolyCanonicalization.cpp.inc"
} // namespace

void FromTensorOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<ToFromTensor>(context);
}

void ToTensorOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<FromToTensor>(context);
}

void NTTOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<INTTAfterNTT>(context);
}

} // namespace mlir::zkir::poly
