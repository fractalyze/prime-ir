#include "zkir/Dialect/Poly/IR/PolyOps.h"

#include "mlir/include/mlir/IR/PatternMatch.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"
#include "zkir/Dialect/Poly/IR/PolyAttributes.h"
#include "zkir/Dialect/Poly/IR/PolyTypes.h"

namespace mlir::zkir::poly {

namespace {
#include "zkir/Dialect/Poly/IR/PolyCanonicalization.cpp.inc"
}  // namespace

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
  results.add<NTTAfterINTT>(context);
}

void INTTOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<INTTAfterNTT>(context);
}

}  // namespace mlir::zkir::poly
