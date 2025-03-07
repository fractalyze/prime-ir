#include "zkir/Utils/ConversionUtils.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "llvm/include/llvm/ADT/TypeSwitch.h"
#include "llvm/include/llvm/Support/ErrorHandling.h"
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/include/mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/IRMapping.h"
#include "mlir/include/mlir/IR/OpDefinition.h"
#include "mlir/include/mlir/IR/OperationSupport.h"
#include "mlir/include/mlir/IR/PatternMatch.h"
#include "mlir/include/mlir/IR/Region.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/IR/Verifier.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Support/LogicalResult.h"
#include "mlir/include/mlir/Transforms/DialectConversion.h"

namespace mlir::zkir {

using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;
using ::mlir::func::ReturnOp;

LogicalResult convertAnyOperand(const TypeConverter *typeConverter,
                                Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) {
  if (typeConverter->isLegal(op)) {
    return failure();
  }

  SmallVector<Type> newOperandTypes;
  if (failed(
          typeConverter->convertTypes(op->getOperandTypes(), newOperandTypes)))
    return failure();

  SmallVector<Type> newResultTypes;
  if (failed(typeConverter->convertTypes(op->getResultTypes(), newResultTypes)))
    return failure();

  SmallVector<std::unique_ptr<Region>, 1> regions;
  IRMapping mapping;
  for (auto &r : op->getRegions()) {
    std::unique_ptr<Region> newRegion(new Region(op));
    rewriter.cloneRegionBefore(r, *newRegion, newRegion->end(), mapping);
    if (failed(rewriter.convertRegionTypes(newRegion.get(), *typeConverter)))
      return failure();
    regions.push_back(std::move(newRegion));
  }

  Operation *newOp = rewriter.create(OperationState(
      op->getLoc(), op->getName().getStringRef(), operands, newResultTypes,
      op->getAttrs(), op->getSuccessors(), regions));

  rewriter.replaceOp(op, newOp);
  return success();
}

void addStructuralConversionPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     ConversionTarget &target) {
  populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                           typeConverter);
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType()) &&
           typeConverter.isLegal(&op.getBody());
  });

  populateReturnOpTypeConversionPattern(patterns, typeConverter);
  target.addDynamicallyLegalOp<func::ReturnOp>(
      [&](func::ReturnOp op) { return typeConverter.isLegal(op); });

  populateCallOpTypeConversionPattern(patterns, typeConverter);
  target.addDynamicallyLegalOp<func::CallOp>(
      [&](func::CallOp op) { return typeConverter.isLegal(op); });

  populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
           isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                            typeConverter) ||
           isLegalForReturnOpTypeConversionPattern(op, typeConverter);
  });

  scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns,
                                                       target);
}

}  // namespace mlir::zkir
