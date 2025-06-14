#include "zkir/Utils/ConversionUtils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/include/mlir/Interfaces/ViewLikeInterface.h"

namespace mlir::zkir {

using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;
using ::mlir::func::ReturnOp;

LogicalResult convertAnyOperand(const TypeConverter *typeConverter,
                                Operation *op, ArrayRef<ValueRange> operands,
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

  SmallVector<Value> flatOperands;
  for (auto operand : operands) {
    flatOperands.append(operand.begin(), operand.end());
  }

  if (auto offsetSizeAndStrideOp =
          dyn_cast<OffsetSizeAndStrideOpInterface>(op)) {
    auto newShapedType = cast<ShapedType>(newResultTypes[0]);
    auto oldShapedType = cast<ShapedType>(op->getResultTypes()[0]);
    if (oldShapedType.getRank() != newShapedType.getRank()) {
      SmallVector<NamedAttribute> newAttrs;
      for (const auto &attr : op->getAttrs()) {
        if (attr.getName() == "static_offsets") {
          ArrayRef<int64_t> offsets = offsetSizeAndStrideOp.getStaticOffsets();
          SmallVector<int64_t> newOffsets(offsets);
          newOffsets.push_back(0);
          newAttrs.push_back(rewriter.getNamedAttr(
              attr.getName(), rewriter.getDenseI64ArrayAttr(newOffsets)));
        } else if (attr.getName() == "static_sizes") {
          ArrayRef<int64_t> sizes = offsetSizeAndStrideOp.getStaticSizes();
          SmallVector<int64_t> newSizes(sizes);
          newSizes.push_back(newShapedType.getShape().back());
          newAttrs.push_back(rewriter.getNamedAttr(
              attr.getName(), rewriter.getDenseI64ArrayAttr(newSizes)));
        } else if (attr.getName() == "static_strides") {
          ArrayRef<int64_t> strides = offsetSizeAndStrideOp.getStaticStrides();
          SmallVector<int64_t> newStrides(strides);
          newStrides.push_back(1);
          newAttrs.push_back(rewriter.getNamedAttr(
              attr.getName(), rewriter.getDenseI64ArrayAttr(newStrides)));
        } else {
          newAttrs.push_back(attr);
        }
      }
      Operation *newOp = rewriter.create(OperationState(
          op->getLoc(), op->getName().getStringRef(), flatOperands,
          newResultTypes, newAttrs, op->getSuccessors(), regions));
      rewriter.replaceOp(op, newOp);
      return success();
    }
  }

  // If the op is tensor.extract, and the type conversion is 1:N, the return
  // type is multiple. In this case, we need to split it into multiple extracts.
  // TODO(batzor): Handle other cases (such as tensor.insert).
  if ((op->getName().getStringRef() == "tensor.extract" ||
       op->getName().getStringRef() == "memref.load") &&
      newResultTypes.size() > 1) {
    // NOTE: We can assume the 1:N conversion are all of the same type.
    // Otherwise, the tensor/memref is ill-formed anyways.

    unsigned numResults = newResultTypes.size();
    newResultTypes.resize(1);
    // The rank should be increased by 1 in the type conversion.
    // i.e. tensor<1x2x!QF> -> tensor<1x2x2x!F>
    // This dimension should be used to extract the result.
    flatOperands.resize(flatOperands.size() + 1);

    SmallVector<Value> entryValues;
    entryValues.reserve(numResults);
    for (size_t i = 0; i < numResults; ++i) {
      flatOperands.back() =
          rewriter.create<arith::ConstantIndexOp>(op->getLoc(), i);
      Value valueAtIndex =
          rewriter
              .create(OperationState(
                  op->getLoc(), op->getName().getStringRef(), flatOperands,
                  newResultTypes, op->getAttrs(), op->getSuccessors(), regions))
              ->getResult(0);
      entryValues.push_back(valueAtIndex);
    }

    SmallVector<ValueRange> newOpResults;
    newOpResults.push_back(entryValues);
    rewriter.replaceOpWithMultiple(op, newOpResults);
    return success();
  } else if (op->getName().getStringRef() == "memref.store" &&
             operands[0].size() > 1) {
    // In this case, the value to be inserted is 1:N converted and should be
    // inserted into the container at multiple indices.
    ValueRange valueToInsert = operands[0];
    assert(operands[1].size() == 1 &&
           "memref.store should have a single container operand");
    Value container = operands[1][0];

    // The rest of the operands are indices.
    SmallVector<Value> indices;
    for (size_t i = 2; i < operands.size(); ++i) {
      indices.append(operands[i].begin(), operands[i].end());
    }

    // The rank should be increased by 1 in the type conversion.
    // i.e. memref<1x2x!QF> -> memref<1x2x2x!F>
    // This dimension should be used to extract the result.
    Operation *newOp;
    for (size_t i = 0; i < valueToInsert.size(); ++i) {
      SmallVector<Value> newOperands(indices.size() + 3);
      newOperands[0] = valueToInsert[i];
      newOperands[1] = container;
      std::copy(indices.begin(), indices.end(), newOperands.begin() + 2);
      newOperands.back() =
          rewriter.create<arith::ConstantIndexOp>(op->getLoc(), i);
      newOp = rewriter.create(OperationState(
          op->getLoc(), op->getName().getStringRef(), newOperands,
          newResultTypes, op->getAttrs(), op->getSuccessors(), regions));
    }
    rewriter.replaceOp(op, newOp);
    return success();
  } else {
    Operation *newOp = rewriter.create(OperationState(
        op->getLoc(), op->getName().getStringRef(), flatOperands,
        newResultTypes, op->getAttrs(), op->getSuccessors(), regions));
    rewriter.replaceOp(op, newOp);
  }
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
