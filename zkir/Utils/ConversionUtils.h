#ifndef ZKIR_UTILS_CONVERSIONUTILS_H_
#define ZKIR_UTILS_CONVERSIONUTILS_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::zkir {

LogicalResult convertAnyOperand(const TypeConverter *typeConverter,
                                Operation *op, ArrayRef<ValueRange> operands,
                                ConversionPatternRewriter &rewriter);

template <typename T>
struct ConvertAny : public ConversionPattern {
  ConvertAny(const TypeConverter &anyTypeConverter, MLIRContext *context)
      : ConversionPattern(anyTypeConverter, RewritePattern::MatchAnyOpTypeTag(),
                          /*benefit=*/1, context) {
    setDebugName("ConvertAny");
    setHasBoundedRewriteRecursion(true);
  }

  // generate a new op where all operands have been replaced with their
  // materialized/typeconverted versions
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<ValueRange> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<T>(op)) {
      return failure();
    }

    return convertAnyOperand(getTypeConverter(), op, operands, rewriter);
  }
};

template <>
struct ConvertAny<void> : public ConversionPattern {
  ConvertAny<void>(const TypeConverter &anyTypeConverter, MLIRContext *context)
      : ConversionPattern(anyTypeConverter, RewritePattern::MatchAnyOpTypeTag(),
                          /*benefit=*/1, context) {
    setDebugName("ConvertAny");
    setHasBoundedRewriteRecursion(true);
  }

  // generate a new op where all operands have been replaced with their
  // materialized/typeconverted versions
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<ValueRange> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return convertAnyOperand(getTypeConverter(), op, operands, rewriter);
  }
};

template <typename SourceArithOp, typename TargetModArithOp>
struct ConvertBinOp : public OpConversionPattern<SourceArithOp> {
  explicit ConvertBinOp(MLIRContext *context)
      : OpConversionPattern<SourceArithOp>(context) {}

  using OpConversionPattern<SourceArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceArithOp op, typename SourceArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto result = b.create<TargetModArithOp>(
        adaptor.getLhs().getType(), adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Adds the standard set of conversion patterns for
// converting types involved in func, cf, etc., which
// don't depend on the logic of the dialect beyond the
// type converter.
void addStructuralConversionPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     ConversionTarget &target);

} // namespace mlir::zkir

#endif // ZKIR_UTILS_CONVERSIONUTILS_H_
