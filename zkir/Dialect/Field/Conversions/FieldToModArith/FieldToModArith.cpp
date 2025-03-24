#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h"

#include <utility>

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "zkir/Dialect/Field/IR/FieldDialect.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"
#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"
#include "zkir/Dialect/ModArith/IR/ModArithOps.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"
#include "zkir/Utils/ConversionUtils.h"

namespace mlir::zkir::field {

#define GEN_PASS_DEF_PRIMEFIELDTOMODARITH
#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h.inc"

mod_arith::ModArithType convertPrimeFieldType(PrimeFieldType type) {
  IntegerAttr modulus = type.getModulus();
  return mod_arith::ModArithType::get(type.getContext(), modulus);
}

Type convertPrimeFieldLikeType(ShapedType type) {
  if (auto primeFieldType =
          llvm::dyn_cast<PrimeFieldType>(type.getElementType())) {
    return type.cloneWith(type.getShape(),
                          convertPrimeFieldType(primeFieldType));
  }
  return type;
}

class PrimeFieldToModArithTypeConverter : public TypeConverter {
 public:
  explicit PrimeFieldToModArithTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([](PrimeFieldType type) -> Type {
      return convertPrimeFieldType(type);
    });
    addConversion([](ShapedType type) -> Type {
      return convertPrimeFieldLikeType(type);
    });
  }
};

struct ConvertConstant : public OpConversionPattern<ConstantOp> {
  explicit ConvertConstant(mlir::MLIRContext *context)
      : OpConversionPattern<ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto pftype = getResultPrimeFieldType(op);
    auto modType = convertPrimeFieldType(pftype);
    auto cval = b.create<mod_arith::ConstantOp>(modType, op.getValue());
    rewriter.replaceOp(op, cval);
    return success();
  }
};

struct ConvertEncapsulate : public OpConversionPattern<EncapsulateOp> {
  explicit ConvertEncapsulate(mlir::MLIRContext *context)
      : OpConversionPattern<EncapsulateOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      EncapsulateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto resultType = typeConverter->convertType(op.getResult().getType());
    auto enc = b.create<mod_arith::EncapsulateOp>(resultType,
                                                  adaptor.getOperands()[0]);
    rewriter.replaceOp(op, enc);
    return success();
  }
};

struct ConvertExtract : public OpConversionPattern<ExtractOp> {
  explicit ConvertExtract(mlir::MLIRContext *context)
      : OpConversionPattern<ExtractOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ExtractOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto resultType = typeConverter->convertType(op.getResult().getType());
    auto extracted =
        b.create<mod_arith::ExtractOp>(resultType, adaptor.getOperands()[0]);
    rewriter.replaceOp(op, extracted);
    return success();
  }
};

struct ConvertInverse : public OpConversionPattern<InverseOp> {
  explicit ConvertInverse(mlir::MLIRContext *context)
      : OpConversionPattern<InverseOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      InverseOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto inv = b.create<mod_arith::InverseOp>(adaptor.getOperands()[0]);
    rewriter.replaceOp(op, inv);
    return success();
  }
};

struct ConvertAdd : public OpConversionPattern<AddOp> {
  explicit ConvertAdd(mlir::MLIRContext *context)
      : OpConversionPattern<AddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto add = b.create<mod_arith::AddOp>(adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, add);
    return success();
  }
};

struct ConvertSub : public OpConversionPattern<SubOp> {
  explicit ConvertSub(mlir::MLIRContext *context)
      : OpConversionPattern<SubOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SubOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto sub = b.create<mod_arith::SubOp>(adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, sub);
    return success();
  }
};

struct ConvertMul : public OpConversionPattern<MulOp> {
  explicit ConvertMul(mlir::MLIRContext *context)
      : OpConversionPattern<MulOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto mul = b.create<mod_arith::MulOp>(adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, mul);
    return success();
  }
};

namespace rewrites {
// In an inner namespace to avoid conflicts with canonicalization patterns
#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.cpp.inc"
}  // namespace rewrites

struct PrimeFieldToModArith
    : impl::PrimeFieldToModArithBase<PrimeFieldToModArith> {
  using PrimeFieldToModArithBase::PrimeFieldToModArithBase;

  void runOnOperation() override;
};

void PrimeFieldToModArith::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  PrimeFieldToModArithTypeConverter typeConverter(context);

  ConversionTarget target(*context);
  target.addIllegalDialect<FieldDialect>();
  target.addLegalDialect<mod_arith::ModArithDialect>();

  RewritePatternSet patterns(context);
  rewrites::populateWithGenerated(patterns);
  patterns
      .add<ConvertConstant, ConvertEncapsulate, ConvertExtract, ConvertInverse,
           ConvertAdd, ConvertSub, ConvertMul, ConvertAny<affine::AffineForOp>,
           ConvertAny<affine::AffineYieldOp>, ConvertAny<linalg::GenericOp>,
           ConvertAny<linalg::YieldOp>, ConvertAny<tensor::CastOp>,
           ConvertAny<tensor::ExtractOp>, ConvertAny<tensor::FromElementsOp>,
           ConvertAny<tensor::InsertOp>>(typeConverter, context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  target.addDynamicallyLegalOp<affine::AffineForOp, affine::AffineYieldOp,
                               linalg::GenericOp, linalg::YieldOp,
                               tensor::CastOp, tensor::ExtractOp,
                               tensor::FromElementsOp, tensor::InsertOp>(
      [&](auto op) { return typeConverter.isLegal(op); });

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace mlir::zkir::field
