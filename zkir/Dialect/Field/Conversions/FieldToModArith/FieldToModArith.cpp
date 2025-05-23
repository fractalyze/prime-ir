#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h"

#include <utility>

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
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

static mod_arith::ModArithType convertPrimeFieldType(PrimeFieldType type) {
  IntegerAttr modulus = type.getModulus();
  return mod_arith::ModArithType::get(type.getContext(), modulus);
}

static Type convertPrimeFieldLikeType(ShapedType type) {
  if (auto primeFieldType = dyn_cast<PrimeFieldType>(type.getElementType())) {
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
    addConversion([](MemRefType type) -> Type {
      if (auto primeFieldType =
              dyn_cast<PrimeFieldType>(type.getElementType())) {
        return type.cloneWith(type.getShape(),
                              convertPrimeFieldType(primeFieldType));
      }
      return type;
    });
  }
};

struct ConvertConstant : public OpConversionPattern<ConstantOp> {
  explicit ConvertConstant(MLIRContext *context)
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
  explicit ConvertEncapsulate(MLIRContext *context)
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
  explicit ConvertExtract(MLIRContext *context)
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

struct ConvertToMont : public OpConversionPattern<ToMontOp> {
  explicit ConvertToMont(MLIRContext *context)
      : OpConversionPattern<ToMontOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ToMontOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type resultType = typeConverter->convertType(op.getResult().getType());
    auto extracted = b.create<mod_arith::ToMontOp>(
        resultType, adaptor.getOperands()[0], op.getMontgomery());
    rewriter.replaceOp(op, extracted);
    return success();
  }
};

struct ConvertFromMont : public OpConversionPattern<FromMontOp> {
  explicit ConvertFromMont(MLIRContext *context)
      : OpConversionPattern<FromMontOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FromMontOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type resultType = typeConverter->convertType(op.getResult().getType());
    auto extracted = b.create<mod_arith::FromMontOp>(
        resultType, adaptor.getOperands()[0], op.getMontgomery());
    rewriter.replaceOp(op, extracted);
    return success();
  }
};

struct ConvertInverse : public OpConversionPattern<InverseOp> {
  explicit ConvertInverse(MLIRContext *context)
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

struct ConvertNegate : public OpConversionPattern<NegateOp> {
  explicit ConvertNegate(MLIRContext *context)
      : OpConversionPattern<NegateOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      NegateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto neg = b.create<mod_arith::NegateOp>(adaptor.getOperands()[0]);
    rewriter.replaceOp(op, neg);
    return success();
  }
};

struct ConvertAdd : public OpConversionPattern<AddOp> {
  explicit ConvertAdd(MLIRContext *context)
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

struct ConvertDouble : public OpConversionPattern<DoubleOp> {
  explicit ConvertDouble(MLIRContext *context)
      : OpConversionPattern<DoubleOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      DoubleOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto doubled =
        b.create<mod_arith::AddOp>(adaptor.getInput(), adaptor.getInput());
    rewriter.replaceOp(op, doubled);
    return success();
  }
};

struct ConvertSub : public OpConversionPattern<SubOp> {
  explicit ConvertSub(MLIRContext *context)
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
  explicit ConvertMul(MLIRContext *context)
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

struct ConvertSquare : public OpConversionPattern<SquareOp> {
  explicit ConvertSquare(MLIRContext *context)
      : OpConversionPattern<SquareOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SquareOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto square =
        b.create<mod_arith::MulOp>(adaptor.getInput(), adaptor.getInput());
    rewriter.replaceOp(op, square);
    return success();
  }
};

struct ConvertMontMul : public OpConversionPattern<MontMulOp> {
  explicit ConvertMontMul(MLIRContext *context)
      : OpConversionPattern<MontMulOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MontMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto mul = b.create<mod_arith::MontMulOp>(
        adaptor.getLhs(), adaptor.getRhs(), op.getMontgomery());
    rewriter.replaceOp(op, mul);
    return success();
  }
};

// TODO(ashjeong): Account for Montgomery domain inputs. Currently only accounts
// for base domain inputs.
struct ConvertCmp : public OpConversionPattern<CmpOp> {
  explicit ConvertCmp(MLIRContext *context)
      : OpConversionPattern<CmpOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CmpOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto cmpOp = b.create<mod_arith::CmpOp>(op.getPredicate(), adaptor.getLhs(),
                                            adaptor.getRhs());
    rewriter.replaceOp(op, cmpOp);
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
  // NOTE(ashjeong): ConvertPointSetOp from the elliptic curve to field pass
  // introduces a tensor::concat op. As tensor::concat op is not supported by
  // the OneShotBufferize pass, we must decompose them with this pattern.
  tensor::populateDecomposeTensorConcatPatterns(patterns);
  patterns.add<
      ConvertConstant, ConvertEncapsulate, ConvertExtract, ConvertToMont,
      ConvertFromMont, ConvertInverse, ConvertNegate, ConvertAdd, ConvertDouble,
      ConvertSub, ConvertMul, ConvertSquare, ConvertMontMul, ConvertCmp,
      ConvertAny<affine::AffineForOp>, ConvertAny<affine::AffineParallelOp>,
      ConvertAny<affine::AffineLoadOp>, ConvertAny<affine::AffineStoreOp>,
      ConvertAny<affine::AffineYieldOp>, ConvertAny<linalg::GenericOp>,
      ConvertAny<linalg::MapOp>, ConvertAny<memref::LoadOp>,
      ConvertAny<memref::StoreOp>, ConvertAny<linalg::YieldOp>,
      ConvertAny<tensor::CastOp>, ConvertAny<tensor::ExtractOp>,
      ConvertAny<tensor::ExtractSliceOp>, ConvertAny<tensor::InsertSliceOp>,
      ConvertAny<tensor::EmptyOp>, ConvertAny<tensor::FromElementsOp>,
      ConvertAny<tensor::ConcatOp>, ConvertAny<tensor::ReshapeOp>,
      ConvertAny<bufferization::MaterializeInDestinationOp>,
      ConvertAny<bufferization::ToMemrefOp>,
      ConvertAny<bufferization::ToTensorOp>, ConvertAny<tensor::InsertOp>>(
      typeConverter, context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  target.addDynamicallyLegalOp<
      affine::AffineForOp, affine::AffineParallelOp, affine::AffineLoadOp,
      affine::AffineStoreOp, affine::AffineYieldOp,
      bufferization::MaterializeInDestinationOp, bufferization::ToMemrefOp,
      bufferization::ToTensorOp, linalg::GenericOp, linalg::MapOp,
      linalg::YieldOp, memref::LoadOp, memref::StoreOp, tensor::CastOp,
      tensor::ExtractOp, tensor::ExtractSliceOp, tensor::InsertSliceOp,
      tensor::EmptyOp, tensor::FromElementsOp, tensor::InsertOp,
      tensor::ConcatOp, tensor::ReshapeOp>(
      [&](auto op) { return typeConverter.isLegal(op); });

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace mlir::zkir::field
