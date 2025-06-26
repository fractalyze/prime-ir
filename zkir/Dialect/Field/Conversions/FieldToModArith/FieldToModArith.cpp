#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h"

#include <utility>

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
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
#include "zkir/Dialect/TensorExt/IR/TensorExtOps.h"
#include "zkir/Utils/ConversionUtils.h"
#include "zkir/Utils/ShapedTypeConverter.h"

namespace mlir::zkir::field {

#define GEN_PASS_DEF_FIELDTOMODARITH
#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h.inc"

class FieldToModArithTypeConverter : public ShapedTypeConverter {
 public:
  explicit FieldToModArithTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([](PrimeFieldType type) -> Type {
      return convertPrimeFieldType(type);
    });
    addConversion([](QuadraticExtFieldType type,
                     SmallVectorImpl<Type> &converted) -> LogicalResult {
      return convertQuadraticExtFieldType(type, converted);
    });
    addConversion([](ShapedType type) -> Type {
      if (auto primeFieldType =
              dyn_cast<PrimeFieldType>(type.getElementType())) {
        return convertShapedType(type, type.getShape(),
                                 convertPrimeFieldType(primeFieldType));
      }
      if (auto quadraticExtFieldType =
              dyn_cast<QuadraticExtFieldType>(type.getElementType())) {
        PrimeFieldType baseFieldType = quadraticExtFieldType.getBaseField();
        mod_arith::ModArithType modArithType =
            convertPrimeFieldType(baseFieldType);
        SmallVector<int64_t> newShape(type.getShape());
        newShape.push_back(2);
        return convertShapedType(type, newShape, modArithType);
      }
      return type;
    });
  }

 private:
  static mod_arith::ModArithType convertPrimeFieldType(PrimeFieldType type) {
    IntegerAttr modulus = type.getModulus();
    bool isMontgomery = type.isMontgomery();
    return mod_arith::ModArithType::get(type.getContext(), modulus,
                                        isMontgomery);
  }

  static LogicalResult convertQuadraticExtFieldType(
      QuadraticExtFieldType type, SmallVectorImpl<Type> &converted) {
    mod_arith::ModArithType modArithType =
        convertPrimeFieldType(type.getBaseField());
    converted.push_back(modArithType);
    converted.push_back(modArithType);
    return success();
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

    mod_arith::ModArithType modType;
    if (auto pfType = dyn_cast<PrimeFieldType>(op.getOutput().getType())) {
      modType =
          cast<mod_arith::ModArithType>(typeConverter->convertType(pfType));
    } else if (auto f2Type =
                   dyn_cast<QuadraticExtFieldType>(op.getOutput().getType())) {
      modType = cast<mod_arith::ModArithType>(
          typeConverter->convertType(f2Type.getBaseField()));
    } else {
      op.emitOpError("unsupported output type");
      return failure();
    }

    if (auto pfAttr = dyn_cast<PrimeFieldAttr>(op.getValueAttr())) {
      auto cval = b.create<mod_arith::ConstantOp>(modType, pfAttr.getValue());
      rewriter.replaceOp(op, cval);
      return success();
    } else if (auto f2Attr =
                   dyn_cast<QuadraticExtFieldAttr>(op.getValueAttr())) {
      auto low =
          b.create<mod_arith::ConstantOp>(modType, f2Attr.getLow().getValue());
      auto high =
          b.create<mod_arith::ConstantOp>(modType, f2Attr.getHigh().getValue());
      rewriter.replaceOpWithMultiple(op, {{low, high}});
      return success();
    } else {
      op.emitOpError("unsupported attribute type");
      return failure();
    }
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
      ToMontOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    if (isa<PrimeFieldType>(fieldType)) {
      Type resultType = typeConverter->convertType(op.getResult().getType());
      auto extracted = b.create<mod_arith::ToMontOp>(
          resultType, adaptor.getOperands()[0][0]);
      rewriter.replaceOp(op, extracted);
      return success();
    }
    if (isa<QuadraticExtFieldType>(fieldType)) {
      auto extFieldType = cast<QuadraticExtFieldType>(fieldType);
      Type resultType = typeConverter->convertType(extFieldType.getBaseField());
      auto c0 = b.create<mod_arith::ToMontOp>(resultType,
                                              adaptor.getOperands()[0][0]);
      auto c1 = b.create<mod_arith::ToMontOp>(resultType,
                                              adaptor.getOperands()[0][1]);
      rewriter.replaceOpWithMultiple(op, {{c0, c1}});
      return success();
    }
    return failure();
  }
};

struct ConvertFromMont : public OpConversionPattern<FromMontOp> {
  explicit ConvertFromMont(MLIRContext *context)
      : OpConversionPattern<FromMontOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FromMontOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    if (isa<PrimeFieldType>(fieldType)) {
      Type resultType = typeConverter->convertType(op.getResult().getType());
      auto extracted = b.create<mod_arith::FromMontOp>(
          resultType, adaptor.getOperands()[0][0]);
      rewriter.replaceOp(op, extracted);
      return success();
    }
    if (isa<QuadraticExtFieldType>(fieldType)) {
      auto extFieldType = cast<QuadraticExtFieldType>(fieldType);
      Type resultType = typeConverter->convertType(extFieldType.getBaseField());
      auto c0 = b.create<mod_arith::FromMontOp>(resultType,
                                                adaptor.getOperands()[0][0]);
      auto c1 = b.create<mod_arith::FromMontOp>(resultType,
                                                adaptor.getOperands()[0][1]);
      rewriter.replaceOpWithMultiple(op, {{c0, c1}});
      return success();
    }
    return failure();
  }
};

struct ConvertInverse : public OpConversionPattern<InverseOp> {
  explicit ConvertInverse(MLIRContext *context)
      : OpConversionPattern<InverseOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      InverseOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    if (isa<PrimeFieldType>(fieldType)) {
      auto inv = b.create<mod_arith::InverseOp>(adaptor.getInput()[0]);
      rewriter.replaceOp(op, inv);
      return success();
    }
    if (isa<QuadraticExtFieldType>(fieldType)) {
      // construct beta as a mod arith constant
      auto extFieldType = cast<QuadraticExtFieldType>(fieldType);
      auto beta = b.create<mod_arith::ConstantOp>(
          typeConverter->convertType(extFieldType.getBaseField()),
          extFieldType.getBeta().getValue());

      // denominator = a₀² - a₁²β
      auto lowSquared = b.create<mod_arith::SquareOp>(adaptor.getInput()[0]);
      auto highSquared = b.create<mod_arith::SquareOp>(adaptor.getInput()[1]);
      auto betaTimesHighSquared = b.create<mod_arith::MulOp>(beta, highSquared);
      auto denominator =
          b.create<mod_arith::SubOp>(lowSquared, betaTimesHighSquared);
      auto denominatorInv = b.create<mod_arith::InverseOp>(denominator);

      // c₀ = a₀ / denominator
      auto c0 =
          b.create<mod_arith::MulOp>(adaptor.getInput()[0], denominatorInv);
      // c₁ = -a₁ / denominator
      auto highNegated = b.create<mod_arith::NegateOp>(adaptor.getInput()[1]);
      auto c1 = b.create<mod_arith::MulOp>(highNegated, denominatorInv);
      rewriter.replaceOpWithMultiple(op, {{c0, c1}});
      return success();
    }
    return failure();
  }
};

struct ConvertNegate : public OpConversionPattern<NegateOp> {
  explicit ConvertNegate(MLIRContext *context)
      : OpConversionPattern<NegateOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      NegateOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    if (isa<PrimeFieldType>(fieldType)) {
      auto neg = b.create<mod_arith::NegateOp>(adaptor.getInput()[0]);
      rewriter.replaceOp(op, neg);
      return success();
    }
    if (isa<QuadraticExtFieldType>(fieldType)) {
      auto c0 = b.create<mod_arith::NegateOp>(adaptor.getInput()[0]);
      auto c1 = b.create<mod_arith::NegateOp>(adaptor.getInput()[1]);
      rewriter.replaceOpWithMultiple(op, {{c0, c1}});
      return success();
    }
    return failure();
  }
};

struct ConvertAdd : public OpConversionPattern<AddOp> {
  explicit ConvertAdd(MLIRContext *context)
      : OpConversionPattern<AddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AddOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    if (isa<PrimeFieldType>(fieldType)) {
      auto add =
          b.create<mod_arith::AddOp>(adaptor.getLhs()[0], adaptor.getRhs()[0]);
      rewriter.replaceOp(op, add);
      return success();
    }
    if (isa<QuadraticExtFieldType>(fieldType)) {
      // c₀ = a₀ + b₀
      // c₁ = a₁ + b₁
      auto c0 =
          b.create<mod_arith::AddOp>(adaptor.getLhs()[0], adaptor.getRhs()[0]);
      auto c1 =
          b.create<mod_arith::AddOp>(adaptor.getLhs()[1], adaptor.getRhs()[1]);
      rewriter.replaceOpWithMultiple(op, {{c0, c1}});
      return success();
    }
    return failure();
  }
};

struct ConvertDouble : public OpConversionPattern<DoubleOp> {
  explicit ConvertDouble(MLIRContext *context)
      : OpConversionPattern<DoubleOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      DoubleOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    if (isa<PrimeFieldType>(fieldType)) {
      auto doubled = b.create<mod_arith::DoubleOp>(adaptor.getInput()[0]);
      rewriter.replaceOp(op, doubled);
      return success();
    }
    if (isa<QuadraticExtFieldType>(fieldType)) {
      // c₀ = a₀ + a₀
      // c₁ = a₁ + a₁
      auto c0 = b.create<mod_arith::DoubleOp>(adaptor.getInput()[0]);
      auto c1 = b.create<mod_arith::DoubleOp>(adaptor.getInput()[1]);
      rewriter.replaceOpWithMultiple(op, {{c0, c1}});
      return success();
    }
    return failure();
  }
};

struct ConvertSub : public OpConversionPattern<SubOp> {
  explicit ConvertSub(MLIRContext *context)
      : OpConversionPattern<SubOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SubOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    if (isa<PrimeFieldType>(fieldType)) {
      auto sub =
          b.create<mod_arith::SubOp>(adaptor.getLhs()[0], adaptor.getRhs()[0]);
      rewriter.replaceOp(op, sub);
      return success();
    }
    if (isa<QuadraticExtFieldType>(fieldType)) {
      auto c0 =
          b.create<mod_arith::SubOp>(adaptor.getLhs()[0], adaptor.getRhs()[0]);
      auto c1 =
          b.create<mod_arith::SubOp>(adaptor.getLhs()[1], adaptor.getRhs()[1]);
      rewriter.replaceOpWithMultiple(op, {{c0, c1}});
      return success();
    }
    return failure();
  }
};

struct ConvertMul : public OpConversionPattern<MulOp> {
  explicit ConvertMul(MLIRContext *context)
      : OpConversionPattern<MulOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MulOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    if (isa<PrimeFieldType>(fieldType)) {
      auto mul =
          b.create<mod_arith::MulOp>(adaptor.getLhs()[0], adaptor.getRhs()[0]);
      rewriter.replaceOp(op, mul);
      return success();
    }
    if (isa<QuadraticExtFieldType>(fieldType)) {
      // construct beta as a mod arith constant
      auto extFieldType = cast<QuadraticExtFieldType>(fieldType);
      auto beta = b.create<mod_arith::ConstantOp>(
          typeConverter->convertType(extFieldType.getBaseField()),
          extFieldType.getBeta().getValue());

      // v₀ = a₀ * b₀
      // v₁ = a₁ * b₁
      auto v0 =
          b.create<mod_arith::MulOp>(adaptor.getLhs()[0], adaptor.getRhs()[0]);
      auto v1 =
          b.create<mod_arith::MulOp>(adaptor.getLhs()[1], adaptor.getRhs()[1]);

      // c₀ = v₀ + βv₁
      auto betaTimesV1 = b.create<mod_arith::MulOp>(beta, v1);
      auto c0 = b.create<mod_arith::AddOp>(v0, betaTimesV1);

      // c₁ = (a₀ + a₁)(b₀ + b₁) - v₀ - v₁
      auto sumLhs =
          b.create<mod_arith::AddOp>(adaptor.getLhs()[0], adaptor.getLhs()[1]);
      auto sumRhs =
          b.create<mod_arith::AddOp>(adaptor.getRhs()[0], adaptor.getRhs()[1]);
      auto sumProduct = b.create<mod_arith::MulOp>(sumLhs, sumRhs);
      Value c1 = b.create<mod_arith::SubOp>(sumProduct, v0);
      c1 = b.create<mod_arith::SubOp>(c1, v1);

      rewriter.replaceOpWithMultiple(op, {{c0, c1}});
      return success();
    }
    return failure();
  }
};

struct ConvertSquare : public OpConversionPattern<SquareOp> {
  explicit ConvertSquare(MLIRContext *context)
      : OpConversionPattern<SquareOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SquareOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    if (isa<PrimeFieldType>(fieldType)) {
      auto square = b.create<mod_arith::SquareOp>(adaptor.getInput()[0]);
      rewriter.replaceOp(op, square);
      return success();
    }
    if (isa<QuadraticExtFieldType>(fieldType)) {
      // construct beta as a mod arith constant
      auto extFieldType = cast<QuadraticExtFieldType>(fieldType);
      auto beta = b.create<mod_arith::ConstantOp>(
          typeConverter->convertType(extFieldType.getBaseField()),
          extFieldType.getBeta().getValue());

      // v₀ = a₀ - a₁
      Value v0 = b.create<mod_arith::SubOp>(adaptor.getInput()[0],
                                            adaptor.getInput()[1]);

      // v₁ = a₀ - βa₁
      auto betaA1 = b.create<mod_arith::MulOp>(beta, adaptor.getInput()[1]);
      auto v1 = b.create<mod_arith::SubOp>(adaptor.getInput()[0], betaA1);

      // v₂ = a₀ * a₁
      auto v2 = b.create<mod_arith::MulOp>(adaptor.getInput()[0],
                                           adaptor.getInput()[1]);

      // v₀ = v₀ * v₁ + v₂
      auto v0TimesV1 = b.create<mod_arith::MulOp>(v0, v1);
      v0 = b.create<mod_arith::AddOp>(v0TimesV1, v2);

      // c₁ = v₂ + v₂
      auto c1 = b.create<mod_arith::DoubleOp>(v2);
      // c₀ = v₀ + βv₂
      auto betaV2 = b.create<mod_arith::MulOp>(beta, v2);
      auto c0 = b.create<mod_arith::AddOp>(v0, betaV2);
      rewriter.replaceOpWithMultiple(op, {{c0, c1}});
      return success();
    }
    return failure();
  }
};

// TODO(ashjeong): Account for Montgomery domain inputs. Currently only accounts
// for base domain inputs.
struct ConvertCmp : public OpConversionPattern<CmpOp> {
  explicit ConvertCmp(MLIRContext *context)
      : OpConversionPattern<CmpOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CmpOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type fieldType = getElementTypeOrSelf(op.getLhs());
    if (isa<PrimeFieldType>(fieldType)) {
      auto cmpOp = b.create<mod_arith::CmpOp>(
          op.getPredicate(), adaptor.getLhs()[0], adaptor.getRhs()[0]);
      rewriter.replaceOp(op, cmpOp);
      return success();
    } else if (isa<QuadraticExtFieldType>(fieldType)) {
      // For quadratic extension fields, we compare the low and high parts
      // separately.
      auto cmpLow = b.create<mod_arith::CmpOp>(
          op.getPredicate(), adaptor.getLhs()[0], adaptor.getRhs()[0]);
      auto cmpHigh = b.create<mod_arith::CmpOp>(
          op.getPredicate(), adaptor.getLhs()[1], adaptor.getRhs()[1]);
      auto result = b.create<arith::AndIOp>(cmpLow, cmpHigh);
      rewriter.replaceOp(op, result);
      return success();
    }
    return failure();
  }
};

struct ConvertF2Constant : public OpConversionPattern<F2ConstantOp> {
  explicit ConvertF2Constant(MLIRContext *context)
      : OpConversionPattern<F2ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      F2ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    SmallVector<ValueRange> values;
    values.push_back({adaptor.getLow(), adaptor.getHigh()});

    rewriter.replaceOpWithMultiple(op, values);
    return success();
  }
};

namespace rewrites {
// In an inner namespace to avoid conflicts with canonicalization patterns
#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.cpp.inc"
}  // namespace rewrites

struct FieldToModArith : impl::FieldToModArithBase<FieldToModArith> {
  using FieldToModArithBase::FieldToModArithBase;

  void runOnOperation() override;
};

void FieldToModArith::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  FieldToModArithTypeConverter typeConverter(context);

  ConversionTarget target(*context);
  target.addIllegalDialect<FieldDialect>();
  target.addLegalDialect<mod_arith::ModArithDialect>();

  RewritePatternSet patterns(context);
  rewrites::populateWithGenerated(patterns);
  patterns.add<
      // clang-format off
      ConvertAdd,
      ConvertConstant,
      ConvertCmp,
      ConvertDouble,
      ConvertEncapsulate,
      ConvertExtract,
      ConvertF2Constant,
      ConvertFromMont,
      ConvertInverse,
      ConvertNegate,
      ConvertMul,
      ConvertSquare,
      ConvertSub,
      ConvertToMont,
      ConvertAny<affine::AffineForOp>,
      ConvertAny<affine::AffineLoadOp>,
      ConvertAny<affine::AffineParallelOp>,
      ConvertAny<affine::AffineStoreOp>,
      ConvertAny<affine::AffineYieldOp>,
      ConvertAny<bufferization::MaterializeInDestinationOp>,
      ConvertAny<bufferization::ToMemrefOp>,
      ConvertAny<bufferization::ToTensorOp>,
      ConvertAny<linalg::BroadcastOp>,
      ConvertAny<linalg::GenericOp>,
      ConvertAny<linalg::MapOp>,
      ConvertAny<linalg::YieldOp>,
      ConvertAny<memref::AllocOp>,
      ConvertAny<memref::CastOp>,
      ConvertAny<memref::CopyOp>,
      ConvertAny<memref::DimOp>,
      ConvertAny<memref::LoadOp>,
      ConvertAny<memref::StoreOp>,
      ConvertAny<memref::SubViewOp>,
      ConvertAny<memref::ViewOp>,
      ConvertAny<sparse_tensor::AssembleOp>,
      ConvertAny<tensor::CastOp>,
      ConvertAny<tensor::ConcatOp>,
      ConvertAny<tensor::DimOp>,
      ConvertAny<tensor::EmptyOp>,
      ConvertAny<tensor::ExtractOp>,
      ConvertAny<tensor::ExtractSliceOp>,
      ConvertAny<tensor::FromElementsOp>,
      ConvertAny<tensor::InsertOp>,
      ConvertAny<tensor::InsertSliceOp>,
      ConvertAny<tensor::ReshapeOp>,
      ConvertAny<tensor_ext::BitReverseOp>
      // clang-format on
      >(typeConverter, context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  target.addDynamicallyLegalOp<
      // clang-format off
      affine::AffineForOp,
      affine::AffineLoadOp,
      affine::AffineParallelOp,
      affine::AffineStoreOp,
      affine::AffineYieldOp,
      bufferization::MaterializeInDestinationOp,
      bufferization::ToMemrefOp,
      bufferization::ToTensorOp,
      linalg::BroadcastOp,
      linalg::GenericOp,
      linalg::MapOp,
      linalg::YieldOp,
      memref::AllocOp,
      memref::CastOp,
      memref::CopyOp,
      memref::DimOp,
      memref::LoadOp,
      memref::StoreOp,
      memref::SubViewOp,
      memref::ViewOp,
      sparse_tensor::AssembleOp,
      tensor::CastOp,
      tensor::ConcatOp,
      tensor::DimOp,
      tensor::EmptyOp,
      tensor::ExtractOp,
      tensor::ExtractSliceOp,
      tensor::FromElementsOp,
      tensor::InsertOp,
      tensor::InsertSliceOp,
      tensor::ReshapeOp,
      tensor_ext::BitReverseOp
      // clang-format on
      >([&](auto op) { return typeConverter.isLegal(op); });

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace mlir::zkir::field
