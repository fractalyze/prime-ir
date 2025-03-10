#include "zkir/Dialect/Poly/Conversions/PolyToField/PolyToField.h"

#include <utility>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "zkir/Dialect/Field/IR/FieldAttributes.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkir/Dialect/Poly/IR/PolyOps.h"
#include "zkir/Dialect/Poly/IR/PolyTypes.h"
#include "zkir/Utils/ConversionUtils.h"

namespace mlir::zkir::poly {

#define GEN_PASS_DEF_POLYTOFIELD
#include "zkir/Dialect/Poly/Conversions/PolyToField/PolyToField.h.inc"

RankedTensorType convertPolyType(PolyType type) {
  int64_t maxDegree = type.getMaxDegree().getValue().getSExtValue();
  return RankedTensorType::get({static_cast<int64_t>(maxDegree + 1)},
                               type.getBaseField());
}

class PolyToFieldTypeConverter : public TypeConverter {
 public:
  explicit PolyToFieldTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([](PolyType type) -> Type { return convertPolyType(type); });
  }
};

struct CommonConversionInfo {
  PolyType polyType;

  field::PrimeFieldType coefficientType;
  Type coefficientStorageType;

  RankedTensorType tensorType;
};

FailureOr<CommonConversionInfo> getCommonConversionInfo(
    Operation *op, const TypeConverter *typeConverter) {
  // Most ops have a single result type that is a polynomial
  PolyType polyTy = dyn_cast<PolyType>(op->getResult(0).getType());

  if (!polyTy) {
    op->emitError(
        "Can't directly lower for a tensor of polynomials. "
        "First run --convert-elementwise-to-affine.");
    return failure();
  }

  CommonConversionInfo info;
  info.polyType = polyTy;
  info.coefficientType =
      llvm::dyn_cast<field::PrimeFieldType>(polyTy.getBaseField());
  if (!info.coefficientType) {
    op->emitError("Polynomial base field must be of field type");
    return failure();
  }
  info.tensorType = cast<RankedTensorType>(typeConverter->convertType(polyTy));
  info.coefficientStorageType = info.coefficientType.getModulus().getType();
  return std::move(info);
}

struct ConvertConstant : public OpConversionPattern<ConstantOp> {
  explicit ConvertConstant(mlir::MLIRContext *context)
      : OpConversionPattern<ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto res = getCommonConversionInfo(op, typeConverter);
    if (failed(res)) return failure();
    auto typeInfo = res.value();

    auto uniPolyAttr = dyn_cast<UnivariatePolyAttr>(op.getValue());
    if (!uniPolyAttr) return failure();
    SmallVector<Attribute> coeffs;
    Type eltStorageType = typeInfo.coefficientStorageType;

    // Create all the attributes as arith types since mod_arith.constant
    // doesn't support tensor attribute inputs. Instead we
    // mod_arith.encapsulate them.
    //
    // This is inefficient for large-degree polys, but as of this writing we
    // don't have a lowering that uses a sparse representation.
    unsigned numTerms = typeInfo.tensorType.getShape()[0];
    coeffs.reserve(numTerms);
    for (size_t i = 0; i < numTerms; ++i) {
      coeffs.push_back(IntegerAttr::get(eltStorageType, 0));
    }

    // WARNING: if you don't store the IntPolynomial as an intermediate value
    // before iterating over the terms, you will get a use-after-free bug. See
    // the "Temporary range expression" section in
    // https://en.cppreference.com/w/cpp/language/range-for
    const polynomial::IntPolynomial &poly =
        uniPolyAttr.getValue().getPolynomial();
    for (const auto &term : poly.getTerms()) {
      int64_t idx = term.getExponent().getSExtValue();
      APInt coeff = term.getCoefficient();
      APInt modulus = typeInfo.polyType.getBaseField().getModulus().getValue();
      // APInt `srem` gives remainder with sign matching the sign of the
      // coefficient
      coeff = coeff.sextOrTrunc(modulus.getBitWidth()).srem(modulus);
      if (coeff.isNegative()) {
        // We need to add the modulus to get the positive remainder.
        coeff += modulus;
      }
      assert(coeff.sge(0));
      coeffs[idx] = IntegerAttr::get(eltStorageType, coeff.getZExtValue());
    }

    auto intTensorType =
        RankedTensorType::get(typeInfo.tensorType.getShape(),
                              typeInfo.coefficientType.getModulus().getType());
    auto constOp = b.create<arith::ConstantOp>(
        DenseElementsAttr::get(intTensorType, coeffs));
    rewriter.replaceOpWithNewOp<field::EncapsulateOp>(op, typeInfo.tensorType,
                                                      constOp.getResult());
    return success();
  }
};

template <typename SourceOp, typename TargetFieldOp>
struct ConvertPolyBinOp : public OpConversionPattern<SourceOp> {
  explicit ConvertPolyBinOp(mlir::MLIRContext *context)
      : OpConversionPattern<SourceOp>(context) {}

  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op, typename SourceOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (PolyType poly_ty = llvm::dyn_cast<PolyType>(op.getResult().getType())) {
      ImplicitLocOpBuilder b(op.getLoc(), rewriter);
      auto result = b.create<TargetFieldOp>(adaptor.getLhs(), adaptor.getRhs());
      rewriter.replaceOp(op, result);
      return success();
    }
    return failure();
  }
};

struct ConvertToTensor : public OpConversionPattern<ToTensorOp> {
  explicit ConvertToTensor(mlir::MLIRContext *context)
      : OpConversionPattern<ToTensorOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ToTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0].getDefiningOp());
    return success();
  }
};

struct ConvertFromTensor : public OpConversionPattern<FromTensorOp> {
  explicit ConvertFromTensor(mlir::MLIRContext *context)
      : OpConversionPattern<FromTensorOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FromTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto res = getCommonConversionInfo(op, typeConverter);
    if (failed(res)) return failure();
    auto typeInfo = res.value();

    auto resultShape = typeInfo.tensorType.getShape()[0];
    auto inputTensorTy = op.getInput().getType();
    auto inputShape = inputTensorTy.getShape()[0];

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto coeffValue = adaptor.getInput();

    // Zero pad the tensor if the coefficients' size is less than the polynomial
    // degree.
    if (inputShape < resultShape) {
      SmallVector<OpFoldResult, 1> low, high;
      low.push_back(rewriter.getIndexAttr(0));
      high.push_back(rewriter.getIndexAttr(resultShape - inputShape));

      auto padValue = b.create<field::ConstantOp>(typeInfo.coefficientType, 0);
      coeffValue = b.create<tensor::PadOp>(typeInfo.tensorType, coeffValue, low,
                                           high, padValue,
                                           /*nofold=*/false);
    }

    rewriter.replaceOp(op, coeffValue);
    return success();
  }
};

struct PolyToField : impl::PolyToFieldBase<PolyToField> {
  using PolyToFieldBase::PolyToFieldBase;

  void runOnOperation() override;
};

void PolyToField::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  PolyToFieldTypeConverter typeConverter(context);

  ConversionTarget target(*context);

  target.addIllegalDialect<PolyDialect>();
  target.addLegalDialect<field::FieldDialect>();
  RewritePatternSet patterns(context);

  patterns.add<ConvertPolyBinOp<AddOp, field::AddOp>,
               ConvertPolyBinOp<SubOp, field::SubOp>, ConvertConstant,
               ConvertFromTensor, ConvertToTensor>(typeConverter, context);
  addStructuralConversionPatterns(typeConverter, patterns, target);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace mlir::zkir::poly
