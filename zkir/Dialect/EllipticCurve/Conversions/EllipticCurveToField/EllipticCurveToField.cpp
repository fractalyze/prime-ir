#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h"

#include <utility>

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "zkir/Dialect/Field/IR/FieldDialect.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"
#include "zkir/Utils/ConversionUtils.h"

namespace mlir::zkir::elliptic_curve {

#define GEN_PASS_DEF_ELLIPTICCURVETOFIELD
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h.inc"

//////////////// TYPE CONVERSION ////////////////

RankedTensorType convertAffineType(AffineType type) {
  field::PrimeFieldType baseField = type.getCurve().getA().getType();
  return RankedTensorType::get({2}, baseField);
}

RankedTensorType convertJacobianType(JacobianType type) {
  field::PrimeFieldType baseField = type.getCurve().getA().getType();
  return RankedTensorType::get({3}, baseField);
}

RankedTensorType convertXYZZType(XYZZType type) {
  field::PrimeFieldType baseField = type.getCurve().getA().getType();
  return RankedTensorType::get({4}, baseField);
}

Type convertAffineLikeType(ShapedType type) {
  if (auto affineType = dyn_cast<AffineType>(type.getElementType())) {
    return type.cloneWith(type.getShape(), convertAffineType(affineType));
  }
  return type;
}

Type convertJacobianLikeType(ShapedType type) {
  if (auto jacobianType = dyn_cast<JacobianType>(type.getElementType())) {
    return type.cloneWith(type.getShape(), convertJacobianType(jacobianType));
  }
  return type;
}

Type convertXYZZLikeType(ShapedType type) {
  if (auto xyzzType = dyn_cast<XYZZType>(type.getElementType())) {
    return type.cloneWith(type.getShape(), convertXYZZType(xyzzType));
  }
  return type;
}

Type convertPointLikeType(ShapedType type) {
  if (auto affineType = dyn_cast<AffineType>(type.getElementType())) {
    return type.cloneWith(type.getShape(), convertAffineType(affineType));
  } else if (auto jacobianType =
                 dyn_cast<JacobianType>(type.getElementType())) {
    return type.cloneWith(type.getShape(), convertJacobianType(jacobianType));
  } else if (auto xyzzType = dyn_cast<XYZZType>(type.getElementType())) {
    return type.cloneWith(type.getShape(), convertXYZZType(xyzzType));
  }
  return type;
}

class EllipticCurveToFieldTypeConverter : public TypeConverter {
 public:
  explicit EllipticCurveToFieldTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion(
        [](AffineType type) -> Type { return convertAffineType(type); });
    addConversion(
        [](JacobianType type) -> Type { return convertJacobianType(type); });
    addConversion([](XYZZType type) -> Type { return convertXYZZType(type); });
    addConversion(
        [](ShapedType type) -> Type { return convertPointLikeType(type); });
  }
};

struct ConvertPoint : public OpConversionPattern<PointOp> {
  explicit ConvertPoint(MLIRContext *context)
      : OpConversionPattern<PointOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PointOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto makePoint =
        b.create<tensor::FromElementsOp>(op.getLoc(), op.getOperands());
    rewriter.replaceOp(op, makePoint);
    return success();
  }
};

namespace rewrites {
// In an inner namespace to avoid conflicts with canonicalization patterns
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.cpp.inc"
}  // namespace rewrites

struct EllipticCurveToField
    : impl::EllipticCurveToFieldBase<EllipticCurveToField> {
  using EllipticCurveToFieldBase::EllipticCurveToFieldBase;

  void runOnOperation() override;
};

void EllipticCurveToField::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  EllipticCurveToFieldTypeConverter typeConverter(context);

  ConversionTarget target(*context);
  target.addIllegalDialect<EllipticCurveDialect>();
  target.addLegalDialect<field::FieldDialect>();

  RewritePatternSet patterns(context);
  rewrites::populateWithGenerated(patterns);
  patterns.add<ConvertPoint>(typeConverter, context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace mlir::zkir::elliptic_curve
