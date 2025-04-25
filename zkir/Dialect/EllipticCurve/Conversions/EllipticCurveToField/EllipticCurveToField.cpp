#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h"

#include <utility>

#include "mlir/Transforms/DialectConversion.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "zkir/Dialect/Field/IR/FieldDialect.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"

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
  EllipticCurveToFieldTypeConverter typeConverter(context);

  ConversionTarget target(*context);
  target.addIllegalDialect<EllipticCurveDialect>();
  target.addLegalDialect<field::FieldDialect>();
}

}  // namespace mlir::zkir::elliptic_curve
