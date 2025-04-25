#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h"

#include <utility>

#include "mlir/Dialect/SCF/IR/SCF.h"
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

static RankedTensorType convertAffineType(AffineType type) {
  field::PrimeFieldType baseField = type.getCurve().getA().getType();
  return RankedTensorType::get({2}, baseField);
}

static RankedTensorType convertJacobianType(JacobianType type) {
  field::PrimeFieldType baseField = type.getCurve().getA().getType();
  return RankedTensorType::get({3}, baseField);
}

static RankedTensorType convertXYZZType(XYZZType type) {
  field::PrimeFieldType baseField = type.getCurve().getA().getType();
  return RankedTensorType::get({4}, baseField);
}

static Type convertAffineLikeType(ShapedType type) {
  if (auto affineType = dyn_cast<AffineType>(type.getElementType())) {
    return type.cloneWith(type.getShape(), convertAffineType(affineType));
  }
  return type;
}

static Type convertJacobianLikeType(ShapedType type) {
  if (auto jacobianType = dyn_cast<JacobianType>(type.getElementType())) {
    return type.cloneWith(type.getShape(), convertJacobianType(jacobianType));
  }
  return type;
}

static Type convertXYZZLikeType(ShapedType type) {
  if (auto xyzzType = dyn_cast<XYZZType>(type.getElementType())) {
    return type.cloneWith(type.getShape(), convertXYZZType(xyzzType));
  }
  return type;
}

static Type convertPointLikeType(ShapedType type) {
  Type elementType = type.getElementType();
  if (isa<AffineType>(elementType)) {
    return convertAffineLikeType(type);
  } else if (isa<JacobianType>(elementType)) {
    return convertJacobianLikeType(type);
  } else if (isa<XYZZType>(elementType)) {
    return convertXYZZLikeType(type);
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

struct ConvertExtract : public OpConversionPattern<ExtractOp> {
  explicit ConvertExtract(MLIRContext *context)
      : OpConversionPattern<ExtractOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ExtractOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

// `point` must be from a tensor::from_elements op
static Value convertConvertPointTypeImpl(Value point, Type inputType,
                                         Type outputType,
                                         ImplicitLocOpBuilder b) {
  auto zero = b.create<arith::ConstantIndexOp>(0);
  auto one = b.create<arith::ConstantIndexOp>(1);
  auto x = b.create<tensor::ExtractOp>(point, ValueRange{zero});
  auto y = b.create<tensor::ExtractOp>(point, ValueRange{one});
  auto pfType = cast<field::PrimeFieldType>(x.getType());

  SmallVector<Value> outputCoords;

  if (isa<AffineType>(inputType)) {
    auto onePF = b.create<field::ConstantOp>(pfType, 1);

    // affine to jacobian
    // (x, y) -> (x, y, 1)
    outputCoords = {x, y, onePF};
    if (isa<XYZZType>(outputType)) {
      outputCoords.push_back(onePF);
      // affine to xyzz
      // (x, y) -> (x, y, 1, 1)
    }
  } else if (isa<JacobianType>(inputType)) {
    auto two = b.create<arith::ConstantIndexOp>(2);
    auto z = b.create<tensor::ExtractOp>(point, ValueRange{two});
    auto zz = b.create<field::SquareOp>(z);
    auto zzz = b.create<field::MulOp>(zz, z);

    if (isa<AffineType>(outputType)) {
      // jacobian to affine
      // (x, y, z) -> (x/z², y/z³)
      auto zero = b.create<field::ConstantOp>(pfType, 0);
      auto cmpEq = b.create<field::CmpOp>(arith::CmpIPredicate::eq, z, zero);
      // if z == 0, then x/z² -> 1, y/z³ -> 1
      auto output = b.create<scf::IfOp>(
          cmpEq,
          /*thenBuilder=*/
          [&](OpBuilder &builder, Location loc) {
            auto onePF = builder.create<field::ConstantOp>(loc, pfType, 1);
            builder.create<scf::YieldOp>(loc, ValueRange{onePF, onePF});
          },
          /*elseBuilder=*/
          [&](OpBuilder &builder, Location loc) {
            // TODO(ashjeong): use Batch Inverse
            auto zzInv = builder.create<field::InverseOp>(loc, zz);
            auto zzzInv = builder.create<field::InverseOp>(loc, zzz);
            auto newX = builder.create<field::MulOp>(loc, x, zzInv);
            auto newY = builder.create<field::MulOp>(loc, y, zzzInv);
            builder.create<scf::YieldOp>(loc, ValueRange{newX, newY});
          });

      outputCoords = {output.getResult(0), output.getResult(1)};
    } else {
      // jacobian to xyzz
      // (x, y, z) -> (x, y, z², z³)
      outputCoords = {x, y, zz, zzz};
    }
  } else {
    auto two = b.create<arith::ConstantIndexOp>(2);
    auto three = b.create<arith::ConstantIndexOp>(3);
    auto zz = b.create<tensor::ExtractOp>(point, ValueRange{two});
    auto zzz = b.create<tensor::ExtractOp>(point, ValueRange{three});

    if (isa<AffineType>(outputType)) {
      // xyzz to affine
      // (x, y, z², z³) -> (x/z², y/z³)
      auto zero = b.create<field::ConstantOp>(pfType, 0);
      auto cmpEq = b.create<field::CmpOp>(arith::CmpIPredicate::eq, zz, zero);
      // if z == 0, then x/z² -> 1, y/z³ -> 1
      auto ifOp = b.create<scf::IfOp>(
          cmpEq,
          /*thenBuilder=*/
          [&](OpBuilder &builder, Location loc) {
            auto onePF = builder.create<field::ConstantOp>(loc, pfType, 1);
            builder.create<scf::YieldOp>(loc, ValueRange{onePF, onePF});
          },
          /*elseBuilder=*/
          [&](OpBuilder &builder, Location loc) {
            // TODO(ashjeong): use Batch Inverse
            auto zzInv = builder.create<field::InverseOp>(loc, zz);
            auto zzzInv = builder.create<field::InverseOp>(loc, zzz);
            auto newX = builder.create<field::MulOp>(loc, x, zzInv);
            auto newY = builder.create<field::MulOp>(loc, y, zzzInv);
            builder.create<scf::YieldOp>(loc, ValueRange{newX, newY});
          });
      outputCoords = {ifOp.getResult(0), ifOp.getResult(1)};
    } else {
      // xyzz to jacobian
      // (x, y, z², z³) -> (x, y, z)
      outputCoords = {x, y};

      auto zero = b.create<field::ConstantOp>(pfType, 0);
      auto cmpEq = b.create<field::CmpOp>(arith::CmpIPredicate::eq, zz, zero);
      auto output = b.create<scf::IfOp>(
          cmpEq,
          /*thenBuilder=*/
          [&](OpBuilder &builder, Location loc) {
            auto zeroPF = builder.create<field::ConstantOp>(loc, pfType, 0);
            builder.create<scf::YieldOp>(loc, ValueRange{zeroPF});
          },
          /*elseBuilder=*/
          [&](OpBuilder &builder, Location loc) {
            auto zzInv = builder.create<field::InverseOp>(loc, zz);
            auto z = builder.create<field::MulOp>(loc, zzz, zzInv);
            builder.create<scf::YieldOp>(loc, ValueRange{z});
          });
      outputCoords.push_back(output.getResult(0));
    }
  }
  return b.create<tensor::FromElementsOp>(outputCoords);
}

struct ConvertConvertPointType
    : public OpConversionPattern<ConvertPointTypeOp> {
  explicit ConvertConvertPointType(MLIRContext *context)
      : OpConversionPattern<ConvertPointTypeOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConvertPointTypeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value point = adaptor.getInput();
    Type inputType = op.getInput().getType();
    Type outputType = op.getOutput().getType();

    rewriter.replaceOp(
        op, convertConvertPointTypeImpl(point, inputType, outputType, b));
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
  patterns.add<ConvertPoint, ConvertExtract, ConvertConvertPointType>(
      typeConverter, context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace mlir::zkir::elliptic_curve
