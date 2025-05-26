#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h"

#include <utility>

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/Jacobian/Add.h"
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/Jacobian/Double.h"
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/XYZZ/Add.h"
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/XYZZ/Double.h"
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

static LogicalResult convertAffineType(AffineType type,
                                       SmallVectorImpl<Type> &converted) {
  field::PrimeFieldType baseFieldType = type.getCurve().getA().getType();
  converted.push_back(baseFieldType);
  converted.push_back(baseFieldType);
  return success();
}

static LogicalResult convertJacobianType(JacobianType type,
                                         SmallVectorImpl<Type> &converted) {
  field::PrimeFieldType baseFieldType = type.getCurve().getA().getType();
  converted.push_back(baseFieldType);
  converted.push_back(baseFieldType);
  converted.push_back(baseFieldType);
  return success();
}

static LogicalResult convertXYZZType(XYZZType type,
                                     SmallVectorImpl<Type> &converted) {
  field::PrimeFieldType baseFieldType = type.getCurve().getA().getType();
  converted.push_back(baseFieldType);
  converted.push_back(baseFieldType);
  converted.push_back(baseFieldType);
  converted.push_back(baseFieldType);
  return success();
}

template <typename T>
static T convertAffineLikeType(T type) {
  auto affineType = cast<AffineType>(type.getElementType());
  field::PrimeFieldType baseFieldType = affineType.getCurve().getA().getType();
  SmallVector<int64_t> newShape(type.getShape());
  newShape.push_back(2);
  if constexpr (std::is_same_v<T, MemRefType>) {
    return MemRefType::get(newShape, baseFieldType);
  } else {
    return type.cloneWith(newShape, baseFieldType);
  }
}

template <typename T>
static T convertJacobianLikeType(T type) {
  auto jacobianType = cast<JacobianType>(type.getElementType());
  field::PrimeFieldType baseFieldType =
      jacobianType.getCurve().getA().getType();
  SmallVector<int64_t> newShape(type.getShape());
  newShape.push_back(3);
  if constexpr (std::is_same_v<T, MemRefType>) {
    return MemRefType::get(newShape, baseFieldType);
  } else {
    return type.cloneWith(newShape, baseFieldType);
  }
}

template <typename T>
static T convertXYZZLikeType(T type) {
  auto xyzzType = cast<XYZZType>(type.getElementType());
  field::PrimeFieldType baseFieldType = xyzzType.getCurve().getA().getType();
  SmallVector<int64_t> newShape(type.getShape());
  newShape.push_back(4);
  if constexpr (std::is_same_v<T, MemRefType>) {
    return MemRefType::get(newShape, baseFieldType);
  } else {
    return type.cloneWith(newShape, baseFieldType);
  }
}

template <typename T>
static T convertPointLikeType(T type) {
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
        [](AffineType type, SmallVectorImpl<Type> &converted) -> LogicalResult {
          return convertAffineType(type, converted);
        });
    addConversion([](JacobianType type,
                     SmallVectorImpl<Type> &converted) -> LogicalResult {
      return convertJacobianType(type, converted);
    });
    addConversion(
        [](XYZZType type, SmallVectorImpl<Type> &converted) -> LogicalResult {
          return convertXYZZType(type, converted);
        });
    addConversion(
        [](ShapedType type) -> Type { return convertPointLikeType(type); });
    addConversion(
        [](MemRefType type) -> Type { return convertPointLikeType(type); });
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

    rewriter.replaceOpWithMultiple(op, {adaptor.getCoords()});
    return success();
  }
};

/// In lowered form, a point is a tensor of prime field values. Creating a
/// set of lowered form points therefore requires a 2D tensor.
struct ConvertPointSet : public OpConversionPattern<PointSetOp> {
  explicit ConvertPointSet(MLIRContext *context)
      : OpConversionPattern<PointSetOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PointSetOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Lowered point (e.g. tensor<3x!PF> [Jacobian])
    Value loweredPoint = adaptor.getOperands()[0];
    RankedTensorType loweredPointType =
        cast<RankedTensorType>(loweredPoint.getType());
    field::PrimeFieldType baseFieldType =
        cast<field::PrimeFieldType>(loweredPointType.getElementType());
    unsigned numCoords =
        cast<RankedTensorType>(adaptor.getPoints()[0].getType()).getShape()[0];

    // Reshape point tensors to 2D (e.g. tensor<3x!PF> -> tensor<1x3x!PF>)
    RankedTensorType outputType =
        RankedTensorType::get({1, numCoords}, baseFieldType);
    SmallVector<Value> _outputShape(2);

    _outputShape[0] = b.create<arith::ConstantIndexOp>(1);
    _outputShape[1] = b.create<arith::ConstantIndexOp>(numCoords);
    auto outputShape = b.create<tensor::FromElementsOp>(_outputShape);

    size_t numPoints = op.getNumOperands();
    SmallVector<Value> expandedPoints(numPoints);
    for (size_t i = 0; i < numPoints; ++i) {
      expandedPoints[i] = b.create<tensor::ReshapeOp>(
          outputType,                // Result type of the operation
          adaptor.getOperands()[i],  // The input lowered point tensor
          outputShape);              // The output 2D tensor shape
    }

    // Concat all 2D point tensors together for final 2D set of point tensors
    RankedTensorType totalOutputType =
        RankedTensorType::get({op.getNumOperands(), numCoords}, baseFieldType);

    auto pointSet =
        b.create<tensor::ConcatOp>(totalOutputType, 0, expandedPoints);
    rewriter.replaceOp(op, pointSet);
    return success();
  }
};

/// The lowered form of point set is a 2D tensor of prime field values, while
/// the ec form is a 1D tensor. Extracting the value must be done specially for
/// the lowered version.
struct ConvertPointSetExtract : public OpConversionPattern<PointSetExtractOp> {
  explicit ConvertPointSetExtract(MLIRContext *context)
      : OpConversionPattern<PointSetExtractOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      PointSetExtractOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // 2D tensor of prime field values, e.g.:
    // |x1, y1, z1|
    // |x2, y2, z2|
    Value loweredPointSet = adaptor.getPointSet();
    Value index = adaptor.getIndex();
    // e.g. tensor<2x3x!PF> (set of jacobian points)
    auto loweredPointSetType =
        cast<RankedTensorType>(loweredPointSet.getType());
    unsigned numCoords = loweredPointSetType.getShape()[1];

    auto zero = b.create<arith::ConstantIndexOp>(0);
    auto one = b.create<arith::ConstantIndexOp>(1);
    auto sz = b.create<arith::ConstantIndexOp>(numCoords);
    SmallVector<Value> offsets{index, zero};
    SmallVector<Value> sizes{one, sz};
    SmallVector<Value> strides{one, one};
    // e.g. tensor<1x3x!PF> (jacobian point in 2D)
    auto higherRankedLoweredPoint = b.create<tensor::ExtractSliceOp>(
        loweredPointSet, offsets, sizes, strides);

    SmallVector<Value> _outputShape{sz};
    auto outputShape = b.create<tensor::FromElementsOp>(_outputShape);

    // e.g. tensor<3x!PF> (jacobian point in 1D)
    auto point = b.create<tensor::ReshapeOp>(
        RankedTensorType::get({numCoords},
                              loweredPointSetType.getElementType()),
        higherRankedLoweredPoint, outputShape);
    rewriter.replaceOp(op, point);
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
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

// `point` must be from a tensor::from_elements op
static SmallVector<Value> convertConvertPointTypeImpl(ValueRange coords,
                                                      Type inputType,
                                                      Type outputType,
                                                      ImplicitLocOpBuilder &b) {
  auto baseFieldType = cast<field::PrimeFieldType>(coords[0].getType());
  SmallVector<Value> outputCoords;

  if (isa<AffineType>(inputType)) {
    auto onePF = b.create<field::ConstantOp>(baseFieldType, 1);

    // affine to jacobian
    // (x, y) -> (x, y, 1)
    outputCoords = {coords[0], coords[1], onePF};
    if (isa<XYZZType>(outputType)) {
      outputCoords.push_back(onePF);
      // affine to xyzz
      // (x, y) -> (x, y, 1, 1)
    }
  } else if (isa<JacobianType>(inputType)) {
    auto zz = b.create<field::SquareOp>(coords[2]);
    auto zzz = b.create<field::MulOp>(zz, coords[2]);

    if (isa<AffineType>(outputType)) {
      // jacobian to affine
      // (x, y, z) -> (x/z², y/z³)
      auto zero = b.create<field::ConstantOp>(baseFieldType, 0);
      auto cmpEq =
          b.create<field::CmpOp>(arith::CmpIPredicate::eq, coords[2], zero);
      // if z == 0, then x/z² -> 1, y/z³ -> 1
      auto output = b.create<scf::IfOp>(
          cmpEq,
          /*thenBuilder=*/
          [&](OpBuilder &builder, Location loc) {
            auto onePF =
                builder.create<field::ConstantOp>(loc, baseFieldType, 1);
            builder.create<scf::YieldOp>(loc, ValueRange{onePF, onePF});
          },
          /*elseBuilder=*/
          [&](OpBuilder &builder, Location loc) {
            // TODO(ashjeong): use Batch Inverse
            auto zzInv = builder.create<field::InverseOp>(loc, zz);
            auto zzzInv = builder.create<field::InverseOp>(loc, zzz);
            auto newX = builder.create<field::MulOp>(loc, coords[0], zzInv);
            auto newY = builder.create<field::MulOp>(loc, coords[1], zzzInv);
            builder.create<scf::YieldOp>(loc, ValueRange{newX, newY});
          });

      outputCoords = {output.getResult(0), output.getResult(1)};
    } else {
      // jacobian to xyzz
      // (x, y, z) -> (x, y, z², z³)
      outputCoords = {coords[0], coords[1], zz, zzz};
    }
  } else {
    if (isa<AffineType>(outputType)) {
      // xyzz to affine
      // (x, y, z², z³) -> (x/z², y/z³)
      auto zero = b.create<field::ConstantOp>(baseFieldType, 0);
      auto cmpEq =
          b.create<field::CmpOp>(arith::CmpIPredicate::eq, coords[2], zero);
      // if z == 0, then x/z² -> 1, y/z³ -> 1
      auto ifOp = b.create<scf::IfOp>(
          cmpEq,
          /*thenBuilder=*/
          [&](OpBuilder &builder, Location loc) {
            auto onePF =
                builder.create<field::ConstantOp>(loc, baseFieldType, 1);
            builder.create<scf::YieldOp>(loc, ValueRange{onePF, onePF});
          },
          /*elseBuilder=*/
          [&](OpBuilder &builder, Location loc) {
            // TODO(ashjeong): use Batch Inverse
            auto zzInv = builder.create<field::InverseOp>(loc, coords[2]);
            auto zzzInv = builder.create<field::InverseOp>(loc, coords[3]);
            auto newX = builder.create<field::MulOp>(loc, coords[0], zzInv);
            auto newY = builder.create<field::MulOp>(loc, coords[1], zzzInv);
            builder.create<scf::YieldOp>(loc, ValueRange{newX, newY});
          });
      outputCoords = {ifOp.getResult(0), ifOp.getResult(1)};
    } else {
      // xyzz to jacobian
      // (x, y, z², z³) -> (x, y, z)
      outputCoords = {coords[0], coords[1]};

      auto zero = b.create<field::ConstantOp>(baseFieldType, 0);
      auto cmpEq =
          b.create<field::CmpOp>(arith::CmpIPredicate::eq, coords[2], zero);
      auto output = b.create<scf::IfOp>(
          cmpEq,
          /*thenBuilder=*/
          [&](OpBuilder &builder, Location loc) {
            auto zeroPF =
                builder.create<field::ConstantOp>(loc, baseFieldType, 0);
            builder.create<scf::YieldOp>(loc, ValueRange{zeroPF});
          },
          /*elseBuilder=*/
          [&](OpBuilder &builder, Location loc) {
            auto zzInv = builder.create<field::InverseOp>(loc, coords[2]);
            auto z = builder.create<field::MulOp>(loc, coords[3], zzInv);
            builder.create<scf::YieldOp>(loc, ValueRange{z});
          });
      outputCoords.push_back(output.getResult(0));
    }
  }
  return outputCoords;
}

struct ConvertConvertPointType
    : public OpConversionPattern<ConvertPointTypeOp> {
  explicit ConvertConvertPointType(MLIRContext *context)
      : OpConversionPattern<ConvertPointTypeOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConvertPointTypeOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    ValueRange points = adaptor.getInput();
    Type inputType = op.getInput().getType();
    Type outputType = op.getOutput().getType();

    rewriter.replaceOpWithMultiple(
        op, {convertConvertPointTypeImpl(points, inputType, outputType, b)});
    return success();
  }
};

///////////// POINT ARITHMETIC OPERATIONS //////////////

// `p1` and `p2` must be tensor::from_elements ops
static Value convertAddImpl(Value p1, Value p2, Type p1Type, Type p2Type,
                            Type outputType, ImplicitLocOpBuilder &b) {
  if (isa<XYZZType>(outputType)) {
    return xyzzAdd(p1, p2, p1Type, p2Type, b);
  } else if (isa<JacobianType>(outputType)) {
    return jacobianAdd(p1, p2, p1Type, p2Type, b);
  } else {
    assert(false && "Unsupported point types for addition");
  }
}

struct ConvertAdd : public OpConversionPattern<AddOp> {
  explicit ConvertAdd(MLIRContext *context)
      : OpConversionPattern<AddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // `p1` and `p2` are tensor::from_elements ops
    Value p1 = adaptor.getLhs();
    Value p2 = adaptor.getRhs();
    Type p1Type = op.getLhs().getType();
    Type p2Type = op.getRhs().getType();
    Type outputType = op.getOutput().getType();

    rewriter.replaceOp(op,
                       convertAddImpl(p1, p2, p1Type, p2Type, outputType, b));
    return success();
  }
};

// `point` must be from a tensor::from_elements op
static Value convertDoubleImpl(Value point, Type inputType, Type outputType,
                               ImplicitLocOpBuilder &b) {
  if (isa<XYZZType>(outputType)) {
    return xyzzDouble(point, outputType, b);
  } else if (isa<JacobianType>(outputType)) {
    return jacobianDouble(point, outputType, b);
  } else {
    assert(false && "Unsupported point types for doubling");
  }
}

struct ConvertDouble : public OpConversionPattern<DoubleOp> {
  explicit ConvertDouble(MLIRContext *context)
      : OpConversionPattern<DoubleOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      DoubleOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value point = adaptor.getInput();
    Type inputType = op.getInput().getType();
    Type outputType = op.getOutput().getType();

    rewriter.replaceOp(op, convertDoubleImpl(point, inputType, outputType, b));
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

    Value point = adaptor.getInput();
    Type inputType = op.getInput().getType();

    auto zero = b.create<arith::ConstantIndexOp>(0);
    auto one = b.create<arith::ConstantIndexOp>(1);
    auto x = b.create<tensor::ExtractOp>(point, ValueRange{zero});
    auto y = b.create<tensor::ExtractOp>(point, ValueRange{one});

    auto negatedY = b.create<field::NegateOp>(y);
    SmallVector<Value> outputCoords{x, negatedY};

    if (isa<JacobianType>(inputType)) {
      auto two = b.create<arith::ConstantIndexOp>(2);
      auto z = b.create<tensor::ExtractOp>(point, ValueRange{two});
      outputCoords.push_back(z);
    } else if (isa<XYZZType>(inputType)) {
      auto two = b.create<arith::ConstantIndexOp>(2);
      auto three = b.create<arith::ConstantIndexOp>(3);
      auto zz = b.create<tensor::ExtractOp>(point, ValueRange{two});
      auto zzz = b.create<tensor::ExtractOp>(point, ValueRange{three});
      outputCoords.push_back(zz);
      outputCoords.push_back(zzz);
    }
    Value makePoint = b.create<tensor::FromElementsOp>(outputCoords);

    rewriter.replaceOp(op, makePoint);
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

    Value negP2 = b.create<elliptic_curve::NegateOp>(op.getRhs());
    Value result = b.create<elliptic_curve::AddOp>(op.getOutput().getType(),
                                                   op.getLhs(), negP2);

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Currently implements Double-and-Add algorithm
// TODO(ashjeong): implement GLV
static Value convertScalarMulImpl(Value point, Value scalarPF, Type pointType,
                                  Type outputType, ImplicitLocOpBuilder &b) {
  auto baseFieldType = cast<field::PrimeFieldType>(scalarPF.getType());
  unsigned outputBitWidth = baseFieldType.getModulus().getValue().getBitWidth();
  auto signlessIntType =
      IntegerType::get(b.getContext(), outputBitWidth, IntegerType::Signless);
  auto scalar = b.create<field::ExtractOp>(signlessIntType, scalarPF);

  auto zeroPF = b.create<field::ConstantOp>(baseFieldType, 0);
  SmallVector<Value> zeroes{zeroPF, zeroPF, zeroPF};
  Value initalPoint;

  if (isa<XYZZType>(outputType)) {
    zeroes.push_back(zeroPF);
    initalPoint = point;
  } else if (isa<AffineType>(pointType)) {
    initalPoint = convertConvertPointTypeImpl(point, pointType, outputType, b);
  } else {
    initalPoint = point;
  }
  Value zeroPoint = b.create<tensor::FromElementsOp>(zeroes);

  // `decreasingScalar` set to `scalar` and `result` set to zero point.
  auto whileOp = b.create<scf::WhileOp>(
      /*resultTypes=*/
      TypeRange({signlessIntType, initalPoint.getType(), zeroPoint.getType()}),
      /*operands=*/
      ValueRange({scalar, initalPoint, zeroPoint}),
      /*beforeBuilder=*/
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
        auto arithZero = b.create<arith::ConstantIntOp>(0, signlessIntType);
        // if `decreasingScalar` > 0, continue
        Value decreasingScalar = args[0];
        auto cmpGt = b.create<arith::CmpIOp>(arith::CmpIPredicate::ugt,
                                             decreasingScalar, arithZero);
        b.create<scf::ConditionOp>(cmpGt, args);
      },
      /*afterBuilder=*/
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
        auto arithOne = b.create<arith::ConstantIntOp>(1, signlessIntType);
        Value decreasingScalar = args[0];
        Value multiplyingPoint = args[1];
        Value result = args[2];

        // if `decreasingScalar` % 1 == 1...
        auto bitAdd = b.create<arith::AndIOp>(decreasingScalar, arithOne);
        auto cmpEq =
            b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, bitAdd, arithOne);
        auto ifOp = b.create<scf::IfOp>(
            cmpEq,
            // ...then add `multiplyingPoint` to `result`
            /*thenBuilder=*/
            [&](OpBuilder &builder, Location loc) {
              ImplicitLocOpBuilder b(loc, builder);
              Value innerResult =
                  convertAddImpl(result, multiplyingPoint, outputType,
                                 outputType, outputType, b);
              b.create<scf::YieldOp>(innerResult);
            },
            /*elseBuilder=*/
            [&](OpBuilder &builder, Location loc) {
              b.create<scf::YieldOp>(result);
            });
        // double `multiplyingPoint`
        Value doubledPoint =
            convertDoubleImpl(multiplyingPoint, outputType, outputType, b);
        // right shift `decreasingScalar` by 1
        decreasingScalar = b.create<arith::ShRUIOp>(decreasingScalar, arithOne);

        // See here for more info:
        // https://github.com/iree-org/iree/issues/16956
        auto multPointBufferOp =
            b.create<bufferization::MaterializeInDestinationOp>(
                doubledPoint, multiplyingPoint);
        auto resultBufferOp =
            b.create<bufferization::MaterializeInDestinationOp>(
                ifOp.getResult(0), result);

        b.create<scf::YieldOp>(
            ValueRange({decreasingScalar, multPointBufferOp.getResult(),
                        resultBufferOp.getResult()}));
      });
  return whileOp.getResult(2);
}

struct ConvertScalarMul : public OpConversionPattern<ScalarMulOp> {
  explicit ConvertScalarMul(MLIRContext *context)
      : OpConversionPattern<ScalarMulOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ScalarMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value point = adaptor.getPoint();
    Value scalarPF = op.getScalar();

    Type pointType = op.getPoint().getType();
    Type outputType = op.getOutput().getType();

    Value scalarMul =
        convertScalarMulImpl(point, scalarPF, pointType, outputType, b);
    rewriter.replaceOp(op, scalarMul);
    return success();
  }
};

struct ConvertMSM : public OpConversionPattern<MSMOp> {
  explicit ConvertMSM(mlir::MLIRContext *context)
      : OpConversionPattern<MSMOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MSMOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // 2D tensor of prime field values, e.g.:
    // |x1, y1, z1|
    // |x2, y2, z2|
    Value loweredPointSet = adaptor.getPoints();
    // 1d tensor of PF, e.g.:
    // | s1 , s2 |
    Value scalars = op.getScalars();
    RankedTensorType loweredPointSetType =
        cast<RankedTensorType>(loweredPointSet.getType());
    field::PrimeFieldType baseFieldType =
        cast<field::PrimeFieldType>(loweredPointSetType.getElementType());
    unsigned numScalarMuls = loweredPointSetType.getShape()[0];
    unsigned numCoords = loweredPointSetType.getShape()[1];

    Type inputPointType =
        cast<RankedTensorType>(op.getPoints().getType()).getElementType();
    Type outputPointType = op.getOutput().getType();
    RankedTensorType loweredOutputPointType =
        RankedTensorType::get({numCoords}, baseFieldType);

    Value accumulator;
    auto zero = b.create<arith::ConstantIndexOp>(0);
    auto one = b.create<arith::ConstantIndexOp>(1);
    auto sz = b.create<arith::ConstantIndexOp>(numCoords);
    SmallVector<Value> sizes{one, sz};
    SmallVector<Value> strides{one, one};
    for (size_t i = 0; i < numScalarMuls; ++i) {
      auto idx = b.create<arith::ConstantIndexOp>(i);

      // scalar
      auto scalar = b.create<tensor::ExtractOp>(scalars, ValueRange{idx});

      // point
      //  - extract point = tensor<2x3x!PF> -> tensor<1x3x!PF>
      SmallVector<Value> offsets{idx, zero};
      auto higherRankedPoint = b.create<tensor::ExtractSliceOp>(
          loweredPointSet, offsets, sizes, strides);
      // - reshape point = tensor<1x3x!PF> -> tensor<3x!PF>
      SmallVector<Value> _outputShape{sz};
      auto outputShape = b.create<tensor::FromElementsOp>(_outputShape);
      auto point = b.create<tensor::ReshapeOp>(loweredOutputPointType,
                                               higherRankedPoint, outputShape);

      Value adder = convertScalarMulImpl(point, scalar, inputPointType,
                                         outputPointType, b);
      if (i != 0) {
        accumulator = convertAddImpl(accumulator, adder, outputPointType,
                                     outputPointType, outputPointType, b);
      } else {
        accumulator = adder;
      }
    }
    rewriter.replaceOp(op, accumulator);
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
  patterns
      .add<ConvertPoint, ConvertPointSet, ConvertPointSetExtract,
           ConvertExtract, ConvertConvertPointType, ConvertAdd, ConvertDouble,
           ConvertNegate, ConvertSub, ConvertScalarMul, ConvertMSM>(
          typeConverter, context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace mlir::zkir::elliptic_curve
