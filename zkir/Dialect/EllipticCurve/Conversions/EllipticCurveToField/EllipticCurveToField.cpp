#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h"

#include <utility>

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
  Type baseFieldType = type.getCurve().getBaseField();
  converted.push_back(baseFieldType);
  converted.push_back(baseFieldType);
  return success();
}

static LogicalResult convertJacobianType(JacobianType type,
                                         SmallVectorImpl<Type> &converted) {
  Type baseFieldType = type.getCurve().getBaseField();
  converted.push_back(baseFieldType);
  converted.push_back(baseFieldType);
  converted.push_back(baseFieldType);
  return success();
}

static LogicalResult convertXYZZType(XYZZType type,
                                     SmallVectorImpl<Type> &converted) {
  Type baseFieldType = type.getCurve().getBaseField();
  converted.push_back(baseFieldType);
  converted.push_back(baseFieldType);
  converted.push_back(baseFieldType);
  converted.push_back(baseFieldType);
  return success();
}

template <typename T>
static T convertAffineLikeType(T type) {
  auto affineType = cast<AffineType>(type.getElementType());
  Type baseFieldType = affineType.getCurve().getBaseField();
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
  Type baseFieldType = jacobianType.getCurve().getBaseField();
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
  Type baseFieldType = xyzzType.getCurve().getBaseField();
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

struct ConvertIsZero : public OpConversionPattern<IsZeroOp> {
  explicit ConvertIsZero(MLIRContext *context)
      : OpConversionPattern<IsZeroOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IsZeroOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    ValueRange coords = adaptor.getInput();
    Type baseField =
        getCurveFromPointLike(op.getInput().getType()).getBaseField();
    Value zeroBF = b.create<field::ConstantOp>(baseField, 0);

    Value isZero;
    if (isa<AffineType>(op.getInput().getType())) {
      Value xIsZero =
          b.create<field::CmpOp>(arith::CmpIPredicate::eq, coords[0], zeroBF);
      Value yIsZero =
          b.create<field::CmpOp>(arith::CmpIPredicate::eq, coords[1], zeroBF);
      isZero = b.create<arith::AndIOp>(xIsZero, yIsZero);
    } else {
      isZero =
          b.create<field::CmpOp>(arith::CmpIPredicate::eq, coords[2], zeroBF);
    }
    rewriter.replaceOp(op, isZero);
    return success();
  }
};

struct ConvertExtract : public OpConversionPattern<ExtractOp> {
  explicit ConvertExtract(MLIRContext *context)
      : OpConversionPattern<ExtractOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ExtractOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ValueRange coordsTmp = adaptor.getInput();

    // NOTE(ashjeong): Here we attempt to restructure the input coordinates like
    // so: [[x,y,z]] to [[x],[y],[z]] (i.e. for Jacobian). A naive copy of the
    // coordinate values into a `SmallVector<ValueRange>` does not work given
    // `ValueRange's` nature, so a deep copy is needed beforehand.
    SmallVector<SmallVector<Value>> coords_copy(coordsTmp.size());
    for (size_t i = 0; i < coordsTmp.size(); ++i) {
      coords_copy[i].push_back(coordsTmp[i]);
    }
    SmallVector<ValueRange> coords(coords_copy.begin(), coords_copy.end());

    rewriter.replaceOpWithMultiple(op, coords);
    return success();
  }
};

struct ConvertConvertPointType
    : public OpConversionPattern<ConvertPointTypeOp> {
  explicit ConvertConvertPointType(MLIRContext *context)
      : OpConversionPattern<ConvertPointTypeOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConvertPointTypeOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    ValueRange coords = adaptor.getInput();
    Type inputType = op.getInput().getType();
    Type outputType = op.getOutput().getType();
    Type baseFieldType = getCurveFromPointLike(inputType).getBaseField();
    Value zeroBF = b.create<field::ConstantOp>(baseFieldType, 0);
    // TODO(chokobole): Fix below after attaching montgomery information to
    // the field type.
    Value oneBF = b.create<field::ToMontOp>(
        baseFieldType, b.create<field::ConstantOp>(baseFieldType, 1));

    SmallVector<Value> outputCoords;

    auto isZero = b.create<IsZeroOp>(op.getInput());
    if (isa<AffineType>(inputType)) {
      auto output = b.create<scf::IfOp>(
          isZero,
          /*thenBuilder=*/
          [&](OpBuilder &builder, Location loc) {
            if (isa<JacobianType>(outputType)) {
              // affine to jacobian
              // (0, 0) -> (1, 1, 0)
              builder.create<scf::YieldOp>(loc,
                                           ValueRange{oneBF, oneBF, zeroBF});
            } else {
              // affine to xyzz
              // (0, 0) -> (1, 1, 0, 0)
              builder.create<scf::YieldOp>(
                  loc, ValueRange{oneBF, oneBF, zeroBF, zeroBF});
            }
          },
          /*elseBuilder=*/
          [&](OpBuilder &builder, Location loc) {
            if (isa<JacobianType>(outputType)) {
              // affine to jacobian
              // (x, y) -> (x, y, 1)
              builder.create<scf::YieldOp>(
                  loc, ValueRange{/*x=*/coords[0], /*y=*/coords[1], oneBF});
            } else {
              // affine to xyzz
              // (x, y) -> (x, y, 1, 1)
              builder.create<scf::YieldOp>(
                  loc,
                  ValueRange{/*x=*/coords[0], /*y=*/coords[1], oneBF, oneBF});
            }
          });
      outputCoords = output.getResults();
    } else if (isa<JacobianType>(inputType)) {
      auto zz = b.create<field::SquareOp>(/*z=*/coords[2]);
      auto zzz = b.create<field::MulOp>(zz, /*z=*/coords[2]);

      if (isa<AffineType>(outputType)) {
        auto output = b.create<scf::IfOp>(
            isZero,
            /*thenBuilder=*/
            [&](OpBuilder &builder, Location loc) {
              // jacobian to affine
              // (x, y, 0) -> (0, 0)
              builder.create<scf::YieldOp>(loc, ValueRange{zeroBF, zeroBF});
            },
            /*elseBuilder=*/
            [&](OpBuilder &builder, Location loc) {
              // jacobian to affine
              // (x, y, z) -> (x/z², y/z³)
              // TODO(ashjeong): use Batch Inverse
              auto zzInv = builder.create<field::InverseOp>(loc, zz);
              auto zzzInv = builder.create<field::InverseOp>(loc, zzz);
              auto newX =
                  builder.create<field::MulOp>(loc, /*x=*/coords[0], zzInv);
              auto newY =
                  builder.create<field::MulOp>(loc, /*y=*/coords[1], zzzInv);
              builder.create<scf::YieldOp>(loc, ValueRange{newX, newY});
            });

        outputCoords = output.getResults();
      } else {
        // jacobian to xyzz
        // (x, y, z) -> (x, y, z², z³)
        outputCoords = {/*x=*/coords[0], /*y=*/coords[1], zz, zzz};
      }
    } else {
      if (isa<AffineType>(outputType)) {
        auto output = b.create<scf::IfOp>(
            isZero,
            /*thenBuilder=*/
            [&](OpBuilder &builder, Location loc) {
              // xyzz to affine
              // (x, y, 0, 0) -> (0, 0)
              builder.create<scf::YieldOp>(loc, ValueRange{zeroBF, zeroBF});
            },
            /*elseBuilder=*/
            [&](OpBuilder &builder, Location loc) {
              // xyzz to affine
              // (x, y, z², z³) -> (x/z², y/z³)
              // TODO(ashjeong): use Batch Inverse
              auto zzInv =
                  builder.create<field::InverseOp>(loc, /*zz=*/coords[2]);
              auto zzzInv =
                  builder.create<field::InverseOp>(loc, /*zzz=*/coords[3]);
              auto newX =
                  builder.create<field::MulOp>(loc, /*x=*/coords[0], zzInv);
              auto newY =
                  builder.create<field::MulOp>(loc, /*y=*/coords[1], zzzInv);
              builder.create<scf::YieldOp>(loc, ValueRange{newX, newY});
            });
        outputCoords = output.getResults();
      } else {
        outputCoords = {/*x=*/coords[0], /*y=*/coords[1]};

        auto output = b.create<scf::IfOp>(
            isZero,
            /*thenBuilder=*/
            [&](OpBuilder &builder, Location loc) {
              // xyzz to jacobian
              // (x, y, 0, 0) -> (x, y, 0)
              builder.create<scf::YieldOp>(loc, ValueRange{zeroBF});
            },
            /*elseBuilder=*/
            [&](OpBuilder &builder, Location loc) {
              // xyzz to jacobian
              // (x, y, z², z³) -> (x, y, z)
              auto zzInv = builder.create<field::InverseOp>(loc, coords[2]);
              auto z =
                  builder.create<field::MulOp>(loc, /*zzz=*/coords[3], zzInv);
              builder.create<scf::YieldOp>(loc, ValueRange{z});
            });
        outputCoords.push_back(output.getResult(0));
      }
    }
    rewriter.replaceOpWithMultiple(op, {outputCoords});
    return success();
  }
};

///////////// POINT ARITHMETIC OPERATIONS //////////////

struct ConvertAdd : public OpConversionPattern<AddOp> {
  explicit ConvertAdd(MLIRContext *context)
      : OpConversionPattern<AddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AddOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value p1 = op.getLhs();
    Value p2 = op.getRhs();
    ValueRange p1Coords = adaptor.getLhs();
    ValueRange p2Coords = adaptor.getRhs();
    Type p1Type = p1.getType();
    Type p2Type = p2.getType();
    Type outputType = op.getOutput().getType();

    // check p1 == zero point
    Value p1IsZero = b.create<elliptic_curve::IsZeroOp>(p1);
    auto output = b.create<scf::IfOp>(
        p1IsZero,
        /*thenBuilder=*/
        [&](OpBuilder &builder, Location loc) {
          ImplicitLocOpBuilder b(loc, builder);
          ValueRange retP2 = p2Coords;
          if (isa<AffineType>(p2Type)) {
            retP2 = {
                b.create<elliptic_curve::ConvertPointTypeOp>(outputType, p2)};
          }

          b.create<scf::YieldOp>(retP2);
        },
        /*elseBuilder=*/
        [&](OpBuilder &builder, Location loc) {
          ImplicitLocOpBuilder b(loc, builder);

          // check p2 == zero point
          Value p2isZero = b.create<elliptic_curve::IsZeroOp>(p2);
          auto output = b.create<scf::IfOp>(
              p2isZero,
              /*thenBuilder=*/
              [&](OpBuilder &builder, Location loc) {
                ImplicitLocOpBuilder b(loc, builder);
                ValueRange retP1 = p1Coords;
                if (isa<AffineType>(p1Type)) {
                  retP1 = {b.create<elliptic_curve::ConvertPointTypeOp>(
                      outputType, p1)};
                }

                b.create<scf::YieldOp>(retP1);
              },
              /*elseBuilder=*/
              [&](OpBuilder &builder, Location loc) {
                ImplicitLocOpBuilder b(loc, builder);
                // run default add
                SmallVector<Value> sum;
                if (auto xyzzType = dyn_cast<XYZZType>(outputType)) {
                  sum = xyzzAdd(p1Coords, p2Coords, xyzzType.getCurve(), b);
                } else if (auto jacobianType =
                               dyn_cast<JacobianType>(outputType)) {
                  sum = jacobianAdd(p1Coords, p2Coords, jacobianType.getCurve(),
                                    b);
                } else {
                  assert(false && "Unsupported point types for addition");
                }
                b.create<scf::YieldOp>(loc, sum);
              });
          b.create<scf::YieldOp>(output.getResults());
        });
    rewriter.replaceOpWithMultiple(op, {output.getResults()});
    return success();
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

    Type outputType = op.getOutput().getType();
    ValueRange coords = adaptor.getInput();
    SmallVector<Value> doubled;

    if (auto xyzzType = dyn_cast<XYZZType>(outputType)) {
      doubled = xyzzDouble(coords, xyzzType.getCurve(), b);
    } else if (auto jacobianType = dyn_cast<JacobianType>(outputType)) {
      doubled = jacobianDouble(coords, jacobianType.getCurve(), b);
    } else {
      assert(false && "Unsupported point type for doubling");
    }

    rewriter.replaceOpWithMultiple(op, {doubled});
    return success();
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

    ValueRange coords = adaptor.getInput();

    auto negatedY = b.create<field::NegateOp>(coords[1]);
    SmallVector<Value> outputCoords(coords);
    outputCoords[1] = negatedY;

    rewriter.replaceOpWithMultiple(op, {outputCoords});
    return success();
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

    Value negP2 = b.create<elliptic_curve::NegateOp>(op.getRhs());
    Value result = b.create<elliptic_curve::AddOp>(op.getOutput().getType(),
                                                   op.getLhs(), negP2);

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Currently implements Double-and-Add algorithm
// TODO(ashjeong): implement GLV
struct ConvertScalarMul : public OpConversionPattern<ScalarMulOp> {
  explicit ConvertScalarMul(MLIRContext *context)
      : OpConversionPattern<ScalarMulOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ScalarMulOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value point = op.getPoint();
    Value scalarPF = op.getScalar();

    Type pointType = op.getPoint().getType();
    Type outputType = op.getOutput().getType();

    auto scalarFieldType = cast<field::PrimeFieldType>(scalarPF.getType());
    unsigned scalarBitWidth =
        scalarFieldType.getModulus().getValue().getBitWidth();
    auto scalarIntType =
        IntegerType::get(b.getContext(), scalarBitWidth, IntegerType::Signless);
    auto scalarReduced = b.create<field::FromMontOp>(scalarPF);
    auto scalarInt = b.create<field::ExtractOp>(scalarIntType, scalarReduced);

    Type baseFieldType =
        getCurveFromPointLike(op.getPoint().getType()).getBaseField();
    auto zeroBF = b.create<field::ConstantOp>(baseFieldType, 0);
    auto oneBF = b.create<field::ToMontOp>(
        baseFieldType, b.create<field::ConstantOp>(baseFieldType, 1));

    Value zeroPoint =
        isa<XYZZType>(outputType)
            ? b.create<elliptic_curve::PointOp>(
                  outputType, ValueRange{oneBF, oneBF, zeroBF, zeroBF})
            : b.create<elliptic_curve::PointOp>(
                  outputType, ValueRange{oneBF, oneBF, zeroBF});

    Value initialPoint =
        isa<AffineType>(pointType)
            ? b.create<elliptic_curve::ConvertPointTypeOp>(outputType, point)
            : point;

    auto whileOp = b.create<scf::WhileOp>(
        /*resultTypes=*/TypeRange{scalarIntType, outputType, outputType},
        /*operands=*/ValueRange{scalarInt, initialPoint, zeroPoint},
        /*beforeBuilder=*/
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          auto arithZero = b.create<arith::ConstantIntOp>(0, scalarIntType);
          // if `decreasingScalar` > 0, continue
          Value decreasingScalar = args[0];
          auto cmpGt = b.create<arith::CmpIOp>(arith::CmpIPredicate::ugt,
                                               decreasingScalar, arithZero);
          b.create<scf::ConditionOp>(cmpGt, args);
        },
        /*afterBuilder=*/
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          auto arithOne = b.create<arith::ConstantIntOp>(1, scalarIntType);
          Value decreasingScalar = args[0];
          Value multiplyingPoint = args[1];
          Value result = args[2];

          // if `decreasingScalar` % 1 == 1...
          auto bitAdd = b.create<arith::AndIOp>(decreasingScalar, arithOne);
          auto cmpEq = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, bitAdd,
                                               arithOne);
          auto ifOp = b.create<scf::IfOp>(
              cmpEq,
              // ...then add `multiplyingPoint` to `result`
              /*thenBuilder=*/
              [&](OpBuilder &builder, Location loc) {
                Value innerResult = builder.create<elliptic_curve::AddOp>(
                    loc, outputType, result, multiplyingPoint);
                builder.create<scf::YieldOp>(loc, innerResult);
              },
              /*elseBuilder=*/
              [&](OpBuilder &builder, Location loc) {
                builder.create<scf::YieldOp>(loc, result);
              });
          // double `multiplyingPoint`
          Value doubledPoint =
              b.create<elliptic_curve::DoubleOp>(outputType, multiplyingPoint);
          // right shift `decreasingScalar` by 1
          decreasingScalar =
              b.create<arith::ShRUIOp>(decreasingScalar, arithOne);

          b.create<scf::YieldOp>(
              ValueRange({decreasingScalar, doubledPoint, ifOp.getResult(0)}));
        });
    rewriter.replaceOp(op, whileOp.getResult(2));
    return success();
  }
};

struct ConvertMSM : public OpConversionPattern<MSMOp> {
  explicit ConvertMSM(mlir::MLIRContext *context)
      : OpConversionPattern<MSMOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MSMOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // point set
    // |(x1, y1, z1), (x2, y2, z2)|
    Value pointSet = op.getPoints();
    // tensor of PF, e.g.:
    // | s1 , s2 |
    Value scalars = op.getScalars();
    RankedTensorType pointSetType = cast<RankedTensorType>(pointSet.getType());
    unsigned numScalarMuls = pointSetType.getShape()[0];
    Type outputPointType = op.getOutput().getType();

    auto zero = b.create<arith::ConstantIndexOp>(0);
    auto one = b.create<arith::ConstantIndexOp>(1);
    auto count = b.create<arith::ConstantIndexOp>(numScalarMuls);

    auto scalar0 = b.create<tensor::ExtractOp>(scalars, ValueRange{zero});
    auto point0 = b.create<tensor::ExtractOp>(pointSet, ValueRange{zero});
    Value initial =
        b.create<elliptic_curve::ScalarMulOp>(outputPointType, scalar0, point0);

    auto forOp = b.create<scf::ForOp>(
        one, count, one,  // induction variable: 1 to numScalarMuls
        ValueRange{initial},
        [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
            ValueRange args) {
          ImplicitLocOpBuilder b(nestedLoc, nestedBuilder);
          Value acc = args.front();

          auto scalar = b.create<tensor::ExtractOp>(scalars, ValueRange{iv});
          auto point = b.create<tensor::ExtractOp>(pointSet, ValueRange{iv});

          auto mul = b.create<elliptic_curve::ScalarMulOp>(outputPointType,
                                                           scalar, point);

          auto sum = b.create<elliptic_curve::AddOp>(outputPointType, acc, mul);

          b.create<scf::YieldOp>(ValueRange{sum});
        });

    rewriter.replaceOp(op, forOp.getResult(0));
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
  patterns.add<
      // clang-format off
      ConvertAdd,
      ConvertConvertPointType,
      ConvertDouble,
      ConvertExtract,
      ConvertIsZero,
      ConvertMSM,
      ConvertNegate,
      ConvertPoint,
      ConvertScalarMul,
      ConvertSub,
      ConvertAny<bufferization::MaterializeInDestinationOp>,
      ConvertAny<bufferization::ToMemrefOp>,
      ConvertAny<bufferization::ToTensorOp>,
      ConvertAny<linalg::BroadcastOp>,
      ConvertAny<memref::LoadOp>,
      ConvertAny<memref::StoreOp>,
      ConvertAny<tensor::ExtractOp>,
      ConvertAny<tensor::FromElementsOp>
      // clang-format on
      >(typeConverter, context);
  target.addDynamicallyLegalOp<
      // clang-format off
      bufferization::MaterializeInDestinationOp,
      bufferization::ToMemrefOp,
      bufferization::ToTensorOp,
      linalg::BroadcastOp,
      memref::LoadOp,
      memref::StoreOp,
      tensor::ExtractOp,
      tensor::FromElementsOp
      // clang-format on
      >([&](auto op) { return typeConverter.isLegal(op); });

  addStructuralConversionPatterns(typeConverter, patterns, target);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace mlir::zkir::elliptic_curve
