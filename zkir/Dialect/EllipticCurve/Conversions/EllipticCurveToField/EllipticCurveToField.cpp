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

struct ConvertIsZero : public OpConversionPattern<IsZeroOp> {
  explicit ConvertIsZero(MLIRContext *context)
      : OpConversionPattern<IsZeroOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IsZeroOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    ValueRange coords = adaptor.getInput();
    field::PrimeFieldType baseField =
        cast<field::PrimeFieldType>(coords[0].getType());
    Value zeroPF = b.create<field::ConstantOp>(baseField, 0);

    Value cmp;
    if (isa<AffineType>(op.getInput().getType())) {
      Value xIsZero =
          b.create<field::CmpOp>(arith::CmpIPredicate::eq, coords[0], zeroPF);
      Value yIsZero =
          b.create<field::CmpOp>(arith::CmpIPredicate::eq, coords[1], zeroPF);
      cmp = b.create<arith::AndIOp>(xIsZero, yIsZero);
    } else {
      cmp = b.create<field::CmpOp>(arith::CmpIPredicate::eq, coords[2], zeroPF);
    }
    rewriter.replaceOp(op, cmp);
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
    auto baseFieldType = cast<field::PrimeFieldType>(coords[0].getType());
    Type inputType = op.getInput().getType();
    Type outputType = op.getOutput().getType();

    SmallVector<Value> outputCoords;

    if (isa<AffineType>(inputType)) {
      auto onePF = b.create<field::ConstantOp>(baseFieldType, 1);
      // affine to jacobian
      // (x, y) -> (x, y, 1)
      outputCoords = {/* x */ coords[0], /* y */ coords[1], onePF};
      if (isa<XYZZType>(outputType)) {
        outputCoords.push_back(onePF);
        // affine to xyzz
        // (x, y) -> (x, y, 1, 1)
      }
    } else if (isa<JacobianType>(inputType)) {
      auto zz = b.create<field::SquareOp>(/* z */ coords[2]);
      auto zzz = b.create<field::MulOp>(zz, /* z */ coords[2]);

      if (isa<AffineType>(outputType)) {
        // jacobian to affine
        // (x, y, z) -> (x/z², y/z³)
        auto zero = b.create<field::ConstantOp>(baseFieldType, 0);
        auto cmpEq = b.create<field::CmpOp>(arith::CmpIPredicate::eq,
                                            /* z */ coords[2], zero);
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
              auto newX =
                  builder.create<field::MulOp>(loc, /* x */ coords[0], zzInv);
              auto newY =
                  builder.create<field::MulOp>(loc, /* y */ coords[1], zzzInv);
              builder.create<scf::YieldOp>(loc, ValueRange{newX, newY});
            });

        outputCoords = {output.getResult(0), output.getResult(1)};
      } else {
        // jacobian to xyzz
        // (x, y, z) -> (x, y, z², z³)
        outputCoords = {/* x */ coords[0], /* y */ coords[1], zz, zzz};
      }
    } else {
      if (isa<AffineType>(outputType)) {
        // xyzz to affine
        // (x, y, z², z³) -> (x/z², y/z³)
        auto zero = b.create<field::ConstantOp>(baseFieldType, 0);
        auto cmpEq = b.create<field::CmpOp>(arith::CmpIPredicate::eq,
                                            /* zz */ coords[2], zero);
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
              auto zzInv =
                  builder.create<field::InverseOp>(loc, /* zz */ coords[2]);
              auto zzzInv =
                  builder.create<field::InverseOp>(loc, /* zzz */ coords[3]);
              auto newX =
                  builder.create<field::MulOp>(loc, /* x */ coords[0], zzInv);
              auto newY =
                  builder.create<field::MulOp>(loc, /* y */ coords[1], zzzInv);
              builder.create<scf::YieldOp>(loc, ValueRange{newX, newY});
            });
        outputCoords = {ifOp.getResult(0), ifOp.getResult(1)};
      } else {
        // xyzz to jacobian
        // (x, y, z², z³) -> (x, y, z)
        outputCoords = {/* x */ coords[0], /* y */ coords[1]};

        auto zero = b.create<field::ConstantOp>(baseFieldType, 0);
        auto cmpEq = b.create<field::CmpOp>(arith::CmpIPredicate::eq,
                                            /* zz */ coords[2], zero);
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
              auto z =
                  builder.create<field::MulOp>(loc, /* zzz */ coords[3], zzInv);
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
    Value p1isZeroCmp = b.create<elliptic_curve::IsZeroOp>(p1);
    auto p1IsZeroOp = b.create<scf::IfOp>(
        p1isZeroCmp,
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
          Value p2isZeroCmp = b.create<elliptic_curve::IsZeroOp>(p2);
          auto p2IsZeroOp = b.create<scf::IfOp>(
              p2isZeroCmp,
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
          b.create<scf::YieldOp>(p2IsZeroOp.getResults());
        });
    rewriter.replaceOpWithMultiple(op, {p1IsZeroOp.getResults()});
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
    SmallVector<Value> sum;

    if (auto xyzzType = dyn_cast<XYZZType>(outputType)) {
      sum = xyzzDouble(coords, xyzzType.getCurve(), b);
    } else if (auto jacobianType = dyn_cast<JacobianType>(outputType)) {
      sum = jacobianDouble(coords, jacobianType.getCurve(), b);
    } else {
      assert(false && "Unsupported point type for doubling");
    }

    rewriter.replaceOpWithMultiple(op, {sum});
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

    auto baseFieldType = cast<field::PrimeFieldType>(scalarPF.getType());
    unsigned outputBitWidth =
        baseFieldType.getModulus().getValue().getBitWidth();
    auto signlessIntType =
        IntegerType::get(b.getContext(), outputBitWidth, IntegerType::Signless);
    auto scalar = b.create<field::ExtractOp>(signlessIntType, scalarPF);

    auto zeroPF = b.create<field::ConstantOp>(baseFieldType, 0);
    auto onePF = b.create<field::ConstantOp>(baseFieldType, 1);
    Value zeroPoint =
        isa<XYZZType>(outputType)
            ? b.create<elliptic_curve::PointOp>(
                  outputType, ValueRange{onePF, onePF, zeroPF, zeroPF})
            : b.create<elliptic_curve::PointOp>(
                  outputType, ValueRange{onePF, onePF, zeroPF});

    Value intialPoint =
        isa<AffineType>(pointType)
            ? b.create<elliptic_curve::ConvertPointTypeOp>(outputType, point)
            : point;

    auto whileOp = b.create<scf::WhileOp>(
        /*resultTypes=*/TypeRange{signlessIntType, outputType, outputType},
        /*operands=*/ValueRange{scalar, intialPoint, zeroPoint},
        /*beforeBuilder=*/
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          auto arithZero = b.create<arith::ConstantIntOp>(0, signlessIntType);
          // if `decreasingScalar` > 0, continue
          Value decreasingScalar = args[0];
          auto cmpGt = b.create<arith::CmpIOp>(arith::CmpIPredicate::ugt,
                                               decreasingScalar, arithZero);
          b.create<scf::ConditionOp>(cmpGt, args);
        },
        /*afterBuilder=*/
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          auto arithOne = b.create<arith::ConstantIntOp>(1, signlessIntType);
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

    auto idx = b.create<arith::ConstantIndexOp>(0);
    auto scalar = b.create<tensor::ExtractOp>(scalars, ValueRange{idx});
    auto point = b.create<tensor::ExtractOp>(pointSet, ValueRange{idx});
    Value accumulator =
        b.create<elliptic_curve::ScalarMulOp>(outputPointType, scalar, point);
    for (size_t i = 1; i < numScalarMuls; ++i) {
      idx = b.create<arith::ConstantIndexOp>(i);
      scalar = b.create<tensor::ExtractOp>(scalars, ValueRange{idx});
      point = b.create<tensor::ExtractOp>(pointSet, ValueRange{idx});
      auto adder =
          b.create<elliptic_curve::ScalarMulOp>(outputPointType, scalar, point);
      accumulator =
          b.create<elliptic_curve::AddOp>(outputPointType, accumulator, adder);
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
      .add<ConvertPoint, ConvertIsZero, ConvertExtract, ConvertConvertPointType,
           ConvertAdd, ConvertDouble, ConvertNegate, ConvertSub,
           ConvertScalarMul, ConvertMSM, ConvertAny<tensor::FromElementsOp>,
           ConvertAny<tensor::ExtractOp>>(typeConverter, context);
  target.addDynamicallyLegalOp<tensor::FromElementsOp, tensor::ExtractOp>(
      [&](auto op) { return typeConverter.isLegal(op); });

  addStructuralConversionPatterns(typeConverter, patterns, target);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace mlir::zkir::elliptic_curve
