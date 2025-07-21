#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h"

#include <utility>

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/MSM/Pippengers/Generic.h"
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
#include "zkir/Utils/ShapedTypeConverter.h"

namespace mlir::zkir::elliptic_curve {

#define GEN_PASS_DEF_ELLIPTICCURVETOFIELD
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h.inc"

//////////////// TYPE CONVERSION ////////////////

class EllipticCurveToFieldTypeConverter : public ShapedTypeConverter {
 public:
  explicit EllipticCurveToFieldTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion(
        [](AffineType type, SmallVectorImpl<Type> &converted) -> LogicalResult {
          return convertPointType(type, converted);
        });
    addConversion([](JacobianType type,
                     SmallVectorImpl<Type> &converted) -> LogicalResult {
      return convertPointType(type, converted);
    });
    addConversion(
        [](XYZZType type, SmallVectorImpl<Type> &converted) -> LogicalResult {
          return convertPointType(type, converted);
        });
    addConversion([](ShapedType type) -> Type {
      Type elementType = type.getElementType();
      Type baseFieldType;
      size_t numCoords = 0;
      if (auto pointType = dyn_cast<AffineType>(elementType)) {
        baseFieldType = pointType.getCurve().getBaseField();
        numCoords = 2;
      } else if (auto pointType = dyn_cast<JacobianType>(elementType)) {
        baseFieldType = pointType.getCurve().getBaseField();
        numCoords = 3;
      } else if (auto pointType = dyn_cast<XYZZType>(elementType)) {
        baseFieldType = pointType.getCurve().getBaseField();
        numCoords = 4;
      } else {
        return type;
      }
      SmallVector<int64_t> newShape(type.getShape());
      newShape.push_back(numCoords);
      return convertShapedType(type, newShape, baseFieldType);
    });
  }

 private:
  template <typename T>
  static LogicalResult convertPointType(T type,
                                        SmallVectorImpl<Type> &converted) {
    Type baseFieldType = type.getCurve().getBaseField();
    size_t numCoords = 0;
    if constexpr (std::is_same_v<T, AffineType>) {
      numCoords = 2;
    } else if constexpr (std::is_same_v<T, JacobianType>) {
      numCoords = 3;
    } else if constexpr (std::is_same_v<T, XYZZType>) {
      numCoords = 4;
    } else {
      return failure();
    }
    for (size_t i = 0; i < numCoords; ++i) {
      converted.push_back(baseFieldType);
    }
    return success();
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
    Type baseFieldType =
        getCurveFromPointLike(op.getInput().getType()).getBaseField();
    Value zeroBF = b.create<field::ConstantOp>(baseFieldType, 0);

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
    // so: [[x,y,z]] to [[x],[y],[z]] (i.e. for jacobian). A naive copy of the
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
        baseFieldType, b.create<field::ConstantOp>(
                           field::getStandardFormType(baseFieldType), 1));

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
              auto zzInv = b.create<field::InverseOp>(zz);
              auto zzzzInv = b.create<field::SquareOp>(loc, zzInv);
              auto zzzInv =
                  builder.create<field::MulOp>(loc, zzzzInv, /*z=*/coords[2]);
              auto newX =
                  builder.create<field::MulOp>(loc, /*x=*/coords[0], zzInv);
              auto newY =
                  builder.create<field::MulOp>(loc, /*y=*/coords[1], zzzInv);
              builder.create<scf::YieldOp>(loc, ValueRange{newX, newY});
            });

        outputCoords = output.getResults();
      } else {
        auto zzz = b.create<field::MulOp>(zz, /*z=*/coords[2]);
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
              auto zzzInv =
                  builder.create<field::InverseOp>(loc, /*zzz=*/coords[3]);
              auto zInv =
                  builder.create<field::MulOp>(loc, zzzInv, /*zz=*/coords[2]);
              auto zzInv = builder.create<field::SquareOp>(loc, zInv);
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
    Value scalarReduced =
        scalarFieldType.isMontgomery()
            ? b.create<field::FromMontOp>(
                  field::getStandardFormType(scalarFieldType), scalarPF)
            : scalarPF;
    Value scalarInt =
        b.create<field::ExtractOp>(TypeRange{scalarIntType}, scalarReduced)
            .getResult(0);

    Type baseFieldType =
        getCurveFromPointLike(op.getPoint().getType()).getBaseField();
    auto zeroBF = b.create<field::ConstantOp>(baseFieldType, 0);
    Value oneBF = field::isMontgomery(baseFieldType)
                      ? b.create<field::ToMontOp>(
                             baseFieldType,
                             b.create<field::ConstantOp>(
                                 field::getStandardFormType(baseFieldType), 1))
                            .getResult()
                      : b.create<field::ConstantOp>(baseFieldType, 1);

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

    auto arithOne = b.create<arith::ConstantIntOp>(1, scalarIntType);
    auto arithZero = b.create<arith::ConstantIntOp>(0, scalarIntType);
    auto result = zeroPoint;
    auto ifOp = b.create<scf::IfOp>(
        b.create<arith::CmpIOp>(arith::CmpIPredicate::ne,
                                b.create<arith::AndIOp>(scalarInt, arithOne),
                                arithZero),
        [&](OpBuilder &builder, Location loc) {
          ImplicitLocOpBuilder b(loc, builder);
          auto newResult =
              b.create<elliptic_curve::AddOp>(outputType, result, initialPoint);
          b.create<scf::YieldOp>(ValueRange{newResult});
        },
        [&](OpBuilder &builder, Location loc) {
          ImplicitLocOpBuilder b(loc, builder);
          b.create<scf::YieldOp>(ValueRange{result});
        });
    result = ifOp.getResult(0);
    scalarInt = b.create<arith::ShRUIOp>(scalarInt, arithOne);

    auto whileOp = b.create<scf::WhileOp>(
        /*resultTypes=*/TypeRange{scalarIntType, outputType, outputType},
        /*operands=*/ValueRange{scalarInt, initialPoint, result},
        /*beforeBuilder=*/
        [&](OpBuilder &beforeBuilder, Location beforeLoc, ValueRange args) {
          ImplicitLocOpBuilder b(beforeLoc, beforeBuilder);
          // if `decreasingScalar` > 0, continue
          Value decreasingScalar = args[0];
          auto cmpGt = b.create<arith::CmpIOp>(arith::CmpIPredicate::ugt,
                                               decreasingScalar, arithZero);
          b.create<scf::ConditionOp>(cmpGt, args);
        },
        /*afterBuilder=*/
        [&](OpBuilder &afterBuilder, Location afterLoc, ValueRange args) {
          ImplicitLocOpBuilder b(afterLoc, afterBuilder);
          Value decreasingScalar = args[0];
          Value multiplyingPoint = args[1];
          Value result = args[2];

          // double `multiplyingPoint`
          Value doubledPoint =
              b.create<elliptic_curve::DoubleOp>(outputType, multiplyingPoint);
          // if `decreasingScalar` % 1 == 1...
          auto bitAdd = b.create<arith::AndIOp>(decreasingScalar, arithOne);
          auto cmpEq = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, bitAdd,
                                               arithOne);
          auto ifOp = b.create<scf::IfOp>(
              cmpEq,
              // ...then add `doubledPoint` to `result`
              /*thenBuilder=*/
              [&](OpBuilder &builder, Location loc) {
                Value innerResult = builder.create<elliptic_curve::AddOp>(
                    loc, outputType, result, doubledPoint);
                builder.create<scf::YieldOp>(loc, innerResult);
              },
              /*elseBuilder=*/
              [&](OpBuilder &builder, Location loc) {
                builder.create<scf::YieldOp>(loc, result);
              });
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

// Currently implements Pippenger's
struct ConvertMSM : public OpConversionPattern<MSMOp> {
  explicit ConvertMSM(mlir::MLIRContext *context)
      : OpConversionPattern<MSMOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MSMOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value scalars = op.getScalars();
    Value points = op.getPoints();

    Type baseFieldType =
        cast<RankedTensorType>(adaptor.getPoints()[0].getType())
            .getElementType();

    Type outputType = op.getOutput().getType();

    PippengersGeneric pippengers(scalars, points, baseFieldType, outputType, b,
                                 adaptor.getParallel(), adaptor.getDegree(),
                                 adaptor.getWindowBits());

    rewriter.replaceOp(op, pippengers.generate());
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
      ConvertAny<bufferization::AllocTensorOp>,
      ConvertAny<bufferization::MaterializeInDestinationOp>,
      ConvertAny<bufferization::ToMemrefOp>,
      ConvertAny<bufferization::ToTensorOp>,
      ConvertAny<memref::AllocOp>,
      ConvertAny<memref::AllocaOp>,
      ConvertAny<memref::CastOp>,
      ConvertAny<memref::DimOp>,
      ConvertAny<memref::LoadOp>,
      ConvertAny<memref::StoreOp>,
      ConvertAny<memref::SubViewOp>,
      ConvertAny<tensor::DimOp>,
      ConvertAny<tensor::ExtractOp>,
      ConvertAny<tensor::ExtractSliceOp>,
      ConvertAny<tensor::FromElementsOp>
      // clang-format on
      >(typeConverter, context);
  target.addDynamicallyLegalOp<
      // clang-format off
      bufferization::AllocTensorOp,
      bufferization::MaterializeInDestinationOp,
      bufferization::ToMemrefOp,
      bufferization::ToTensorOp,
      memref::AllocOp,
      memref::AllocaOp,
      memref::CastOp,
      memref::DimOp,
      memref::LoadOp,
      memref::StoreOp,
      memref::SubViewOp,
      tensor::DimOp,
      tensor::ExtractOp,
      tensor::ExtractSliceOp,
      tensor::FromElementsOp
      // clang-format on
      >([&](auto op) { return typeConverter.isLegal(op); });

  addStructuralConversionPatterns(typeConverter, patterns, target);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace mlir::zkir::elliptic_curve
