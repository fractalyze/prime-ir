#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h"

#include <utility>

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
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

namespace mlir::zkir::elliptic_curve {

#define GEN_PASS_DEF_ELLIPTICCURVETOFIELD
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h.inc"

namespace {
SmallVector<Type> coordsTypeRange(Type type) {
  if (auto affineType = dyn_cast<AffineType>(type)) {
    return SmallVector<Type>(2, affineType.getCurve().getBaseField());
  } else if (auto jacobianType = dyn_cast<JacobianType>(type)) {
    return SmallVector<Type>(3, jacobianType.getCurve().getBaseField());
  } else if (auto xyzzType = dyn_cast<XYZZType>(type)) {
    return SmallVector<Type>(4, xyzzType.getCurve().getBaseField());
  } else {
    llvm_unreachable("Unsupported point-like type for coords type range");
    return SmallVector<Type>();
  }
}

Operation::result_range extractCoords(ImplicitLocOpBuilder &b, Value point) {
  return b
      .create<elliptic_curve::ExtractOp>(coordsTypeRange(point.getType()),
                                         point)
      .getOutput();
}
} // namespace

struct ConvertIsZero : public OpConversionPattern<IsZeroOp> {
  explicit ConvertIsZero(MLIRContext *context)
      : OpConversionPattern<IsZeroOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IsZeroOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Operation::result_range coords = extractCoords(b, op.getInput());
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

struct ConvertConvertPointType
    : public OpConversionPattern<ConvertPointTypeOp> {
  explicit ConvertConvertPointType(MLIRContext *context)
      : OpConversionPattern<ConvertPointTypeOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConvertPointTypeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Operation::result_range coords = extractCoords(b, op.getInput());
    Type inputType = op.getInput().getType();
    Type outputType = op.getType();
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
    auto outputPt = b.create<elliptic_curve::PointOp>(outputType, outputCoords);
    rewriter.replaceOp(op, outputPt);
    return success();
  }
};

///////////// POINT ARITHMETIC OPERATIONS //////////////

struct ConvertAdd : public OpConversionPattern<AddOp> {
  explicit ConvertAdd(MLIRContext *context)
      : OpConversionPattern<AddOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value p1 = op.getLhs();
    Value p2 = op.getRhs();
    Operation::result_range p1Coords = extractCoords(b, op.getLhs());
    Operation::result_range p2Coords = extractCoords(b, op.getRhs());
    Type p1Type = p1.getType();
    Type p2Type = p2.getType();
    Type outputType = op.getType();

    // check p1 == zero point
    Value p1IsZero = b.create<elliptic_curve::IsZeroOp>(p1);
    auto output = b.create<scf::IfOp>(
        p1IsZero,
        /*thenBuilder=*/
        [&](OpBuilder &builder, Location loc) {
          ImplicitLocOpBuilder b(loc, builder);
          Value retP2 = p2;
          if (isa<AffineType>(p2Type)) {
            retP2 =
                b.create<elliptic_curve::ConvertPointTypeOp>(outputType, p2);
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
                Value retP1 = p1;
                if (isa<AffineType>(p1Type)) {
                  retP1 = b.create<elliptic_curve::ConvertPointTypeOp>(
                      outputType, p1);
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
                  llvm_unreachable("Unsupported point types for addition");
                }
                Value outputPt =
                    b.create<elliptic_curve::PointOp>(op.getType(), sum);
                b.create<scf::YieldOp>(outputPt);
              });
          b.create<scf::YieldOp>(output.getResults());
        });
    rewriter.replaceOp(op, output.getResults());
    return success();
  }
};

struct ConvertDouble : public OpConversionPattern<DoubleOp> {
  explicit ConvertDouble(MLIRContext *context)
      : OpConversionPattern<DoubleOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DoubleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type outputType = op.getType();
    Operation::result_range coords = extractCoords(b, op.getInput());
    SmallVector<Value> doubled;

    if (auto xyzzType = dyn_cast<XYZZType>(outputType)) {
      doubled = xyzzDouble(coords, xyzzType.getCurve(), b);
    } else if (auto jacobianType = dyn_cast<JacobianType>(outputType)) {
      doubled = jacobianDouble(coords, jacobianType.getCurve(), b);
    } else {
      llvm_unreachable("Unsupported point type for doubling");
    }

    auto outputPt = b.create<elliptic_curve::PointOp>(outputType, doubled);
    rewriter.replaceOp(op, outputPt);
    return success();
  }
};

struct ConvertNegate : public OpConversionPattern<NegateOp> {
  explicit ConvertNegate(MLIRContext *context)
      : OpConversionPattern<NegateOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NegateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Operation::result_range coords = extractCoords(b, op.getInput());

    auto negatedY = b.create<field::NegateOp>(coords[1]);
    SmallVector<Value> outputCoords(coords);
    outputCoords[1] = negatedY;

    auto outputPt =
        b.create<elliptic_curve::PointOp>(op.getType(), outputCoords);
    rewriter.replaceOp(op, outputPt);
    return success();
  }
};

struct ConvertSub : public OpConversionPattern<SubOp> {
  explicit ConvertSub(MLIRContext *context)
      : OpConversionPattern<SubOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value negP2 = b.create<elliptic_curve::NegateOp>(op.getRhs());
    Value result =
        b.create<elliptic_curve::AddOp>(op.getType(), op.getLhs(), negP2);

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

  LogicalResult
  matchAndRewrite(ScalarMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value point = op.getPoint();
    Value scalarPF = op.getScalar();

    Type pointType = op.getPoint().getType();
    Type outputType = op.getType();

    auto scalarFieldType = cast<field::PrimeFieldType>(scalarPF.getType());
    auto scalarIntType = scalarFieldType.getStorageType();
    Value scalarReduced =
        scalarFieldType.isMontgomery()
            ? b.create<field::FromMontOp>(
                  field::getStandardFormType(scalarFieldType), scalarPF)
            : scalarPF;
    Value scalarInt =
        b.create<field::BitcastOp>(TypeRange{scalarIntType}, scalarReduced);

    Value zeroPoint = createZeroPoint(b, outputType);

    Value initialPoint =
        isa<AffineType>(pointType)
            ? b.create<elliptic_curve::ConvertPointTypeOp>(outputType, point)
            : point;

    auto arithOne = b.create<arith::ConstantIntOp>(scalarIntType, 1);
    auto arithZero = b.create<arith::ConstantIntOp>(scalarIntType, 0);
    auto result = zeroPoint;
    auto ifOp = b.create<scf::IfOp>(
        b.create<arith::CmpIOp>(arith::CmpIPredicate::ne,
                                b.create<arith::AndIOp>(scalarInt, arithOne),
                                arithZero),
        [&](OpBuilder &builder, Location loc) {
          ImplicitLocOpBuilder b(loc, builder);
          Value newResult =
              b.create<elliptic_curve::AddOp>(outputType, result, initialPoint);
          b.create<scf::YieldOp>(newResult);
        },
        [&](OpBuilder &builder, Location loc) {
          ImplicitLocOpBuilder b(loc, builder);
          b.create<scf::YieldOp>(result);
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

  LogicalResult
  matchAndRewrite(MSMOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value scalars = op.getScalars();
    Value points = op.getPoints();

    Type outputType = op.getType();

    PippengersGeneric pippengers(scalars, points, outputType, b,
                                 adaptor.getParallel(), adaptor.getDegree(),
                                 adaptor.getWindowBits());

    rewriter.replaceOp(op, pippengers.generate());
    return success();
  }
};

struct ConvertBucketAcc : public OpConversionPattern<BucketAccOp> {
  explicit ConvertBucketAcc(mlir::MLIRContext *context)
      : OpConversionPattern<BucketAccOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BucketAccOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value zero = b.create<arith::ConstantIndexOp>(0);
    Value one = b.create<arith::ConstantIndexOp>(1);

    Value points = op.getPoints();
    Value sortedPointIndices = op.getSortedPointIndices();
    Value sortedUniqueBucketIndices = op.getSortedUniqueBucketIndices();
    Value bucketOffsets = op.getBucketOffsets();
    TensorType bucketResultsType = op.getBucketResults().getType();
    Type outputType = bucketResultsType.getElementType();

    // Create buckets and initialize all buckets to zeroPoint
    MemRefType memrefBucketResultsType =
        MemRefType::get(bucketResultsType.getShape(), outputType);
    Value bucketResults = b.create<memref::AllocOp>(memrefBucketResultsType);
    Value zeroPoint = createZeroPoint(b, outputType);
    b.create<linalg::FillOp>(zeroPoint, bucketResults);

    // Compute bucket accumulation across all buckets
    Value nofBucketsToCompute = b.create<arith::ConstantIndexOp>(
        cast<TensorType>(sortedUniqueBucketIndices.getType()).getNumElements());
    b.create<scf::ParallelOp>(
        zero, nofBucketsToCompute, one,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          ImplicitLocOpBuilder b0(loc, builder);
          Value i = ivs[0];

          Value bucketOffsetStart =
              b0.create<tensor::ExtractOp>(bucketOffsets, i);
          Value indexPlusOne = b0.create<arith::AddIOp>(i, one);
          Value bucketOffsetEnd =
              b0.create<tensor::ExtractOp>(bucketOffsets, indexPlusOne);

          // TODO(ashjeong): Replace with linalg::ReduceOp once supported on the
          // EC level
          // Aggregate all points per bucket
          auto pointLoop = b0.create<scf::ForOp>(
              /*lowerBound=*/bucketOffsetStart, /*upperBound=*/bucketOffsetEnd,
              /*step=*/one, zeroPoint,
              [&](OpBuilder &builder, Location loc, Value j, ValueRange args) {
                ImplicitLocOpBuilder b1(loc, builder);
                Value bucketResult = args[0];

                Value pointIndex =
                    b1.create<tensor::ExtractOp>(sortedPointIndices, j);
                Value point = b1.create<tensor::ExtractOp>(points, pointIndex);
                bucketResult = b1.create<elliptic_curve::AddOp>(
                    outputType, bucketResult, point);
                b1.create<scf::YieldOp>(bucketResult);
              });

          Value bucketIndex =
              b0.create<tensor::ExtractOp>(sortedUniqueBucketIndices, i);
          b0.create<memref::StoreOp>(pointLoop.getResult(0), bucketResults,
                                     bucketIndex);
          b0.create<scf::ReduceOp>();
        });

    // Convert bucket results to tensor
    Value bucketResultsTensor =
        b.create<bufferization::ToTensorOp>(bucketResultsType, bucketResults,
                                            /*restrict=*/true);
    rewriter.replaceOp(op, bucketResultsTensor);
    return success();
  }
};

struct ConvertBucketReduce : public OpConversionPattern<BucketReduceOp> {
  explicit ConvertBucketReduce(mlir::MLIRContext *context)
      : OpConversionPattern<BucketReduceOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BucketReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value buckets = op.getBuckets();
    RankedTensorType bucketsType = cast<RankedTensorType>(buckets.getType());
    Type pointType = bucketsType.getElementType();
    Type standardScalarType = field::getStandardFormType(op.getScalarType());
    Type scalarIntType = op.getScalarType().getStorageType();

    // Create bucket weights vector
    int64_t numBucketsPerWindow = bucketsType.getShape()[1];
    RankedTensorType arithWeightsType =
        bucketsType.clone({numBucketsPerWindow}, scalarIntType);
    SmallVector<Value> weightDims =
        numBucketsPerWindow == ShapedType::kDynamic
            ? SmallVector<Value>{b.create<tensor::DimOp>(buckets, 1)}
            : SmallVector<Value>{};
    Value arithWeights = b.create<tensor::GenerateOp>(
        arithWeightsType, weightDims,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          ImplicitLocOpBuilder b0(loc, builder);
          Value arithScalar =
              b0.create<arith::IndexCastOp>(scalarIntType, args[0]);
          b0.create<tensor::YieldOp>(arithScalar);
        });
    RankedTensorType weightsType = arithWeightsType.clone(standardScalarType);
    Value fieldWeights = b.create<field::BitcastOp>(weightsType, arithWeights);

    // Create output windows tensor
    int64_t numWindows = bucketsType.getShape()[0];
    RankedTensorType windowsType = bucketsType.clone({numWindows}, pointType);
    SmallVector<Value> windowDims =
        numWindows == ShapedType::kDynamic
            ? SmallVector<Value>{b.create<tensor::DimOp>(buckets, 0)}
            : SmallVector<Value>{};
    Value zeroPoint = createZeroPoint(b, pointType);
    Value windows = b.create<tensor::GenerateOp>(
        windowsType, windowDims,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          builder.create<tensor::YieldOp>(loc, zeroPoint);
        });

    // Calculate windows
    auto res = b.create<linalg::MatvecOp>(ValueRange{buckets, fieldWeights},
                                          ValueRange{windows});

    rewriter.replaceOp(op, res.getResult(0));
    return success();
  }
};

struct ConvertWindowReduce : public OpConversionPattern<WindowReduceOp> {
  explicit ConvertWindowReduce(mlir::MLIRContext *context)
      : OpConversionPattern<WindowReduceOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WindowReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value windows = op.getWindows();
    Type pointType = cast<RankedTensorType>(windows.getType()).getElementType();
    Type standardScalarType = getStandardFormType(op.getScalarType());
    Type scalarIntType = op.getScalarType().getStorageType();

    // Create output 0D tensor
    Value zeroPoint = createZeroPoint(b, pointType);
    Value outputTensor = b.create<tensor::FromElementsOp>(
        RankedTensorType::get({}, pointType), zeroPoint);

    // Calculate weighted windows reduction
    SmallVector<int64_t, 1> reductionDims = {0};
    Value c =
        b.create<arith::ConstantIntOp>(scalarIntType, op.getBitsPerWindow());
    Value base = b.create<field::ConstantOp>(standardScalarType, 2);
    // TODO(ashjeong): Try benchmarking against creating a separate weights
    // tensor & dot product. We want to test whether calculating  2ᶜⁱ in
    // parallel and reducing is faster than two loops of calculating 2ᶜⁱ
    // iteratively then reducing.
    auto total = b.create<linalg::ReduceOp>(
        windows, outputTensor, reductionDims,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          ImplicitLocOpBuilder b0(loc, builder);
          // Current point (args[0]) and accumulator (args[1]).
          Value i = b0.create<linalg::IndexOp>(0);
          Value arithI = b0.create<arith::IndexCastOp>(scalarIntType, i);
          Value exp = b0.create<arith::MulIOp>(c, arithI);
          Value weight = b0.create<field::PowUIOp>(base, exp);
          Value weightedValue = b0.create<elliptic_curve::ScalarMulOp>(
              pointType, weight, args[0]);

          Value sum = b0.create<elliptic_curve::AddOp>(pointType, args[1],
                                                       weightedValue);
          b0.create<linalg::YieldOp>(sum);
        });

    auto result = b.create<tensor::ExtractOp>(total.getResult(0));
    rewriter.replaceOp(op, result);
    return success();
  }
};

namespace rewrites {
// In an inner namespace to avoid conflicts with canonicalization patterns
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.cpp.inc"
} // namespace rewrites

struct EllipticCurveToField
    : impl::EllipticCurveToFieldBase<EllipticCurveToField> {
  using EllipticCurveToFieldBase::EllipticCurveToFieldBase;

  void runOnOperation() override;
};

void EllipticCurveToField::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addIllegalOp<
      // clang-format off
      AddOp,
      BucketAccOp,
      ConvertPointTypeOp,
      DoubleOp,
      IsZeroOp,
      linalg::MatvecOp,
      MSMOp,
      ScalarMulOp,
      SubOp,
      WindowReduceOp
      // clang-format on
      >();

  target.addLegalDialect<
      // clang-format off
      arith::ArithDialect,
      bufferization::BufferizationDialect,
      field::FieldDialect,
      linalg::LinalgDialect,
      memref::MemRefDialect,
      scf::SCFDialect,
      tensor::TensorDialect
      // clang-format on
      >();

  target.addLegalOp<
      // clang-format off
      elliptic_curve::ExtractOp,
      elliptic_curve::PointOp
      // clang-format on
      >();

  RewritePatternSet patterns(context);
  linalg::populateLinalgNamedOpsGeneralizationPatterns(patterns);
  rewrites::populateWithGenerated(patterns);
  patterns.add<
      // clang-format off
      ConvertAdd,
      ConvertBucketAcc,
      ConvertBucketReduce,
      ConvertConvertPointType,
      ConvertDouble,
      ConvertIsZero,
      ConvertMSM,
      ConvertNegate,
      ConvertScalarMul,
      ConvertSub,
      ConvertWindowReduce
      // clang-format on
      >(context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace mlir::zkir::elliptic_curve
