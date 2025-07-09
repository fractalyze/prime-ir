#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/MSM/Pippengers/SignedBucketIndex.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"

namespace mlir::zkir::elliptic_curve {

void PippengersSignedBucketIndex::populateBuckets(Value cutScalar,
                                                  Value buckets, Value point,
                                                  ImplicitLocOpBuilder &b) {
  Type windowBitsIntType =
      IntegerType::get(b_.getContext(), bitsPerWindow_ + 1);
  auto zeroInt = b.create<arith::ConstantIntOp>(0, windowBitsIntType);
  auto isPositive =
      b.create<arith::CmpIOp>(arith::CmpIPredicate::sgt, cutScalar, zeroInt);
  b.create<scf::IfOp>(
      isPositive,
      [&](OpBuilder &builder, Location loc) {
        // Positive cutScalar: add point to buckets[cutScalar - 1]
        ImplicitLocOpBuilder b0(loc, builder);
        Value scalarMinusOne = b0.create<arith::SubIOp>(
            b0.create<arith::IndexCastUIOp>(b0.getIndexType(), cutScalar),
            one_);
        Value bucket = b0.create<memref::LoadOp>(buckets, scalarMinusOne);
        Value newBucket =
            b0.create<elliptic_curve::AddOp>(outputType_, bucket, point);
        b0.create<memref::StoreOp>(newBucket, buckets, scalarMinusOne);
        b0.create<scf::YieldOp>();
      },
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder b0(loc, builder);
        auto isNotZero = b0.create<arith::CmpIOp>(arith::CmpIPredicate::ne,
                                                  cutScalar, zeroInt);
        b0.create<scf::IfOp>(isNotZero, [&](OpBuilder &builder, Location loc) {
          // Negative cutScalar: subtract point from buckets[-cutScalar - 1]
          ImplicitLocOpBuilder b1(loc, builder);
          Value negScalar = b1.create<arith::SubIOp>(zeroInt, cutScalar);
          Value negScalarMinusOne = b1.create<arith::SubIOp>(
              b1.create<arith::IndexCastUIOp>(b1.getIndexType(), negScalar),
              one_);
          Value bucket = b1.create<memref::LoadOp>(buckets, negScalarMinusOne);
          Value newBucket =
              b1.create<elliptic_curve::SubOp>(outputType_, bucket, point);
          b1.create<memref::StoreOp>(newBucket, buckets, negScalarMinusOne);
          b1.create<scf::YieldOp>();
        });
        b0.create<scf::YieldOp>();
      });
  b.create<scf::YieldOp>();
}

void PippengersSignedBucketIndex::runSingleWindow(Value j, Value carries,
                                                  Value buckets,
                                                  Value numBuckets,
                                                  Value isLastWindow) {
  uint64_t radix = uint64_t{1} << bitsPerWindow_;
  Type scalarType = IntegerType::get(
      b_.getContext(), scalarFieldType_.getModulus().getValue().getBitWidth());
  Value bitOffset = b_.create<arith::IndexCastOp>(
      scalarType, b_.create<arith::MulIOp>(
                      j, b_.create<arith::ConstantIndexOp>(bitsPerWindow_)));
  Value mask =
      b_.create<arith::ConstantIntOp>((1 << bitsPerWindow_) - 1, scalarType);
  Type windowBitsIntType =
      IntegerType::get(b_.getContext(), bitsPerWindow_ + 1);
  Value bitsPerWindow =
      b_.create<arith::ConstantIntOp>(bitsPerWindow_, windowBitsIntType);

  // Loop through all scalar-point pairs.
  b_.create<scf::ForOp>(
      zero_, numScalarMuls_, one_, std::nullopt,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value i,
          ValueRange args) {
        ImplicitLocOpBuilder b0(nestedLoc, nestedBuilder);

        Value scalar = b0.create<field::ExtractOp>(scalarType,
                                                   b0.create<tensor::ExtractOp>(
                                                       scalars_, ValueRange{i}))
                           .getResult(0);
        Value upperBitsScalar = b0.create<arith::ShRUIOp>(scalar, bitOffset);
        Value scalarPerWindow = b0.create<arith::TruncIOp>(
            windowBitsIntType, b0.create<arith::AndIOp>(upperBitsScalar, mask));
        Value prevCarry = b0.create<arith::IndexCastOp>(
            windowBitsIntType, b0.create<memref::LoadOp>(carries, i));

        // add the carry from the previous iteration to the scalar per window
        Value newScalar = b0.create<arith::AddIOp>(prevCarry, scalarPerWindow);

        // newCarry = (newScalar + radix / 2) >> bitsPerWindow;
        Value newCarry = b0.create<arith::ShRUIOp>(
            b0.create<arith::AddIOp>(
                newScalar,
                b0.create<arith::ConstantIntOp>(radix / 2, windowBitsIntType)),
            bitsPerWindow);
        // store the new carry
        b0.create<memref::StoreOp>(
            b0.create<arith::IndexCastOp>(b0.getIndexType(), newCarry), carries,
            i);

        // cutScalar = newScalar - (relevant bits of newCarry)
        Value cutScalar = b0.create<arith::SubIOp>(
            newScalar, b0.create<arith::ShLIOp>(newCarry, bitsPerWindow));

        // Add or subtract the point from the corresponding bucket.
        Value point = b0.create<tensor::ExtractOp>(points_, ValueRange{i});
        populateBuckets(cutScalar, buckets, point, b0);
      });

  bucketReduction(j, zeroPoint_, buckets, numBuckets, b_);
  return;
}

void PippengersSignedBucketIndex::bucketAccReduc() {
  auto carriesType = MemRefType::get({ShapedType::kDynamic}, b_.getIndexType());
  Value carries = b_.create<memref::AllocOp>(carriesType, numScalarMuls_);

  auto changedScalarsAndPoints = b_.create<scf::ForOp>(
      zero_, numScalarMuls_, one_, ValueRange{scalars_, points_},
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value i,
          ValueRange args) {
        ImplicitLocOpBuilder b0(nestedLoc, nestedBuilder);
        // initialize carries
        b0.create<memref::StoreOp>(zero_, carries, i);

        // https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/signed-bucket-index#yrrids-trick
        // This ensures the last window will not have a carry
        Value scalars = args[0];
        Value points = args[1];
        if (enableYrridsTrick_) {
          uint64_t scalarBitWidth =
              scalarFieldType_.getModulus().getValue().getBitWidth();
          Type scalarType = IntegerType::get(b_.getContext(), scalarBitWidth);
          Value scalarField =
              b0.create<tensor::ExtractOp>(scalars, ValueRange{i});

          Value msbMask = b0.create<arith::ConstantIntOp>(
              1 << (scalarBitWidth - 1), scalarType);
          Value msb = b0.create<arith::AndIOp>(
              b0.create<field::ExtractOp>(scalarType, scalarField).getResult(0),
              msbMask);
          Value hasMostSignificantBit = b0.create<arith::CmpIOp>(
              arith::CmpIPredicate::ugt, msb,
              b0.create<arith::ConstantIntOp>(0, scalarType));
          Value point = b0.create<tensor::ExtractOp>(points, ValueRange{i});

          auto trick = b0.create<scf::IfOp>(
              hasMostSignificantBit,
              [&](OpBuilder &builder, Location loc) {
                ImplicitLocOpBuilder b1(loc, builder);
                Value trueScalar = b1.create<field::NegateOp>(scalarField);
                Value truePoint = b1.create<elliptic_curve::NegateOp>(point);
                b1.create<scf::YieldOp>(ValueRange{trueScalar, truePoint});
              },
              [&](OpBuilder &builder, Location loc) {
                ImplicitLocOpBuilder b1(loc, builder);
                Value trueScalar = scalarField;
                Value truePoint = point;
                b1.create<scf::YieldOp>(ValueRange{trueScalar, truePoint});
              });
          scalars = b0.create<tensor::InsertOp>(trick.getResult(0), scalars,
                                                ValueRange{i});
          points = b0.create<tensor::InsertOp>(trick.getResult(1), points,
                                               ValueRange{i});
        }
        b0.create<scf::YieldOp>(ValueRange{scalars, points});
      });
  scalars_ = changedScalarsAndPoints.getResult(0);
  points_ = changedScalarsAndPoints.getResult(1);

  auto windowsForOp = b_.create<scf::ForOp>(zero_, numWindows_, one_);
  b_.setInsertionPointToStart(windowsForOp.getBody());
  Value j = windowsForOp.getInductionVar();
  // Determine the correct number of buckets. For the last window, we need 2^c
  // buckets.
  Value numWindowsMinusOne = b_.create<arith::SubIOp>(numWindows_, one_);
  Value isLastWindow =
      b_.create<arith::CmpIOp>(arith::CmpIPredicate::eq, j, numWindowsMinusOne);
  Value numBuckets =
      b_.create<arith::ConstantIndexOp>(1 << (bitsPerWindow_ - 1));

  // Allocate and initialize the buckets.
  auto bucketsType = MemRefType::get({ShapedType::kDynamic}, outputType_);
  Value buckets = b_.create<memref::AllocOp>(bucketsType, numBuckets);
  b_.create<scf::ForOp>(zero_, numBuckets, one_, std::nullopt,
                        [&](OpBuilder &nestedBuilder, Location nestedLoc,
                            Value i, ValueRange args) {
                          ImplicitLocOpBuilder b0(nestedLoc, nestedBuilder);
                          b0.create<memref::StoreOp>(zeroPoint_, buckets, i);
                          b0.create<scf::YieldOp>();
                        });

  runSingleWindow(j, carries, buckets, numBuckets, isLastWindow);

  b_.setInsertionPointAfter(windowsForOp);
}

}  // namespace mlir::zkir::elliptic_curve
