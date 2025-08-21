#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/MSM/Pippengers/Generic.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"

namespace mlir::zkir::elliptic_curve {

// We only process unit scalars once in the first window.
ValueRange PippengersGeneric::scalarIsOneBranch(Value point, Value windowOffset,
                                                Value windowSum,
                                                ImplicitLocOpBuilder &b) {
  auto windowOffsetIsZero =
      b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, windowOffset, zero_);
  auto windowOffsetIsZeroIfOp = b.create<scf::IfOp>(
      windowOffsetIsZero,
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder b0(loc, builder);
        if (isa<JacobianType>(point.getType()) && isa<XYZZType>(outputType_)) {
          point =
              b0.create<elliptic_curve::ConvertPointTypeOp>(outputType_, point);
        }
        Value addedPoint =
            b0.create<elliptic_curve::AddOp>(outputType_, windowSum, point);
        b0.create<scf::YieldOp>(addedPoint);
      },
      [&](OpBuilder &builder, Location loc) {
        builder.create<scf::YieldOp>(loc, windowSum);
      });
  return windowOffsetIsZeroIfOp.getResults();
}

// https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/pippengers-algorithm#id-1.-scalar-decomposition
Value PippengersGeneric::scalarDecomposition(Value scalar,
                                             Value windowOffsetIndex,
                                             ImplicitLocOpBuilder &b) {
  size_t scalarBitWidth =
      scalarFieldType_.getModulus().getValue().getBitWidth();
  auto scalarIntType = IntegerType::get(b.getContext(), scalarBitWidth);
  Value windowOffset =
      b.create<arith::IndexCastOp>(scalarIntType, windowOffsetIndex);

  // We right-shift by `windowOffset`, thus getting rid
  // of the lower bits.
  Value signlessScalar =
      b.create<field::ExtractOp>(TypeRange{scalarIntType}, scalar).getResult(0);
  auto upperBitsScalar = b.create<arith::ShRUIOp>(signlessScalar, windowOffset);

  auto windowBitIntType = IntegerType::get(b.getContext(), bitsPerWindow_);
  return b.create<arith::TruncIOp>(windowBitIntType, upperBitsScalar);
}

void PippengersGeneric::scalarIsNotOneBranch(Value scalar, Value point,
                                             Value buckets, Value windowOffset,
                                             ImplicitLocOpBuilder &b) {
  auto windowBitIntType = IntegerType::get(b.getContext(), bitsPerWindow_);

  Value zeroInt = b.create<arith::ConstantIntOp>(windowBitIntType, 0);
  Value oneInt = b.create<arith::ConstantIntOp>(windowBitIntType, 1);
  Value scalarForWindow = scalarDecomposition(scalar, windowOffset, b);

  // If the scalar is non-zero, we update the corresponding bucket. (Recall that
  // `buckets` doesn't have a zero bucket.)
  auto scalarPerWindowIsNotZero = b.create<arith::CmpIOp>(
      arith::CmpIPredicate::ne, scalarForWindow, zeroInt);
  b.create<scf::IfOp>(
      scalarPerWindowIsNotZero,
      /*thenBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder b0(loc, builder);
        auto scalarPerWindowMinusOne =
            b0.create<arith::SubIOp>(scalarForWindow, oneInt);
        Value adjustedIdx = b0.create<arith::IndexCastUIOp>(
            b0.getIndexType(), scalarPerWindowMinusOne);
        auto bucketAtAdjustedIdx =
            b0.create<memref::LoadOp>(outputType_, buckets, adjustedIdx);
        if (isa<JacobianType>(point.getType()) && isa<XYZZType>(outputType_)) {
          point =
              b0.create<elliptic_curve::ConvertPointTypeOp>(outputType_, point);
        }
        auto newPoint = b0.create<elliptic_curve::AddOp>(
            outputType_, bucketAtAdjustedIdx, point);
        b0.create<memref::StoreOp>(newPoint, buckets, adjustedIdx);
        b0.create<scf::YieldOp>();
      });
}

ValueRange PippengersGeneric::bucketSingleAcc(Value i, Value windowSum,
                                              Value buckets, Value windowOffset,
                                              ImplicitLocOpBuilder &b) {
  auto scalar = b.create<tensor::ExtractOp>(scalars_, ValueRange{i});
  auto point = b.create<tensor::ExtractOp>(points_, ValueRange{i});

  auto zeroSF = b.create<field::ConstantOp>(scalarFieldType_, 0);
  auto scalarIsZero =
      b.create<field::CmpOp>(arith::CmpIPredicate::eq, scalar, zeroSF);
  auto zeroScalarMulIfOp = b.create<scf::IfOp>(
      scalarIsZero,
      /*thenBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        // early exit for scalar == 0 or zero point
        builder.create<scf::YieldOp>(loc, windowSum);
      },
      /*elseBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder b0(loc, builder);
        auto oneSF = b0.create<field::ConstantOp>(scalarFieldType_, 1);
        auto scalarIsOne =
            b0.create<field::CmpOp>(arith::CmpIPredicate::eq, scalar, oneSF);
        auto scalarIsOneIfOp = b0.create<scf::IfOp>(
            scalarIsOne,
            /*thenBuilder=*/
            [&](OpBuilder &scalarIsOneBuilder, Location scalarIsOneLoc) {
              ImplicitLocOpBuilder b1(scalarIsOneLoc, scalarIsOneBuilder);
              ValueRange res =
                  scalarIsOneBranch(point, windowOffset, windowSum, b1);
              b1.create<scf::YieldOp>(res);
            },
            /*elseBuilder=*/
            [&](OpBuilder &scalarIsNotOneBuilder, Location scalarIsNotOneLoc) {
              ImplicitLocOpBuilder b1(scalarIsNotOneLoc, scalarIsNotOneBuilder);
              scalarIsNotOneBranch(scalar, point, buckets, windowOffset, b1);
              b1.create<scf::YieldOp>(windowSum);
            });
        b0.create<scf::YieldOp>(scalarIsOneIfOp.getResults());
      });
  return zeroScalarMulIfOp.getResults();
}

void PippengersGeneric::bucketAccReduc() {
  Operation *windowsForOp = nullptr;
  Value j;
  Value numBuckets = b_.create<arith::ConstantIndexOp>(numBuckets_);
  Value numWindows = b_.create<arith::ConstantIndexOp>(numWindows_);

  if (parallel_) {
    auto bucketsType = MemRefType::get(
        {static_cast<int64_t>(numWindows_), static_cast<int64_t>(numBuckets_)},
        outputType_);
    allBuckets_ = b_.create<memref::AllocOp>(bucketsType);
    // Initialize all buckets to zero point
    b_.create<scf::ParallelOp>(
        ValueRange{zero_, zero_}, ValueRange{numWindows, numBuckets},
        ValueRange{one_, one_},
        [&](OpBuilder &parallelBuilder, Location parallelLoc, ValueRange ivs) {
          ImplicitLocOpBuilder b0(parallelLoc, parallelBuilder);
          Value windowIdx = ivs[0];
          Value bucketIdx = ivs[1];
          b0.create<memref::StoreOp>(zeroPoint_, allBuckets_,
                                     ValueRange{windowIdx, bucketIdx});
          b0.create<scf::ReduceOp>();
        });
  } else {
    auto bucketsType =
        MemRefType::get({static_cast<int64_t>(numBuckets_)}, outputType_);
    allBuckets_ = b_.create<memref::AllocOp>(bucketsType);
    // Initialization should be done per window since we reuse same buffer in
    // the serial case.
  }

  if (parallel_) {
    auto parOp = b_.create<scf::ParallelOp>(zero_, numWindows, one_);
    b_.setInsertionPointToStart(parOp.getBody());
    j = parOp.getInductionVars()[0];
    windowsForOp = parOp;
  } else {
    auto forOp = b_.create<scf::ForOp>(zero_, numWindows, one_);
    b_.setInsertionPointToStart(forOp.getBody());
    j = forOp.getInductionVar();
    windowsForOp = forOp;
  }

  // https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/pippengers-algorithm#id-2.-bucket-accumulation
  Value bitsPerWindow = b_.create<arith::ConstantIndexOp>(bitsPerWindow_);
  Value windowOffset = b_.create<arith::MulIOp>(bitsPerWindow, j);

  Value buckets;
  if (parallel_) {
    SmallVector<OpFoldResult> offsets = {Value(j), b_.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes = {b_.getIndexAttr(1),
                                       b_.getIndexAttr(numBuckets_)};
    SmallVector<OpFoldResult> strides = {b_.getIndexAttr(1),
                                         b_.getIndexAttr(1)};

    // Create subview for current window's buckets
    auto layout =
        StridedLayoutAttr::get(b_.getContext(), ShapedType::kDynamic, {1});
    MemRefType bucketsType = MemRefType::get(
        {static_cast<int64_t>(numBuckets_)}, outputType_, layout);
    buckets = b_.create<memref::SubViewOp>(bucketsType, allBuckets_,
                                           /*offsets=*/offsets,
                                           /*sizes=*/sizes,
                                           /*strides=*/strides);
  } else {
    buckets = allBuckets_;
    // Initialize the buckets to zero point
    b_.create<scf::ForOp>(zero_, numBuckets, one_, std::nullopt,
                          [&](OpBuilder &nestedBuilder, Location nestedLoc,
                              Value i, ValueRange args) {
                            ImplicitLocOpBuilder b0(nestedLoc, nestedBuilder);
                            b0.create<memref::StoreOp>(zeroPoint_, buckets, i);
                            b0.create<scf::YieldOp>();
                          });
  }

  auto scalarMulsForOp = b_.create<scf::ForOp>(
      zero_, numScalarMuls_, one_,
      /*windowSum=*/zeroPoint_,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value i,
          ValueRange args) {
        ImplicitLocOpBuilder b0(nestedLoc, nestedBuilder);
        auto windowSum = bucketSingleAcc(i, args[0], buckets, windowOffset, b0);
        b0.create<scf::YieldOp>(windowSum);
      });

  bucketReduction(j, scalarMulsForOp.getResult(0), buckets, b_);
  b_.setInsertionPointAfter(windowsForOp);
}

} // namespace mlir::zkir::elliptic_curve
