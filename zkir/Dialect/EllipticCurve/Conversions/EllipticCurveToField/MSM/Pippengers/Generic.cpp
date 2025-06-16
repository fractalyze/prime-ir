#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/MSM/Pippengers/Generic.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"

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
Value PippengersGeneric::scalarDecomposition(IntegerType scalarIntType,
                                             Value scalar,
                                             Value windowOffsetIndex,
                                             ImplicitLocOpBuilder &b) {
  Value windowOffset =
      b.create<arith::IndexCastOp>(scalarIntType, windowOffsetIndex);

  // We right-shift by `windowOffset`, thus getting rid
  // of the lower bits.
  auto signlessScalar = b.create<field::ExtractOp>(scalarIntType, scalar);
  auto upperBitsScalar = b.create<arith::ShRUIOp>(signlessScalar, windowOffset);

  // We mod the remaining bits by 2^{bitsPerWindow},
  // thus taking `bitsPerWindow` total bits.
  Value mask =
      b.create<arith::ConstantIntOp>((1 << bitsPerWindow_) - 1, scalarIntType);
  auto scalarPerWindow = b.create<arith::AndIOp>(upperBitsScalar, mask);

  return scalarPerWindow;
}

void PippengersGeneric::scalarIsNotOneBranch(Value scalar, Value point,
                                             Value buckets, Value windowOffset,
                                             ImplicitLocOpBuilder &b) {
  size_t scalarBitWidth =
      scalarFieldType_.getModulus().getValue().getBitWidth();
  auto scalarIntType = IntegerType::get(b.getContext(), scalarBitWidth);

  Value zeroInt = b.create<arith::ConstantIntOp>(0, scalarIntType);
  Value oneInt = b.create<arith::ConstantIntOp>(1, scalarIntType);
  Value scalarForWindow =
      scalarDecomposition(scalarIntType, scalar, windowOffset, b);

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
        Value adjustedIdx = b0.create<arith::IndexCastOp>(
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
  auto pointIsZero = b.create<elliptic_curve::IsZeroOp>(point);
  auto zeroScalarMul = b.create<arith::OrIOp>(scalarIsZero, pointIsZero);
  auto zeroScalarMulIfOp = b.create<scf::IfOp>(
      zeroScalarMul,
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
  b_.create<scf::ParallelOp>(
      zero_, numWindows_, one_,
      [&](OpBuilder &builder, Location loc, ValueRange inducVars) {
        ImplicitLocOpBuilder b0(loc, builder);
        Value j = inducVars[0];

        // https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/pippengers-algorithm#id-2.-bucket-accumulation
        Value bitsPerWindow = b0.create<arith::ConstantIndexOp>(bitsPerWindow_);
        Value windowOffset = b0.create<arith::MulIOp>(bitsPerWindow, j);
        MemRefType bucketsType =
            MemRefType::get({static_cast<int64_t>(numBuckets_)}, outputType_);
        auto buckets = b0.create<memref::AllocOp>(bucketsType);

        Value numBuckets = b0.create<arith::ConstantIndexOp>(numBuckets_);
        b0.create<scf::ForOp>(
            zero_, numBuckets, one_, std::nullopt,
            [&](OpBuilder &nestedBuilder, Location nestedLoc, Value i,
                ValueRange args) {
              ImplicitLocOpBuilder b1(nestedLoc, nestedBuilder);
              b1.create<memref::StoreOp>(zeroPoint_, buckets, i);
              b1.create<scf::YieldOp>();
            });

        auto scalarMulsForOp = b0.create<scf::ForOp>(
            zero_, numScalarMuls_, one_,
            /*windowSum=*/zeroPoint_,
            [&](OpBuilder &nestedBuilder, Location nestedLoc, Value i,
                ValueRange args) {
              ImplicitLocOpBuilder b1(nestedLoc, nestedBuilder);
              auto windowSum =
                  bucketSingleAcc(i, args[0], buckets, windowOffset, b1);
              b1.create<scf::YieldOp>(windowSum);
            });

        bucketReduction(j, scalarMulsForOp.getResult(0), buckets, b0);
      });
}

}  // namespace mlir::zkir::elliptic_curve
