#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/MSM/Pippengers/Pippengers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"

namespace mlir::zkir::elliptic_curve {

// https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/pippengers-algorithm#id-3.-bucket-reduction
void Pippengers::bucketReduction(Value j, Value initialPoint, Value buckets,
                                 ImplicitLocOpBuilder &b) {
  auto numBuckets = b_.create<arith::ConstantIndexOp>(numBuckets_);

  // TODO(ashjeong): explore potential for loop parallelization
  auto bucketsForOp = b_.create<scf::ForOp>(
      zero_, numBuckets, one_,
      ValueRange{/*runningSum=*/zeroPoint_,
                 /*windowSum=*/initialPoint},
      [&](OpBuilder &builder, Location loc, Value i, ValueRange args) {
        ImplicitLocOpBuilder b_(loc, builder);
        auto idxTmp1 = b_.create<arith::SubIOp>(numBuckets, i);
        Value idx = b_.create<arith::SubIOp>(idxTmp1, one_);

        auto bucket = b_.create<memref::LoadOp>(buckets, idx);
        auto rSum =
            b_.create<elliptic_curve::AddOp>(outputType_, args[0], bucket);
        auto wSum =
            b_.create<elliptic_curve::AddOp>(outputType_, args[1], rSum);

        b_.create<scf::YieldOp>(ValueRange{rSum, wSum});
      });
  b_.create<memref::StoreOp>(bucketsForOp.getResult(1), windowSums_, j);
}

// https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/pippengers-algorithm#id-4.-window-reduction-final-msm-result
Value Pippengers::windowReduction() {
  // We're traversing windows from high to low.
  auto windowsForOp = b_.create<scf::ForOp>(
      one_, numWindows_, one_, ValueRange{/*accumulator=*/zeroPoint_},
      [&](OpBuilder &winReducBuilder, Location winReducLoc, Value j,
          ValueRange winReducArgs) {
        ImplicitLocOpBuilder b1(winReducLoc, winReducBuilder);
        // scf::ForOp does not support reverse traversal. Reverse traversal
        // must be simulated using arithmetic with the for op index
        // (numWindows - j)
        Value idx = b1.create<arith::SubIOp>(numWindows_, j);
        Value bitsPerWindow = b1.create<arith::ConstantIndexOp>(bitsPerWindow_);

        auto accumulator = winReducArgs[0];
        auto windowSum = b1.create<memref::LoadOp>(windowSums_, idx);
        accumulator = b1.create<elliptic_curve::AddOp>(outputType_, accumulator,
                                                       windowSum);
        auto bitAccForOp = b1.create<scf::ForOp>(
            zero_, bitsPerWindow, one_, accumulator,
            [&](OpBuilder &bitAccBuilder, Location bitAccLoc, Value i,
                ValueRange bitAccArgs) {
              ImplicitLocOpBuilder b2(bitAccLoc, bitAccBuilder);
              Value doubled = b2.create<elliptic_curve::DoubleOp>(
                  outputType_, bitAccArgs[0]);
              b2.create<scf::YieldOp>(doubled);
            });
        b1.create<scf::YieldOp>(bitAccForOp.getResult(0));
      });

  auto windowSumsAtZero = b_.create<memref::LoadOp>(windowSums_, zero_);
  return b_.create<elliptic_curve::AddOp>(
      outputType_, windowsForOp.getResult(0), windowSumsAtZero);
}

}  // namespace mlir::zkir::elliptic_curve
