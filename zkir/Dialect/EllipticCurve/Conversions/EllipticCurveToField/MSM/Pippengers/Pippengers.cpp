/* Copyright 2025 The ZKIR Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/MSM/Pippengers/Pippengers.h"

#include <cmath>
#include <limits>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"

namespace mlir::zkir::elliptic_curve {
namespace {

// https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/pippengers-algorithm#total-complexity
constexpr size_t estimateOptimalWindowBits(size_t lambda, size_t n) {
  size_t optimalBits = 1;
  size_t minCost = std::numeric_limits<size_t>::max();

  for (size_t s = 1; s <= lambda; ++s) {
    // ⌈λ/s⌉ × (n + 2^(s+1) - 1) × PointAdd
    size_t numWindows = (lambda + s - 1) / s;
    size_t pointAddCost = numWindows * (n + (size_t{1} << (s + 1)) - 1);

    if (pointAddCost < minCost) {
      minCost = pointAddCost;
      optimalBits = s;
    } else {
      // We've found the optimal window size.
      break;
    }
  }

  return optimalBits;
}

constexpr size_t computeWindowsCount(size_t scalarBitWidth,
                                     size_t bitsPerWindow) {
  return (scalarBitWidth + bitsPerWindow - 1) / bitsPerWindow;
}

} // namespace

Pippengers::Pippengers(Value scalars, Value points, Type outputType,
                       ImplicitLocOpBuilder &b, int32_t degree,
                       int32_t windowBits)
    : points_(points), outputType_(outputType), b_(b) {
  zero_ = b.create<arith::ConstantIndexOp>(0);
  one_ = b.create<arith::ConstantIndexOp>(1);

  auto scalarsType = cast<RankedTensorType>(scalars.getType());

  scalarFieldType_ = cast<field::PrimeFieldType>(
      field::getStandardFormType(scalarsType.getElementType()));
  scalars_ = field::isMontgomery(scalarsType)
                 ? b.create<field::FromMontOp>(
                        field::getStandardFormType(scalarsType), scalars)
                       .getResult()
                 : scalars;

  size_t scalarBitWidth = scalarFieldType_.getStorageBitWidth();
  bitsPerWindow_ =
      windowBits > 0
          ? windowBits
          : estimateOptimalWindowBits(scalarBitWidth, size_t{1} << degree);
  numWindows_ = computeWindowsCount(scalarBitWidth, bitsPerWindow_);
  numScalarMuls_ = b.create<tensor::DimOp>(scalars_, 0);

  zeroPoint_ = createZeroPoint(b, outputType);

  auto windowSumsType =
      MemRefType::get({static_cast<int64_t>(numWindows_)}, outputType_);
  windowSums_ = b.create<memref::AllocaOp>(windowSumsType);

  Value numWindows = b.create<arith::ConstantIndexOp>(numWindows_);
  b.create<scf::ForOp>(
      zero_, numWindows, one_, std::nullopt,
      [&](OpBuilder &builder, Location loc, Value i, ValueRange args) {
        ImplicitLocOpBuilder b0(loc, builder);
        b0.create<memref::StoreOp>(zeroPoint_, windowSums_, i);
        b0.create<scf::YieldOp>();
      });
}

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
        auto rSum = b_.create<AddOp>(outputType_, args[0], bucket);
        auto wSum = b_.create<AddOp>(outputType_, args[1], rSum);

        b_.create<scf::YieldOp>(ValueRange{rSum, wSum});
      });
  b_.create<memref::StoreOp>(bucketsForOp.getResult(1), windowSums_, j);
}

// https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/pippengers-algorithm#id-4.-window-reduction-final-msm-result
Value Pippengers::windowReduction() {
  Value numWindows = b_.create<arith::ConstantIndexOp>(numWindows_);
  // We're traversing windows from high to low.
  auto windowsForOp = b_.create<scf::ForOp>(
      one_, numWindows, one_, ValueRange{/*accumulator=*/zeroPoint_},
      [&](OpBuilder &winReducBuilder, Location winReducLoc, Value j,
          ValueRange winReducArgs) {
        ImplicitLocOpBuilder b1(winReducLoc, winReducBuilder);
        // scf::ForOp does not support reverse traversal. Reverse traversal
        // must be simulated using arithmetic with the for op index
        // (numWindows - j)
        Value idx = b1.create<arith::SubIOp>(numWindows, j);
        Value bitsPerWindow = b1.create<arith::ConstantIndexOp>(bitsPerWindow_);

        auto accumulator = winReducArgs[0];
        auto windowSum = b1.create<memref::LoadOp>(windowSums_, idx);
        accumulator = b1.create<AddOp>(outputType_, accumulator, windowSum);
        auto bitAccForOp = b1.create<scf::ForOp>(
            zero_, bitsPerWindow, one_, accumulator,
            [&](OpBuilder &bitAccBuilder, Location bitAccLoc, Value i,
                ValueRange bitAccArgs) {
              ImplicitLocOpBuilder b2(bitAccLoc, bitAccBuilder);
              Value doubled = b2.create<DoubleOp>(outputType_, bitAccArgs[0]);
              b2.create<scf::YieldOp>(doubled);
            });
        b1.create<scf::YieldOp>(bitAccForOp.getResult(0));
      });

  auto windowSumsAtZero = b_.create<memref::LoadOp>(windowSums_, zero_);
  return b_.create<AddOp>(outputType_, windowsForOp.getResult(0),
                          windowSumsAtZero);
}

} // namespace mlir::zkir::elliptic_curve
