/* Copyright 2025 The PrimeIR Authors.

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

#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/MSM/Pippengers/Pippengers.h"

#include <cmath>
#include <limits>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"

namespace mlir::prime_ir::elliptic_curve {
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

Pippengers::Pippengers(Value scalarsIn, Value points, Type outputType,
                       ImplicitLocOpBuilder &b, int32_t degree,
                       int32_t windowBits)
    : points(points), outputType(outputType), b(b) {
  zero = arith::ConstantIndexOp::create(b, 0);
  one = arith::ConstantIndexOp::create(b, 1);

  auto scalarsType = cast<RankedTensorType>(scalarsIn.getType());

  scalarFieldType = cast<field::PrimeFieldType>(
      field::getStandardFormType(scalarsType.getElementType()));
  scalars = field::isMontgomery(scalarsType)
                ? field::FromMontOp::create(
                      b, field::getStandardFormType(scalarsType), scalarsIn)
                      .getResult()
                : scalarsIn;

  size_t scalarBitWidth = scalarFieldType.getDenseElementBitSize();
  bitsPerWindow =
      windowBits > 0
          ? windowBits
          : estimateOptimalWindowBits(scalarBitWidth, size_t{1} << degree);
  numWindows = computeWindowsCount(scalarBitWidth, bitsPerWindow);
  numScalarMuls = tensor::DimOp::create(b, scalars, 0);

  zeroPoint = createZeroPoint(b, outputType);

  auto windowSumsType =
      MemRefType::get({static_cast<int64_t>(numWindows)}, outputType);
  windowSums = memref::AllocaOp::create(b, windowSumsType);

  Value numWindows = arith::ConstantIndexOp::create(b, this->numWindows);
  scf::ForOp::create(
      b, zero, numWindows, one, ValueRange{},
      [&](OpBuilder &builder, Location loc, Value i, ValueRange args) {
        ImplicitLocOpBuilder b0(loc, builder);
        memref::StoreOp::create(b0, zeroPoint, windowSums, i);
        scf::YieldOp::create(b0);
      });
}

// https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/pippengers-algorithm#id-3.-bucket-reduction
void Pippengers::bucketReduction(Value j, Value initialPoint, Value buckets,
                                 ImplicitLocOpBuilder &b) {
  auto numBuckets = arith::ConstantIndexOp::create(b, this->numBuckets);

  // TODO(ashjeong): explore potential for loop parallelization
  auto bucketsForOp = scf::ForOp::create(
      b, zero, numBuckets, one,
      ValueRange{/*runningSum=*/zeroPoint,
                 /*windowSum=*/initialPoint},
      [&](OpBuilder &builder, Location loc, Value i, ValueRange args) {
        ImplicitLocOpBuilder b0(loc, builder);
        auto idxTmp1 = arith::SubIOp::create(b0, numBuckets, i);
        Value idx = arith::SubIOp::create(b0, idxTmp1, one);

        auto bucket = memref::LoadOp::create(b0, buckets, idx);
        auto rSum = AddOp::create(b0, outputType, args[0], bucket);
        auto wSum = AddOp::create(b0, outputType, args[1], rSum);

        scf::YieldOp::create(b0, ValueRange{rSum, wSum});
      });
  memref::StoreOp::create(b, bucketsForOp.getResult(1), windowSums, j);
}

// https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/pippengers-algorithm#id-4.-window-reduction-final-msm-result
Value Pippengers::windowReduction() {
  Value numWindows = arith::ConstantIndexOp::create(b, this->numWindows);
  // We're traversing windows from high to low.
  auto windowsForOp = scf::ForOp::create(
      b, one, numWindows, one, ValueRange{/*accumulator=*/zeroPoint},
      [&](OpBuilder &winReducBuilder, Location winReducLoc, Value j,
          ValueRange winReducArgs) {
        ImplicitLocOpBuilder b1(winReducLoc, winReducBuilder);
        // scf::ForOp does not support reverse traversal. Reverse traversal
        // must be simulated using arithmetic with the for op index
        // (numWindows - j)
        Value idx = arith::SubIOp::create(b1, numWindows, j);
        Value bitsPerWindow =
            arith::ConstantIndexOp::create(b1, this->bitsPerWindow);

        auto accumulator = winReducArgs[0];
        auto windowSum = memref::LoadOp::create(b1, windowSums, idx);
        accumulator = AddOp::create(b1, outputType, accumulator, windowSum);
        auto bitAccForOp = scf::ForOp::create(
            b1, zero, bitsPerWindow, one, accumulator,
            [&](OpBuilder &bitAccBuilder, Location bitAccLoc, Value i,
                ValueRange bitAccArgs) {
              ImplicitLocOpBuilder b2(bitAccLoc, bitAccBuilder);
              Value doubled = DoubleOp::create(b2, outputType, bitAccArgs[0]);
              scf::YieldOp::create(b2, doubled);
            });
        scf::YieldOp::create(b1, bitAccForOp.getResult(0));
      });

  auto windowSumsAtZero = memref::LoadOp::create(b, windowSums, zero);
  return AddOp::create(b, outputType, windowsForOp.getResult(0),
                       windowSumsAtZero);
}

} // namespace mlir::prime_ir::elliptic_curve
