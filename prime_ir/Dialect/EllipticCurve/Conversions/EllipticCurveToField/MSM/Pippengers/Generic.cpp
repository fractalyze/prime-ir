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

#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/MSM/Pippengers/Generic.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"

namespace mlir::prime_ir::elliptic_curve {

// We only process unit scalars once in the first window.
ValueRange PippengersGeneric::scalarIsOneBranch(Value point, Value windowOffset,
                                                Value windowSum,
                                                ImplicitLocOpBuilder &b) {
  auto windowOffsetIsZero =
      arith::CmpIOp::create(b, arith::CmpIPredicate::eq, windowOffset, zero);
  auto windowOffsetIsZeroIfOp = scf::IfOp::create(
      b, windowOffsetIsZero,
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder b0(loc, builder);
        if (isa<JacobianType>(point.getType()) && isa<XYZZType>(outputType)) {
          point = ConvertPointTypeOp::create(b0, outputType, point);
        }
        Value addedPoint = AddOp::create(b0, outputType, windowSum, point);
        scf::YieldOp::create(b0, addedPoint);
      },
      [&](OpBuilder &builder, Location loc) {
        scf::YieldOp::create(builder, loc, windowSum);
      });
  return windowOffsetIsZeroIfOp.getResults();
}

// https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/pippengers-algorithm#id-1.-scalar-decomposition
Value PippengersGeneric::scalarDecomposition(Value scalar,
                                             Value windowOffsetIndex,
                                             ImplicitLocOpBuilder &b) {
  auto scalarIntType = scalarFieldType.getStorageType();
  Value windowOffset =
      arith::IndexCastOp::create(b, scalarIntType, windowOffsetIndex);

  // We right-shift by `windowOffset`, thus getting rid
  // of the lower bits.
  Value signlessScalar =
      field::BitcastOp::create(b, TypeRange{scalarIntType}, scalar);
  auto upperBitsScalar =
      arith::ShRUIOp::create(b, signlessScalar, windowOffset);

  auto windowBitIntType = IntegerType::get(b.getContext(), bitsPerWindow);
  return arith::TruncIOp::create(b, windowBitIntType, upperBitsScalar);
}

void PippengersGeneric::scalarIsNotOneBranch(Value scalar, Value point,
                                             Value buckets, Value windowOffset,
                                             ImplicitLocOpBuilder &b) {
  auto windowBitIntType = IntegerType::get(b.getContext(), bitsPerWindow);

  Value zeroInt = arith::ConstantIntOp::create(b, windowBitIntType, 0);
  Value oneInt = arith::ConstantIntOp::create(b, windowBitIntType, 1);
  Value scalarForWindow = scalarDecomposition(scalar, windowOffset, b);

  // If the scalar is non-zero, we update the corresponding bucket. (Recall that
  // `buckets` doesn't have a zero bucket.)
  auto scalarPerWindowIsNotZero = arith::CmpIOp::create(
      b, arith::CmpIPredicate::ne, scalarForWindow, zeroInt);
  scf::IfOp::create(
      b, scalarPerWindowIsNotZero,
      /*thenBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder b0(loc, builder);
        auto scalarPerWindowMinusOne =
            arith::SubIOp::create(b0, scalarForWindow, oneInt);
        Value adjustedIdx = arith::IndexCastUIOp::create(
            b0, b0.getIndexType(), scalarPerWindowMinusOne);
        auto bucketAtAdjustedIdx =
            memref::LoadOp::create(b0, outputType, buckets, adjustedIdx);
        if (isa<JacobianType>(point.getType()) && isa<XYZZType>(outputType)) {
          point = ConvertPointTypeOp::create(b0, outputType, point);
        }
        auto newPoint =
            AddOp::create(b0, outputType, bucketAtAdjustedIdx, point);
        memref::StoreOp::create(b0, newPoint, buckets, adjustedIdx);
        scf::YieldOp::create(b0);
      });
}

ValueRange PippengersGeneric::bucketSingleAcc(Value i, Value windowSum,
                                              Value buckets, Value windowOffset,
                                              ImplicitLocOpBuilder &b) {
  auto scalar = tensor::ExtractOp::create(b, scalars, ValueRange{i});
  auto point = tensor::ExtractOp::create(b, points, ValueRange{i});

  Value zeroSF = field::createFieldZero(scalarFieldType, b);
  auto scalarIsZero =
      field::CmpOp::create(b, arith::CmpIPredicate::eq, scalar, zeroSF);
  auto zeroScalarMulIfOp = scf::IfOp::create(
      b, scalarIsZero,
      /*thenBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        // early exit for scalar == 0 or zero point
        scf::YieldOp::create(builder, loc, windowSum);
      },
      /*elseBuilder=*/
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder b0(loc, builder);
        Value oneSF = field::createFieldOne(scalarFieldType, b0);
        auto scalarIsOne =
            field::CmpOp::create(b0, arith::CmpIPredicate::eq, scalar, oneSF);
        auto scalarIsOneIfOp = scf::IfOp::create(
            b0, scalarIsOne,
            /*thenBuilder=*/
            [&](OpBuilder &scalarIsOneBuilder, Location scalarIsOneLoc) {
              ImplicitLocOpBuilder b1(scalarIsOneLoc, scalarIsOneBuilder);
              ValueRange res =
                  scalarIsOneBranch(point, windowOffset, windowSum, b1);
              scf::YieldOp::create(b1, res);
            },
            /*elseBuilder=*/
            [&](OpBuilder &scalarIsNotOneBuilder, Location scalarIsNotOneLoc) {
              ImplicitLocOpBuilder b1(scalarIsNotOneLoc, scalarIsNotOneBuilder);
              scalarIsNotOneBranch(scalar, point, buckets, windowOffset, b1);
              scf::YieldOp::create(b1, windowSum);
            });
        scf::YieldOp::create(b0, scalarIsOneIfOp.getResults());
      });
  return zeroScalarMulIfOp.getResults();
}

void PippengersGeneric::bucketAccReduc() {
  Operation *windowsForOp = nullptr;
  Value j;
  Value numBuckets = arith::ConstantIndexOp::create(b, this->numBuckets);
  Value numWindows = arith::ConstantIndexOp::create(b, this->numWindows);

  if (parallel) {
    auto bucketsType = MemRefType::get({static_cast<int64_t>(this->numWindows),
                                        static_cast<int64_t>(this->numBuckets)},
                                       outputType);
    allBuckets = memref::AllocOp::create(b, bucketsType);
    // Initialize all buckets to zero point
    scf::ParallelOp::create(
        b, ValueRange{zero, zero}, ValueRange{numWindows, numBuckets},
        ValueRange{one, one},
        [&](OpBuilder &parallelBuilder, Location parallelLoc, ValueRange ivs) {
          ImplicitLocOpBuilder b0(parallelLoc, parallelBuilder);
          Value windowIdx = ivs[0];
          Value bucketIdx = ivs[1];
          memref::StoreOp::create(b0, zeroPoint, allBuckets,
                                  ValueRange{windowIdx, bucketIdx});
          scf::ReduceOp::create(b0);
        });
  } else {
    auto bucketsType =
        MemRefType::get({static_cast<int64_t>(this->numBuckets)}, outputType);
    allBuckets = memref::AllocOp::create(b, bucketsType);
    // Initialization should be done per window since we reuse same buffer in
    // the serial case.
  }

  if (parallel) {
    auto parOp = scf::ParallelOp::create(b, zero, numWindows, one);
    b.setInsertionPointToStart(parOp.getBody());
    j = parOp.getInductionVars()[0];
    windowsForOp = parOp;
  } else {
    auto forOp = scf::ForOp::create(b, zero, numWindows, one);
    b.setInsertionPointToStart(forOp.getBody());
    j = forOp.getInductionVar();
    windowsForOp = forOp;
  }

  // https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/pippengers-algorithm#id-2.-bucket-accumulation
  Value bitsPerWindow = arith::ConstantIndexOp::create(b, this->bitsPerWindow);
  Value windowOffset = arith::MulIOp::create(b, bitsPerWindow, j);

  Value buckets;
  if (parallel) {
    SmallVector<OpFoldResult> offsets = {Value(j), b.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes = {b.getIndexAttr(1),
                                       b.getIndexAttr(this->numBuckets)};
    SmallVector<OpFoldResult> strides = {b.getIndexAttr(1), b.getIndexAttr(1)};

    // Create subview for current window's buckets
    auto layout =
        StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic, {1});
    MemRefType bucketsType = MemRefType::get(
        {static_cast<int64_t>(this->numBuckets)}, outputType, layout);
    buckets = memref::SubViewOp::create(b, bucketsType, allBuckets,
                                        /*offsets=*/offsets,
                                        /*sizes=*/sizes,
                                        /*strides=*/strides);
  } else {
    buckets = allBuckets;
    // Initialize the buckets to zero point
    linalg::FillOp::create(b, zeroPoint, buckets);
  }

  auto scalarMulsForOp = scf::ForOp::create(
      b, zero, numScalarMuls, one,
      /*windowSum=*/zeroPoint,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value i,
          ValueRange args) {
        ImplicitLocOpBuilder b0(nestedLoc, nestedBuilder);
        auto windowSum = bucketSingleAcc(i, args[0], buckets, windowOffset, b0);
        scf::YieldOp::create(b0, windowSum);
      });

  bucketReduction(j, scalarMulsForOp.getResult(0), buckets, b);
  b.setInsertionPointAfter(windowsForOp);
}

} // namespace mlir::prime_ir::elliptic_curve
