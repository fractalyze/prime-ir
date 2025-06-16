#ifndef ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_MSM_PIPPENGERS_PIPPENGERS_H_
#define ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_MSM_PIPPENGERS_PIPPENGERS_H_

#include <cmath>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::zkir::elliptic_curve {

// The result of this function is only approximately `ln(a)`.
// See https://github.com/scipr-lab/zexe/issues/79#issue-556220473
constexpr static size_t lnWithoutFloats(size_t a) {
  // log2(a) * ln(2)
  return std::log2(a) * 69 / 100;
}

constexpr size_t computeWindowsBits(size_t size) {
  if (size < 32) {
    return 3;
  } else {
    return lnWithoutFloats(size) + 2;
  }
}

constexpr size_t computeWindowsCount(size_t scalarBitWidth,
                                     size_t bitsPerWindow) {
  return (scalarBitWidth + bitsPerWindow - 1) / bitsPerWindow;
}

class Pippengers {
 public:
  Pippengers(Value scalars, Value points, Type baseFieldType, Type outputType,
             ImplicitLocOpBuilder &b)
      : points_(points), outputType_(outputType), b_(b) {
    zero_ = b_.create<arith::ConstantIndexOp>(0);
    one_ = b_.create<arith::ConstantIndexOp>(1);

    auto scalarsType = cast<RankedTensorType>(scalars.getType());

    scalarFieldType_ = cast<field::PrimeFieldType>(
        field::getStandardFormType(scalarsType.getElementType()));
    scalars_ = field::isMontgomery(scalarsType)
                   ? b.create<field::FromMontOp>(
                          field::getStandardFormType(scalarsType), scalars)
                         .getResult()
                   : scalars;

    size_t numScalarMuls = scalarsType.getShape()[0];
    size_t scalarBitWidth =
        scalarFieldType_.getModulus().getValue().getBitWidth();
    bitsPerWindow_ = computeWindowsBits(numScalarMuls);
    size_t numWindows = computeWindowsCount(scalarBitWidth, bitsPerWindow_);

    numScalarMuls_ = b.create<arith::ConstantIndexOp>(numScalarMuls);
    numWindows_ = b.create<arith::ConstantIndexOp>(numWindows);

    auto zeroBF = b.create<field::ConstantOp>(baseFieldType, 0);
    Value oneBF = field::isMontgomery(baseFieldType)
                      ? b.create<field::ToMontOp>(
                             baseFieldType,
                             b.create<field::ConstantOp>(
                                 field::getStandardFormType(baseFieldType), 1))
                            .getResult()
                      : b.create<field::ConstantOp>(baseFieldType, 1);
    zeroPoint_ = isa<XYZZType>(outputType)
                     ? b.create<elliptic_curve::PointOp>(
                           outputType, ValueRange{oneBF, oneBF, zeroBF, zeroBF})
                     : b.create<elliptic_curve::PointOp>(
                           outputType, ValueRange{oneBF, oneBF, zeroBF});

    auto windowSumsType =
        MemRefType::get({static_cast<int64_t>(numWindows)}, outputType_);
    windowSums_ = b.create<memref::AllocOp>(windowSumsType);

    b.create<scf::ForOp>(
        zero_, numWindows_, one_, std::nullopt,
        [&](OpBuilder &builder, Location loc, Value i, ValueRange args) {
          ImplicitLocOpBuilder b0(loc, builder);
          b0.create<memref::StoreOp>(zeroPoint_, windowSums_, i);
          b0.create<scf::YieldOp>();
        });
  }

  Value generate();

 protected:
  // Bucket Reduction - reduce buckets to one window sum
  void bucketReduction(Value j, Value initialPoint, Value buckets,
                       ImplicitLocOpBuilder &b);

  // Window Reduction - reduce windows to one total MSM result
  Value windowReduction();

  size_t numBuckets_;
  size_t bitsPerWindow_;
  Value numScalarMuls_;  // Index
  Value numWindows_;     // Index

  Value zero_;  // Index
  Value one_;   // Index
  Value zeroPoint_;
  Value scalars_;
  Value points_;

  Value windowSums_;

  field::PrimeFieldType scalarFieldType_;
  Type outputType_;
  ImplicitLocOpBuilder &b_;
};

}  // namespace mlir::zkir::elliptic_curve

#endif  // ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_MSM_PIPPENGERS_PIPPENGERS_H_
