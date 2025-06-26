#ifndef ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_MSM_PIPPENGERS_GENERIC_H_
#define ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_MSM_PIPPENGERS_GENERIC_H_

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/MSM/Pippengers/Pippengers.h"

namespace mlir::zkir::elliptic_curve {

// https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/pippengers-algorithm
class PippengersGeneric : public Pippengers {
 public:
  PippengersGeneric(Value scalars, Value points, Type baseFieldType,
                    Type outputType, ImplicitLocOpBuilder &b, bool parallel,
                    int32_t degree, int32_t windowBits)
      : Pippengers(scalars, points, baseFieldType, outputType, b, degree,
                   windowBits),
        parallel_(parallel) {
    // Note that the required number of buckets per window is 2^{bitsPerWindow}
    // - 1 since we don't need the "zero" bucket.
    numBuckets_ = (1 << bitsPerWindow_) - 1;
  }

  // Process is as follows:
  //
  // bucketAccReduc(): generate window sums from scalars and points {
  //   Windows Loop {
  //     Scalar muls Loop {
  //       bucketSingleAcc() {
  //         - Early exit if scalar or point is 0
  //         - scalarIsOneBranch(): IF scalar is 1
  //             - if window offset is 0, add point to window sum
  //         - scalarIsNotOneBranch(): ELSE
  //             - scalarDecomposition(): calculate scalar slice @ window
  //             - populate bucket
  //       }
  //       bucketReduction(): reduce buckets to one window sum per window
  //     }
  //   }
  // }
  // windowReduction(): reduce window sums to MSM result
  Value generate() {
    bucketAccReduc();
    return windowReduction();
  }

 private:
  ValueRange scalarIsOneBranch(Value point, Value windowOffset, Value windowSum,
                               ImplicitLocOpBuilder &b);

  // Scalar Decomposition - reduce scalars into window slices
  Value scalarDecomposition(IntegerType scalarIntType, Value scalar,
                            Value windowOffset, ImplicitLocOpBuilder &b);

  void scalarIsNotOneBranch(Value scalar, Value point, Value buckets,
                            Value windowOffset, ImplicitLocOpBuilder &b);

  // Bucket Single Accumulation - potentially populate a single bucket for a
  // given scalar mul and window
  ValueRange bucketSingleAcc(Value i, Value windowSum, Value buckets,
                             Value windowOffset, ImplicitLocOpBuilder &b);

  // Bucket Accumulation and Reduction - populate buckets for each window
  // (accumulation), then reduce buckets per window (reduction)
  void bucketAccReduc();

  bool parallel_;
};

}  // namespace mlir::zkir::elliptic_curve

#endif  // ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_MSM_PIPPENGERS_GENERIC_H_
