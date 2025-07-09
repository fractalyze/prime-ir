#ifndef ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_MSM_PIPPENGERS_SIGNEDBUCKETINDEX_H_
#define ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_MSM_PIPPENGERS_SIGNEDBUCKETINDEX_H_

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/MSM/Pippengers/Pippengers.h"

namespace mlir::zkir::elliptic_curve {

// Process is as follows:
//
// bucketAccReduc(): generate window sums from scalars and points {
// Windows Loop {
//   runSingleWindow():  {
//   Scalar muls Loop {
//     calculate carry + scalar per window
//     populateBuckets()
//   }
//   bucketReduction(): reduce buckets to one window sum per window
//   }
// }
// }
// windowReduction(): reduce window sums to MSM result
// https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/signed-bucket-index
class PippengersSignedBucketIndex : public Pippengers {
 public:
  PippengersSignedBucketIndex(Value scalars, Value points, Type baseFieldType,
                              Type outputType, ImplicitLocOpBuilder &b,
                              bool parallel, int32_t degree, int32_t windowBits)
      : Pippengers(parallel, scalars, points, baseFieldType, outputType, b,
                   degree, windowBits) {}

 private:
  // Populate buckets by adding or subtracting the point from the corresponding
  // bucket.
  void populateBuckets(Value cutScalar, Value buckets, Value point,
                       ImplicitLocOpBuilder &b);
  // Loop through all scalar-point pairs.
  void runSingleWindow(Value j, Value carries, Value buckets, Value numBuckets,
                       Value isLastWindow);
  void bucketAccReduc() override;
};

}  // namespace mlir::zkir::elliptic_curve

#endif  // ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_MSM_PIPPENGERS_SIGNEDBUCKETINDEX_H_
