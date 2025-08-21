#ifndef ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_MSM_PIPPENGERS_PIPPENGERS_H_
#define ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_MSM_PIPPENGERS_PIPPENGERS_H_

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::zkir::elliptic_curve {

class Pippengers {
public:
  Pippengers(Value scalars, Value points, Type baseFieldType, Type outputType,
             ImplicitLocOpBuilder &b, int32_t degree, int32_t windowBits);
  Value generate();

protected:
  // Bucket Reduction - reduce buckets to one window sum
  void bucketReduction(Value j, Value initialPoint, Value buckets,
                       ImplicitLocOpBuilder &b);

  // Window Reduction - reduce windows to one total MSM result
  Value windowReduction();

  size_t numBuckets_;
  size_t bitsPerWindow_;
  Value numScalarMuls_; // Index
  Value numWindows_;    // Index

  Value zero_; // Index
  Value one_;  // Index
  Value zeroPoint_;
  Value scalars_;
  Value points_;

  Value windowSums_;

  field::PrimeFieldType scalarFieldType_;
  Type outputType_;
  ImplicitLocOpBuilder &b_;
};

} // namespace mlir::zkir::elliptic_curve

// NOLINTNEXTLINE(whitespace/line_length)
#endif // ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_MSM_PIPPENGERS_PIPPENGERS_H_
