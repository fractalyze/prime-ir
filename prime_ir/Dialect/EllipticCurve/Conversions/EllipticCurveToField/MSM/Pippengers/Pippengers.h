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

#ifndef PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_MSM_PIPPENGERS_PIPPENGERS_H_
#define PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_MSM_PIPPENGERS_PIPPENGERS_H_

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::prime_ir::elliptic_curve {

class Pippengers {
public:
  Pippengers(Value scalars, Value points, Type outputType,
             ImplicitLocOpBuilder &b, int32_t degree, int32_t windowBits);
  Value generate();

protected:
  // Bucket Reduction - reduce buckets to one window sum
  void bucketReduction(Value j, Value initialPoint, Value buckets,
                       ImplicitLocOpBuilder &b);

  // Window Reduction - reduce windows to one total MSM result
  Value windowReduction();

  size_t numBuckets;
  size_t bitsPerWindow;
  size_t numWindows;
  Value numScalarMuls; // Index

  Value zero; // Index
  Value one;  // Index
  Value zeroPoint;
  Value scalars;
  Value points;

  Value windowSums;
  Value allBuckets;

  field::PrimeFieldType scalarFieldType;
  Type outputType;
  ImplicitLocOpBuilder &b;
};

} // namespace mlir::prime_ir::elliptic_curve

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_MSM_PIPPENGERS_PIPPENGERS_H_
