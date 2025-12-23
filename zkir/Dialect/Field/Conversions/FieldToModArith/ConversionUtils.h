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

#ifndef ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_CONVERSIONUTILS_H_
#define ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_CONVERSIONUTILS_H_

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::zkir::field {

Operation::result_range toCoeffs(ImplicitLocOpBuilder &b,
                                 Value extFieldElement);

Value fromCoeffs(ImplicitLocOpBuilder &b, Type type, ValueRange coeffs);

// Create a mod_arith constant with value n.
Value createConst(ImplicitLocOpBuilder &b, PrimeFieldType baseField, int64_t n);

// Create a mod_arith constant with the multiplicative inverse of n.
Value createInvConst(ImplicitLocOpBuilder &b, PrimeFieldType baseField,
                     int64_t n);

// Create a mod_arith constant with value numerator / denominator.
Value createRationalConst(ImplicitLocOpBuilder &b, PrimeFieldType baseField,
                          int64_t numerator, int64_t denominator);

} // namespace mlir::zkir::field

#endif // ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_CONVERSIONUTILS_H_
