/* Copyright 2026 The PrimeIR Authors.

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

#ifndef PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_CONVERSIONUTILS_H_
#define PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_CONVERSIONUTILS_H_

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

namespace mlir::prime_ir::elliptic_curve {

Operation::result_range toCoords(ImplicitLocOpBuilder &b,
                                 Value extFieldElement);

Value fromCoords(ImplicitLocOpBuilder &b, Type type, ValueRange coeffs);

} // namespace mlir::prime_ir::elliptic_curve

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_CONVERSIONUTILS_H_
