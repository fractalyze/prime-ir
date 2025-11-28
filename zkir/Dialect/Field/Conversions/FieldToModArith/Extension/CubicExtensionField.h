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

#ifndef ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_EXTENSION_CUBICEXTENSIONFIELD_H_
#define ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_EXTENSION_CUBICEXTENSIONFIELD_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"

namespace mlir::zkir::field {

// Helper class to perform cubic extension field operations.
// This class encapsulates the logic for generating MLIR operations
// that implement cubic extension field arithmetic (Fp3).
//
// A cubic extension field element is represented as:
//   a = c0 + c1*u + c2*u²
// where u^3 = xi (the non-residue).
class CubicExtensionField {
public:
  // Constructs a CubicExtensionField with the given builder and non-residue.
  // The xi value is a constant representing the non-residue element.
  explicit CubicExtensionField(ImplicitLocOpBuilder &b, Value xi);

  // Computes the square of a cubic extension field element.
  // Uses the CH-SQR2 algorithm for optimized squaring.
  // Reference: "Multiplication and Squaring on Pairing-Friendly Fields"
  //            by Devegili, OhEigeartaigh, Scott, Dahab (Section 4).
  // https://eprint.iacr.org/2006/471.pdf
  //
  // Input: coefficients c0, c1, c2 where a = c0 + c1*u + c2*u²
  // Returns: coefficients of a²
  SmallVector<Value, 3> square(Value c0, Value c1, Value c2);

  // Computes the product of two cubic extension field elements.
  // Uses the schoolbook multiplication algorithm.
  //
  // Input: lhs = (a0, a1, a2), rhs = (b0, b1, b2)
  // Returns: coefficients of lhs * rhs
  SmallVector<Value, 3> mul(Value a0, Value a1, Value a2, Value b0, Value b1,
                            Value b2);

  // Computes the inverse of a cubic extension field element.
  //
  // Input: coefficients c0, c1, c2 where a = c0 + c1*u + c2*u²
  // Returns: coefficients of a^(-1)
  SmallVector<Value, 3> inverse(Value c0, Value c1, Value c2);

private:
  // Multiplies a base field element by the non-residue xi.
  Value mulByNonResidue(Value v);

  ImplicitLocOpBuilder &b_;
  Value xi_;
};

} // namespace mlir::zkir::field

// NOLINTNEXTLINE(whitespace/line_length)
#endif // ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_EXTENSION_CUBICEXTENSIONFIELD_H_
