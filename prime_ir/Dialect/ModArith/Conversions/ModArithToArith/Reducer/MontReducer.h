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

#ifndef PRIME_IR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_REDUCER_MONTREDUCER_H_
#define PRIME_IR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_REDUCER_MONTREDUCER_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h" // IWYU pragma: keep

namespace mlir {
class TypedAttr;
}

namespace mlir::prime_ir::mod_arith {

// Helper class to perform Montgomery reduction operations.
// This class encapsulates the logic for generating MLIR operations
// that implement the REDC (Montgomery Reduction) algorithm.
class MontReducer {
public:
  // Constructs a MontReducer with the given builder and ModArithType.
  // Extracts the modulus and Montgomery parameters from the type.
  explicit MontReducer(ImplicitLocOpBuilder &b, ModArithType modArithType);

  // Performs Montgomery reduction on the given input values.
  // Given T = tLow + tHigh * 2ʷ (where w is the modulus bit width),
  // computes T * R⁻¹ mod n, where R is the Montgomery radix.
  Value reduce(Value tLow, Value tHigh);

  // Gets the canonical form from an input value in [0, 2n).
  Value getCanonicalFromExtended(Value input);

  // Gets the canonical form from an input value in [0, 2n) and an overflow
  // flag.
  Value getCanonicalFromExtended(Value input, Value overflow);

  // Gets the canonical difference of two values in modular arithmetic.
  // Computes (lhs - rhs) mod n, returning a value in [0, n).
  // Assumes that both are in the range [0, n).
  Value getCanonicalDiff(Value lhs, Value rhs);

private:
  // Creates a properly typed constant for the modulus based on the input type.
  // Handles splatting for vector types automatically.
  Value createModulusConst(Type inputType);

  // Performs single-limb Montgomery reduction.
  // Used when the modulus fits in a single limb (2ʷ > modulus).
  Value reduceSingleLimb(Value tLow, Value tHigh);

  // Performs multi-limb Montgomery reduction.
  // Used when the modulus requires multiple limbs (2ʷ <= modulus).
  Value reduceMultiLimb(Value tLow, Value tHigh);

  // Checks if the input is from a signed multiplication.
  bool isFromSignedMul(Value input);

  ImplicitLocOpBuilder &b;
  TypedAttr modAttr;
  MontgomeryAttr montAttr;
};

} // namespace mlir::prime_ir::mod_arith

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_REDUCER_MONTREDUCER_H_
