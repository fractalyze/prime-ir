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
  // When lazy is false, returns a value in [0, p).
  // When lazy is true, skips final conditional subtraction and returns [0, 2p).
  Value reduce(Value tLow, Value tHigh, bool lazy = false);

  // Reduces a value in [0, bound * p) to canonical form [0, p).
  // For bound <= 1 returns the input as-is; for bound == 2 uses a single
  // conditional subtraction; for bound > 2 uses binary conditional subtraction
  // in ceil(log₂(bound)) steps.
  Value getCanonicalFromExtended(Value input, uint64_t bound = 2);

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
  // For dynamic tensor shapes, inputValue is used to extract runtime dims
  // for the linalg.fill broadcast pattern.
  Value createModulusConst(Type inputType, Value inputValue = {});

  // Single-limb REDC: modulus fits in one machine word.
  // When lazy is true, skips the final conditional subtraction and returns
  // a value in [0, 2p); when false, returns a value in [0, p).
  Value reduceSingleLimb(Value tLow, Value tHigh, bool lazy);

  // Multi-limb REDC: modulus requires multiple machine words.
  // When lazy is true, skips the final conditional subtraction and returns
  // a value in [0, 2p); when false, returns a value in [0, p).
  Value reduceMultiLimb(Value tLow, Value tHigh, bool lazy);

  // Checks if the input is from a signed multiplication.
  bool isFromSignedMul(Value input);

  ImplicitLocOpBuilder &b;
  TypedAttr modAttr;
  MontgomeryAttr montAttr;
};

} // namespace mlir::prime_ir::mod_arith

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_REDUCER_MONTREDUCER_H_
