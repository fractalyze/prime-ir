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

#ifndef PRIME_IR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_REDUCER_BARRETTREDUCER_H_
#define PRIME_IR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_REDUCER_BARRETTREDUCER_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h" // IWYU pragma: keep

namespace mlir::prime_ir::mod_arith {

// Helper class to perform Barrett reduction.
// Lowers a modular multiplication `(a * b) mod p` to a sequence of arith
// operations using a precomputed `mu = floor(2^(2k) / p)`, where `k` is the
// modulus storage bit width.
//
// Algorithm (single conditional subtraction; assumes `a, b < p`):
//   prod = a * b                       in [0, p²)
//   q'   = high_half(prod * mu)        ≈ floor(prod / p), in [0, mu]
//   r    = prod - q' * p               in [0, 2p)
//   out  = (r >= p) ? r - p : r        in [0, p)
class BarrettReducer {
public:
  // Constructs a BarrettReducer with the given builder and ModArithType.
  // Extracts the modulus and Barrett parameters from the type.
  explicit BarrettReducer(ImplicitLocOpBuilder &b, ModArithType modArithType);

  // Lowers `(lhs * rhs) mod p` to arith ops and returns the canonical result
  // (a value of `modArithType`'s storage type in `[0, p)`).
  // Precondition: `lhs` and `rhs` are canonical (`< p`).
  Value reduce(Value lhs, Value rhs);

private:
  // Creates an extended-width constant for the modulus, broadcasting to a
  // shaped type when `inputType` is shaped. `inputValue` is used to recover
  // runtime dims for dynamic-shaped tensors (mirrors MontReducer).
  Value createExtModulusConst(Type inputType, Value inputValue = {});

  // Creates an extended-width constant for `mu`, broadcasting to a shaped
  // type when `inputType` is shaped.
  Value createExtMuConst(Type inputType, Value inputValue = {});

  ImplicitLocOpBuilder &b;
  IntegerAttr modAttr;
  IntegerAttr muAttr;
  unsigned extBitWidth;
};

} // namespace mlir::prime_ir::mod_arith

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_REDUCER_BARRETTREDUCER_H_
