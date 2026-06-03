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

#ifndef PRIME_IR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_REDUCER_SOLINASREDUCER_H_
#define PRIME_IR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_REDUCER_SOLINASREDUCER_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h" // IWYU pragma: keep

namespace mlir::prime_ir::mod_arith {

// Helper class to perform Solinas reduction for the Goldilocks prime
// `p = 2^64 - 2^32 + 1`. This is the field's reason for being: reduction is a
// handful of 64-bit shifts/adds rather than a division.
//
// Works on the 64-bit halves of the 128-bit product directly (as produced by
// `arith.mului_extended`), staying entirely in 64-bit arithmetic — no `i128`
// and no `arith.remui i128` (which would lower to libgcc's `__umodti3`, absent
// from the CPU JIT runtime).
//
// Using `ε = 2^32 - 1` (so `2^64 ≡ ε` and `2^96 ≡ -1 (mod p)`), for
// `x = hi·2^64 + lo` with `hi = hi_hi·2^32 + hi_lo`:
//   x ≡ lo - hi_hi + hi_lo·ε  (mod p)
// Each ± wraps by `2^64 ≡ ε`, corrected by a conditional ∓ε off the
// borrow/carry bit; a final conditional subtract of `p` canonicalizes into
// `[0, p)`. (Plonky2/sppark reduce128.)
class SolinasReducer {
public:
  // Constructs a SolinasReducer with the given builder and ModArithType.
  // Precondition: the type's modulus is the Goldilocks prime.
  explicit SolinasReducer(ImplicitLocOpBuilder &b, ModArithType modArithType);

  // Reduces the 128-bit product `hi·2^64 + lo` (each half a canonical-width
  // `k`-bit integer, e.g. from `arith.mului_extended`) to the canonical
  // `k`-bit residue in `[0, p)`. Precondition: `hi·2^64 + lo < p²`.
  Value reduce(Value lo, Value hi);

private:
  ImplicitLocOpBuilder &b;
  IntegerAttr modAttr;
  unsigned bitWidth;
};

} // namespace mlir::prime_ir::mod_arith

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_REDUCER_SOLINASREDUCER_H_
