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

#ifndef PRIME_IR_DIALECT_FIELD_CONVERSIONS_BINARYFIELDTOARITH_BINARYFIELDOUTLINER_H_
#define PRIME_IR_DIALECT_FIELD_CONVERSIONS_BINARYFIELDTOARITH_BINARYFIELDOUTLINER_H_

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "prime_ir/Utils/FunctionOutlinerBase.h"

namespace mlir::prime_ir::field {

/// Generates outlined `func.func` helpers for tower-basis binary field
/// multiply and square.
///
/// A level-k tower multiply is a 3ᵏ-way Karatsuba recursion emitted as
/// straight-line IR, so inlining it at every call site is exponential in the
/// level: a single `bf<7>` multiply is ~25.7k `arith` ops and a `bf<7>`
/// inverse (whose norm descent is built from level-6 muls and squares) is
/// ~43k. A `divide` is both, and every additional divide in the module pays
/// the full cost again — enough to stall LLVM for over 1800 s
/// (fractalyze/prime-ir#390).
///
/// Outlining turns that exponential into a linear chain: the level-k helper
/// body expands exactly ONE Karatsuba level and calls the level-(k−1) helper
/// for its three sub-products, so the whole tower costs one shared body per
/// level instead of 3ᵏ inline leaves per call site. Levels below
/// `minTowerLevel` still expand inline, which bounds the call depth — the
/// threshold trades IR size against call overhead.
///
/// Helpers carry `llvm.no_inline`. Without it LLVM re-inlines the small
/// bodies and the IR reduction disappears before it reaches codegen; that
/// effect was measured directly in fractalyze/prime-ir#367.
class BinaryFieldOutliner : public FunctionOutlinerBase<BinaryFieldOutliner> {
  using Base = FunctionOutlinerBase<BinaryFieldOutliner>;

public:
  BinaryFieldOutliner(ModuleOp module, unsigned minTowerLevel)
      : Base(module), minTowerLevel_(minTowerLevel) {}

  /// Levels at or above the threshold are emitted as a call to a shared
  /// helper; below it the tower expands inline as before.
  bool shouldOutline(unsigned towerLevel) const {
    return towerLevel >= minTowerLevel_;
  }

  /// Emit a call to the outlined level-k multiply: (iⁿ, iⁿ) -> iⁿ, n = 2ᵏ.
  Value emitMulCall(ImplicitLocOpBuilder &b, Value lhs, Value rhs,
                    unsigned towerLevel);

  /// Emit a call to the outlined level-k square: (iⁿ) -> iⁿ, n = 2ᵏ.
  Value emitSquareCall(ImplicitLocOpBuilder &b, Value input,
                       unsigned towerLevel);

private:
  unsigned minTowerLevel_;
};

} // namespace mlir::prime_ir::field

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_FIELD_CONVERSIONS_BINARYFIELDTOARITH_BINARYFIELDOUTLINER_H_
