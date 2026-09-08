// Copyright 2026 The PrimeIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

// Outlining of the tower-basis mul/square expansion (fractalyze/prime-ir#390).
// Inline, a level-k multiply is a 3ᵏ-way Karatsuba recursion emitted as
// straight-line IR, so it is exponential in the level and is re-emitted at
// every call site. Outlined, each level becomes ONE shared helper that calls
// the level below, making the IR linear in the level and shared across sites.

// RUN: prime-ir-opt --binary-field-to-arith="outline-tower-ops=true outline-min-tower-level=4" %s \
// RUN:   | FileCheck %s --check-prefix=OUTLINE
// RUN: prime-ir-opt --binary-field-to-arith %s | FileCheck %s --check-prefix=INLINE
// The attribute that matters is the one on the LOWERED function: `func.func`'s
// inherent `no_inline` is not forwarded by `FuncToLLVM`, so checking only the
// pre-lowering form would still pass while LLVM re-inlined every helper and
// undid the outlining entirely.
// RUN: prime-ir-opt --field-to-llvm="outline-tower-ops=true" %s \
// RUN:   | FileCheck %s --check-prefix=LOWERED

!BF32 = !field.bf<5>   // GF(2³²), above the threshold used here
!BF8 = !field.bf<3>    // GF(2⁸), below it

// A level-5 multiply becomes a call; nothing of the tower is expanded here.
// OUTLINE-LABEL: func.func @mul32
// OUTLINE: call @__prime_ir_bf_mul_l5
// OUTLINE-NOT: arith.andi

// Without the option the tower expands inline: no helper, no call, and the
// GF(2) leaves of the Karatsuba recursion appear as arith.andi.
// INLINE-LABEL: func.func @mul32
// INLINE-NOT: call @__prime_ir_bf_
// INLINE: arith.andi
func.func @mul32(%a: !BF32, %b: !BF32) -> !BF32 {
  %c = field.mul %a, %b : !BF32
  return %c : !BF32
}

// Square is outlined on the same threshold, under its own symbol.
// OUTLINE-LABEL: func.func @square32
// OUTLINE: call @__prime_ir_bf_square_l5
func.func @square32(%a: !BF32) -> !BF32 {
  %c = field.square %a : !BF32
  return %c : !BF32
}

// Level 3 is below the threshold, so it still expands inline and creates no
// helper. The threshold is what bounds call depth: every level below it is
// paid once, inline, inside the lowest helper.
// OUTLINE-LABEL: func.func @mul8
// OUTLINE-NOT: call @__prime_ir_bf_
// OUTLINE: arith.andi
func.func @mul8(%a: !BF8, %b: !BF8) -> !BF8 {
  %c = field.mul %a, %b : !BF8
  return %c : !BF8
}

// Both multiplies bind to the SAME helper. That sharing is the point: N
// multiplies cost one body plus N calls, not N inline expansions.
// OUTLINE-LABEL: func.func @two_muls
// OUTLINE-COUNT-2: call @__prime_ir_bf_mul_l5
func.func @two_muls(%a: !BF32, %b: !BF32, %c: !BF32) -> !BF32 {
  %d = field.mul %a, %b : !BF32
  %e = field.mul %d, %c : !BF32
  return %e : !BF32
}

// The level-5 helper expands exactly ONE Karatsuba level: three level-4
// sub-products, each a call to the next helper down. This is what turns the
// 3ᵏ expansion into a chain of k bodies.
//
// The helpers are private and internally linked (so CPU codegen still sees an
// externally visible definition in the module) and carry llvm.no_inline:
// without it LLVM folds these small bodies back into every caller and the IR
// reduction disappears before codegen — measured in fractalyze/prime-ir#367.
//
// OUTLINE:      func.func private @__prime_ir_bf_mul_l5
// OUTLINE-SAME: llvm.linkage = #llvm.linkage<internal>
// OUTLINE-SAME: llvm.no_inline
// OUTLINE-COUNT-3: call @__prime_ir_bf_mul_l4

// The threshold-level helper terminates the chain: its body expands the rest
// of the tower inline, so it calls nothing further.
// OUTLINE:      func.func private @__prime_ir_bf_mul_l4
// OUTLINE-SAME: llvm.no_inline
// OUTLINE-NOT:  call @__prime_ir_bf_

// Squaring recurses on both halves, so its level-5 helper makes two level-4
// calls (a² = (a₀² + a₁²) + βₖ₋₁·a₁²·X needs a₀² and a₁²).
// OUTLINE:      func.func private @__prime_ir_bf_square_l5
// OUTLINE-SAME: llvm.no_inline
// OUTLINE-COUNT-2: call @__prime_ir_bf_square_l4
// OUTLINE:      func.func private @__prime_ir_bf_square_l4

// LOWERED: llvm.func internal @__prime_ir_bf_mul_l5
// LOWERED-SAME: no_inline
// LOWERED: llvm.func internal @__prime_ir_bf_mul_l4
// LOWERED-SAME: no_inline
// LOWERED: llvm.func internal @__prime_ir_bf_square_l5
// LOWERED-SAME: no_inline
