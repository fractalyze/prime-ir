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

// Every Solinas prime p = 2^a - 2^b + 1 (its p-2 has one zero bit) is
// auto-detected from the modulus and lowered under fermat to ONE general
// branch-free addition chain — a fixed count of mod_arith.mul (+ square), no
// mod_arith.inverse, no per-field code. A non-Solinas prime expands via the
// generic square-and-multiply; bernstein-yang keeps mod_arith.inverse (safegcd).
// RUN: prime-ir-opt %s -field-to-mod-arith="inverse-algorithm=fermat" -split-input-file | FileCheck %s -check-prefix=FERMAT
// RUN: prime-ir-opt %s -field-to-mod-arith="inverse-algorithm=bernstein-yang" -split-input-file | FileCheck %s -check-prefix=BY

// Goldilocks: p = 2^64 - 2^32 + 1 (a=64, b=32). Chain: 64 squarings + 10 muls.
!G = !field.pf<18446744069414584321 : i64>
// FERMAT-LABEL: @goldilocks_inverse
// FERMAT-NOT: mod_arith.inverse
// FERMAT-COUNT-10: mod_arith.mul
// FERMAT-NOT: mod_arith.mul
// BY-LABEL: @goldilocks_inverse
// BY: mod_arith.inverse
func.func @goldilocks_inverse(%a: !G) -> !G {
  %r = field.inverse %a : !G
  return %r : !G
}

// -----

// BabyBear: p = 2^31 - 2^27 + 1 (a=31, b=27). Chain: 54 squarings + 8 muls.
!BB = !field.pf<2013265921 : i32>
// FERMAT-LABEL: @babybear_inverse
// FERMAT-NOT: mod_arith.inverse
// FERMAT-COUNT-8: mod_arith.mul
// FERMAT-NOT: mod_arith.mul
// BY-LABEL: @babybear_inverse
// BY: mod_arith.inverse
func.func @babybear_inverse(%a: !BB) -> !BB {
  %r = field.inverse %a : !BB
  return %r : !BB
}

// -----

// KoalaBear: p = 2^31 - 2^24 + 1 (a=31, b=24). Chain: 48 squarings + 6 muls.
!KB = !field.pf<2130706433 : i32>
// FERMAT-LABEL: @koalabear_inverse
// FERMAT-NOT: mod_arith.inverse
// FERMAT-COUNT-6: mod_arith.mul
// FERMAT-NOT: mod_arith.mul
// BY-LABEL: @koalabear_inverse
// BY: mod_arith.inverse
func.func @koalabear_inverse(%a: !KB) -> !KB {
  %r = field.inverse %a : !KB
  return %r : !KB
}

// -----

// Mersenne31: p = 2^31 - 1 = 2^31 - 2^1 + 1 (a=31, b=1). Same general chain,
// no per-field code — 30 squarings + 8 muls.
!M31 = !field.pf<2147483647 : i32>
// FERMAT-LABEL: @mersenne31_inverse
// FERMAT-NOT: mod_arith.inverse
// FERMAT-COUNT-8: mod_arith.mul
// FERMAT-NOT: mod_arith.mul
// BY-LABEL: @mersenne31_inverse
// BY: mod_arith.inverse
func.func @mersenne31_inverse(%a: !M31) -> !M31 {
  %r = field.inverse %a : !M31
  return %r : !M31
}

// -----

// A non-Solinas prime (2^64 - 59): p-2 is not "all ones except one zero", so no
// short chain — fermat falls back to the generic square-and-multiply (one mul
// per set bit, far more than the chains above), no safegcd op.
!P = !field.pf<18446744073709551557 : i64>
// FERMAT-LABEL: @non_solinas_inverse
// FERMAT-NOT: mod_arith.inverse
// FERMAT-COUNT-16: mod_arith.mul
func.func @non_solinas_inverse(%a: !P) -> !P {
  %r = field.inverse %a : !P
  return %r : !P
}

// -----

// A prime whose p-2 is a "random" bit pattern (~30 runs of ones): a run-chain
// would cost about as many muls as a generic square-and-multiply, so `fermat`
// falls back to the generic pow (one mul per set bit — far more than 9).
!Q = !field.pf<12297829382473034303 : i64>
// FERMAT-LABEL: @generic_pow_inverse
// FERMAT-NOT: mod_arith.inverse
// FERMAT-COUNT-30: mod_arith.mul
func.func @generic_pow_inverse(%a: !Q) -> !Q {
  %r = field.inverse %a : !Q
  return %r : !Q
}

// -----

// Cubic extension over Goldilocks: the norm-trick's internal base-field inverse
// also uses the chain under fermat (no safegcd op), and safegcd under
// bernstein-yang — the option reaches the extension base inverse too.
!G = !field.pf<18446744069414584321 : i64>
!C = !field.ef<3x!G, 7:i64>
// FERMAT-LABEL: @goldilocks_cubic_inverse
// FERMAT-NOT: mod_arith.inverse
// BY-LABEL: @goldilocks_cubic_inverse
// BY: mod_arith.inverse
func.func @goldilocks_cubic_inverse(%a: !C) -> !C {
  %r = field.inverse %a : !C
  return %r : !C
}
