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

// The `auto` inverse policy: take the specialized Solinas chain (branch-free
// mod_arith.mul/square, no mod_arith.inverse) ONLY for a Solinas prime
// p = 2^a - 2^b + 1 that is <= 64 bits; otherwise lower via safegcd
// (mod_arith.inverse). Explicit `fermat` ignores the width gate and always
// takes Fermat (chain for Solinas, generic pow otherwise), shown for contrast.
// RUN: prime-ir-opt %s -field-to-mod-arith="inverse-algorithm=auto" -split-input-file | FileCheck %s -check-prefix=AUTO
// RUN: prime-ir-opt %s -field-to-mod-arith="inverse-algorithm=fermat" -split-input-file | FileCheck %s -check-prefix=FERMAT

// Narrow Solinas prime (Goldilocks, 64-bit): auto takes the chain.
!G = !field.pf<18446744069414584321 : i64>
// AUTO-LABEL: @narrow_solinas
// AUTO-NOT: mod_arith.inverse
// AUTO: mod_arith.mul
func.func @narrow_solinas(%a: !G) -> !G {
  %r = field.inverse %a : !G
  return %r : !G
}

// -----

// Wide Solinas prime (Mersenne127 = 2^127 - 1, 127-bit): the chain's ~127
// squarings lose to safegcd, so auto uses safegcd — but explicit fermat still
// emits the chain (no safegcd op) for a caller who wants it.
!M127 = !field.pf<170141183460469231731687303715884105727 : i128>
// AUTO-LABEL: @wide_solinas
// AUTO: mod_arith.inverse
// FERMAT-LABEL: @wide_solinas
// FERMAT-NOT: mod_arith.inverse
func.func @wide_solinas(%a: !M127) -> !M127 {
  %r = field.inverse %a : !M127
  return %r : !M127
}

// -----

// Non-Solinas prime (2^64 - 59): no short chain, so auto uses safegcd; explicit
// fermat falls back to the generic square-and-multiply (branch-free, still no
// safegcd op).
!P = !field.pf<18446744073709551557 : i64>
// AUTO-LABEL: @non_solinas
// AUTO: mod_arith.inverse
// FERMAT-LABEL: @non_solinas
// FERMAT-NOT: mod_arith.inverse
// FERMAT: mod_arith.mul
func.func @non_solinas(%a: !P) -> !P {
  %r = field.inverse %a : !P
  return %r : !P
}
