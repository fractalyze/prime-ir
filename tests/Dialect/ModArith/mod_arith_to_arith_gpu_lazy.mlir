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

// Lazy-reduction contract under target=gpu, multi-limb CIOS path.
//
// BLS12-377 (p is 377-bit in i384, spare bit clear): setLazyRedcRange accepts
// it because 2p < 2^384. The CIOS path applies: multi-limb, sign bit clear.
//
// Invariant under lazy+gpu vs eager+gpu for a chained mont_mul:
//   eager: 2 arith.minui — one after each CIOS REDC to canonicalise [0,2p)->[0,p)
//   lazy:  1 arith.minui — the intermediate result feeds the second multiply
//          without the final conditional subtraction; only the boundary
//          reduction at return is emitted.
//
// Each canonical CIOS REDC tail emits exactly one arith.subi + arith.minui
// pair (getCanonicalFromExtended(., 2)).  Lazy elides the pair for non-final
// results, so the count delta is 1 per elided intermediate.

// RUN: prime-ir-opt '-mod-arith-to-arith=target=gpu lazy-reduction=true' -split-input-file %s | FileCheck %s -enable-var-scope --check-prefix=LAZY
// RUN: prime-ir-opt -mod-arith-to-arith=target=gpu -split-input-file %s | FileCheck %s -enable-var-scope --check-prefix=EAGER

!Fpm = !mod_arith.int<258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177 : i384, true>

// LAZY-LABEL: @gpu_cios_lazy_mont_mul_chain
// CIOS path: 32-bit limb products via i64 multiplies.
// LAZY: arith.muli {{.*}} : i64
// LAZY-NOT: arith.mului_extended {{.*}} : i384
// Intermediate result feeds the second multiply without a canonical reduction:
// exactly one arith.minui total (the boundary reduction at return).
// LAZY-COUNT-1: arith.minui
// LAZY-NOT: arith.minui
// LAZY: return

// EAGER-LABEL: @gpu_cios_lazy_mont_mul_chain
// Eager: two arith.minui, one after each CIOS REDC.
// EAGER-COUNT-2: arith.minui
// EAGER-NOT: arith.minui
// EAGER: return
func.func @gpu_cios_lazy_mont_mul_chain(%a : !Fpm, %b : !Fpm, %c : !Fpm) -> !Fpm {
  %t = mod_arith.mont_mul %a, %b : !Fpm
  %r = mod_arith.mont_mul %t, %c : !Fpm
  return %r : !Fpm
}
