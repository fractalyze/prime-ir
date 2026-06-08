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

// Verify that field.mul over a degree-2 extension of the BLS12-377 base field
// (in Montgomery form) decomposes into base-field mont_muls that ride the
// 32-bit-limb CIOS path under limb-bits=32.
//
// BLS12-377 Fp2 = Fp[u] / (u^2 + 5).  The irreducible polynomial coefficient
// stored in the ef type is beta = -5 mod p = p - 5.

// RUN: prime-ir-opt -field-to-mod-arith -mod-arith-to-arith=limb-bits=32 %s | FileCheck %s

!BLS377m = !field.pf<258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177 : i384, true>
// beta = p - 5
!BLS377Fp2m = !field.ef<2x!BLS377m, 258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458172 : i384>

// CHECK-LABEL: @fp2_bls377_mul_gpu_cios
// field.mul decomposes into base mont_muls via Karatsuba; each must take the
// CIOS path (i64 limb products) and must not fall back to the wide-int path.
// CHECK: arith.muli {{.*}} : i64
// CHECK-NOT: arith.mului_extended {{.*}} : i384
// CHECK: return
func.func @fp2_bls377_mul_gpu_cios(%a : !BLS377Fp2m, %b : !BLS377Fp2m) -> !BLS377Fp2m {
  %r = field.mul %a, %b : !BLS377Fp2m
  return %r : !BLS377Fp2m
}
