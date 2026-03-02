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

// RUN: prime-ir-opt '-mod-arith-to-arith=lazy-reduction=true' -split-input-file %s | FileCheck %s --check-prefix=LAZY -enable-var-scope
// RUN: prime-ir-opt -mod-arith-to-arith -split-input-file %s | FileCheck %s --check-prefix=EAGER -enable-var-scope

// p = 65537 (17-bit), i32 storage → storageWidth - modWidth = 15.
// maxLazyBound = floor(2³² / 65537) = 65535.

!Zp = !mod_arith.int<65537 : i32>

// Lazy add skips conditional reduction. Two chained adds produce no
// intermediate cmpi/select; a single boundary reduction at return.

// LAZY-LABEL: @test_lazy_add
// LAZY:           arith.addi {{.*}} overflow<nsw, nuw>
// LAZY-NOT:       arith.cmpi ult
// LAZY:           arith.addi {{.*}} overflow<nsw, nuw>
// LAZY:           arith.cmpi ult
// LAZY:           arith.select
// LAZY:           return

// EAGER-LABEL:    @test_lazy_add
// EAGER:          arith.addi {{.*}} overflow<nsw, nuw>
// EAGER:          arith.cmpi ult
// EAGER:          arith.select
// EAGER:          arith.addi {{.*}} overflow<nsw, nuw>
// EAGER:          arith.cmpi ult
// EAGER:          arith.select
// EAGER:          return
func.func @test_lazy_add(%lhs : !Zp, %rhs : !Zp) -> !Zp {
  %a = mod_arith.add %lhs, %rhs : !Zp
  %b = mod_arith.add %a, %rhs : !Zp
  return %b : !Zp
}

// -----

!Zp = !mod_arith.int<65537 : i32>

// Lazy sub: (lhs + correction) - rhs without conditional reduction.
// Eager sub: subi then cmpi for underflow then conditional addi.

// LAZY-LABEL:     @test_lazy_sub
// LAZY:           arith.addi
// LAZY:           arith.subi
// LAZY-NOT:       arith.select
// LAZY:           arith.cmpi ult
// LAZY:           arith.select
// LAZY:           return

// EAGER-LABEL:    @test_lazy_sub
// EAGER:          arith.subi
// EAGER:          arith.cmpi ult
// EAGER:          arith.addi
// EAGER:          arith.select
// EAGER:          return
func.func @test_lazy_sub(%lhs : !Zp, %rhs : !Zp) -> !Zp {
  %r = mod_arith.sub %lhs, %rhs : !Zp
  return %r : !Zp
}

// -----

!Zp = !mod_arith.int<65537 : i32>

// Chained adds — no intermediate reductions. Lazy produces exactly one
// arith.select (at boundary); eager produces one per add.

// LAZY-LABEL:     @test_chain
// LAZY-COUNT-2:   arith.addi {{.*}} overflow<nsw, nuw>
// LAZY:           arith.select
// LAZY-NOT:       arith.select
// LAZY:           return

// EAGER-LABEL:    @test_chain
// EAGER:          arith.select
// EAGER:          arith.select
// EAGER-NOT:      arith.select
// EAGER:          return
func.func @test_chain(%a : !Zp, %b : !Zp, %c : !Zp) -> !Zp {
  %ab = mod_arith.add %a, %b : !Zp
  %abc = mod_arith.add %ab, %c : !Zp
  return %abc : !Zp
}

// -----

!Zp = !mod_arith.int<65537 : i32>

// Consumer forces reduction — negate. In lazy mode, the add produces a lazy
// value (bound=2). ConvertNegate inserts getCanonicalFromExtended before the
// negate body. Both modes emit cmpi ult (reduction) then cmpi eq (negate).

// LAZY-LABEL:     @test_lazy_then_negate
// LAZY:           arith.addi {{.*}} overflow<nsw, nuw>
// LAZY:           arith.cmpi ult
// LAZY:           arith.select
// LAZY:           arith.cmpi eq
// LAZY:           arith.select
// LAZY:           return

// EAGER-LABEL:    @test_lazy_then_negate
// EAGER:          arith.addi {{.*}} overflow<nsw, nuw>
// EAGER:          arith.cmpi ult
// EAGER:          arith.select
// EAGER:          arith.cmpi eq
// EAGER:          arith.select
// EAGER:          return
func.func @test_lazy_then_negate(%a : !Zp, %b : !Zp) -> !Zp {
  %sum = mod_arith.add %a, %b : !Zp
  %neg = mod_arith.negate %sum : !Zp
  return %neg : !Zp
}

// -----

!Zp = !mod_arith.int<65537 : i32>

// Consumer forces reduction — cmp. The lazy add result is reduced to
// canonical form before the comparison. Both modes produce
// cmpi ult (reduction) followed by cmpi eq (comparison).

// LAZY-LABEL:     @test_lazy_then_cmp
// LAZY:           arith.addi {{.*}} overflow<nsw, nuw>
// LAZY:           arith.cmpi ult
// LAZY:           arith.select
// LAZY:           arith.cmpi eq
// LAZY:           return

// EAGER-LABEL:    @test_lazy_then_cmp
// EAGER:          arith.addi {{.*}} overflow<nsw, nuw>
// EAGER:          arith.cmpi ult
// EAGER:          arith.select
// EAGER:          arith.cmpi eq
// EAGER:          return
func.func @test_lazy_then_cmp(%a : !Zp, %b : !Zp) -> i1 {
  %sum = mod_arith.add %a, %b : !Zp
  %cmp = mod_arith.cmp eq, %sum, %a : !Zp
  return %cmp : i1
}

// -----

!ZpMont = !mod_arith.int<65537 : i32, true>

// Montgomery lazy mont_mul. Lazy REDC (reduceLazy) emits subi then addi
// unconditionally; eager REDC (reduceSingleLimb via getCanonicalDiff) emits
// subi then cmpi ult (underflow check) then addi.

// LAZY-LABEL:     @test_lazy_mont_mul
// LAZY:           arith.mului_extended
// LAZY:           arith.muli
// LAZY:           arith.mului_extended
// LAZY:           arith.subi
// LAZY-NOT:       arith.cmpi
// LAZY:           arith.addi
// LAZY:           arith.cmpi ult
// LAZY:           arith.select
// LAZY:           return

// EAGER-LABEL:    @test_lazy_mont_mul
// EAGER:          arith.mului_extended
// EAGER:          arith.muli
// EAGER:          arith.mului_extended
// EAGER:          arith.subi
// EAGER:          arith.cmpi ult
// EAGER:          arith.addi
// EAGER:          arith.select
// EAGER:          return
func.func @test_lazy_mont_mul(%a : !ZpMont, %b : !ZpMont) -> !ZpMont {
  %r = mod_arith.mont_mul %a, %b : !ZpMont
  return %r : !ZpMont
}

// -----

!ZpMont = !mod_arith.int<65537 : i32, true>

// Montgomery lazy mont_square — same REDC pattern as mont_mul.

// LAZY-LABEL:     @test_lazy_mont_square
// LAZY:           arith.mului_extended
// LAZY:           arith.muli
// LAZY:           arith.mului_extended
// LAZY:           arith.subi
// LAZY-NOT:       arith.cmpi
// LAZY:           arith.addi
// LAZY:           arith.cmpi ult
// LAZY:           arith.select
// LAZY:           return

// EAGER-LABEL:    @test_lazy_mont_square
// EAGER:          arith.mului_extended
// EAGER:          arith.muli
// EAGER:          arith.mului_extended
// EAGER:          arith.subi
// EAGER:          arith.cmpi ult
// EAGER:          arith.addi
// EAGER:          arith.select
// EAGER:          return
func.func @test_lazy_mont_square(%a : !ZpMont) -> !ZpMont {
  %r = mod_arith.mont_square %a : !ZpMont
  return %r : !ZpMont
}

// -----

!ZpMont = !mod_arith.int<65537 : i32, true>

// Chained mont_mul — lazy through chain. Lazy produces exactly one
// arith.select (at boundary); eager produces one per REDC.

// LAZY-LABEL:     @test_mont_mul_chain
// LAZY:           arith.select
// LAZY-NOT:       arith.select
// LAZY:           return

// EAGER-LABEL:    @test_mont_mul_chain
// EAGER:          arith.select
// EAGER:          arith.select
// EAGER-NOT:      arith.select
// EAGER:          return
func.func @test_mont_mul_chain(%a : !ZpMont, %b : !ZpMont) -> !ZpMont {
  %r1 = mod_arith.mont_mul %a, %b : !ZpMont
  %r2 = mod_arith.mont_mul %r1, %b : !ZpMont
  return %r2 : !ZpMont
}

// -----

!Zp = !mod_arith.int<65537 : i32>

// Non-Montgomery single add — verifies non-Montgomery type also gets
// the lazy path. Same output as eager for a single op.

// LAZY-LABEL:     @test_non_mont_add
// LAZY:           arith.addi {{.*}} overflow<nsw, nuw>
// LAZY:           arith.cmpi ult
// LAZY:           arith.select
// LAZY:           return

// EAGER-LABEL:    @test_non_mont_add
// EAGER:          arith.addi {{.*}} overflow<nsw, nuw>
// EAGER:          arith.cmpi ult
// EAGER:          arith.select
// EAGER:          return
func.func @test_non_mont_add(%a : !Zp, %b : !Zp) -> !Zp {
  %r = mod_arith.add %a, %b : !Zp
  return %r : !Zp
}
