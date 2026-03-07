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
//   Binary reduction for bound=3: subtract 2p then p → 2 selects.
// LAZY:           arith.select
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

// Chained adds — no intermediate reductions. Lazy skips inline reduction
// on both adds; boundary uses binary reduction for bound=3 ([0, 3p)):
// subtract 2p then p → 2 selects. Eager produces one select per add.

// LAZY-LABEL:     @test_chain
// LAZY-COUNT-2:   arith.addi {{.*}} overflow<nsw, nuw>
// LAZY:           arith.select
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

!Zp = !mod_arith.int<65537 : i32, true>

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
func.func @test_lazy_mont_mul(%a : !Zp, %b : !Zp) -> !Zp {
  %r = mod_arith.mont_mul %a, %b : !Zp
  return %r : !Zp
}

// -----

!Zp = !mod_arith.int<65537 : i32, true>

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
func.func @test_lazy_mont_square(%a : !Zp) -> !Zp {
  %r = mod_arith.mont_square %a : !Zp
  return %r : !Zp
}

// -----

!Zp = !mod_arith.int<65537 : i32, true>

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
func.func @test_mont_mul_chain(%a : !Zp, %b : !Zp) -> !Zp {
  %r1 = mod_arith.mont_mul %a, %b : !Zp
  %r2 = mod_arith.mont_mul %r1, %b : !Zp
  return %r2 : !Zp
}

// -----

// Lazy double skips conditional reduction. Double followed by add produces
// no intermediate cmpi/select; a single boundary reduction at return.

!Zp = !mod_arith.int<65537 : i32>

// LAZY-LABEL:     @test_lazy_double
// LAZY:           arith.shli {{.*}} overflow<nsw, nuw>
// LAZY-NOT:       arith.cmpi ult
// LAZY:           arith.addi {{.*}} overflow<nsw, nuw>
//   Binary reduction for bound=3: subtract 2p then p → 2 selects.
// LAZY:           arith.select
// LAZY:           arith.select
// LAZY:           return

// EAGER-LABEL:    @test_lazy_double
// EAGER:          arith.shli {{.*}} overflow<nsw, nuw>
// EAGER:          arith.cmpi ult
// EAGER:          arith.select
// EAGER:          arith.addi {{.*}} overflow<nsw, nuw>
// EAGER:          arith.cmpi ult
// EAGER:          arith.select
// EAGER:          return
func.func @test_lazy_double(%a : !Zp, %b : !Zp) -> !Zp {
  %d = mod_arith.double %a : !Zp
  %sum = mod_arith.add %d, %b : !Zp
  return %sum : !Zp
}

// -----

// BabyBear: p = 2013265921, i32. getMaxBound() = floor((2³² - 1) / p) = 2.
// When two lazy mont_mul results (bound=2 each) feed into another mont_mul,
// lhsBound * rhsBound = 4 > 2. The REDC precondition T < p * 2^w is violated,
// so inputs must be pre-reduced before the widening multiply.

!Zp = !mod_arith.int<2013265921 : i32, true>

// LAZY-LABEL:     @test_redc_precondition_guard
//   First two mont_muls: lazy REDC (subi + addi, no cmpi/select inside REDC).
// LAZY:           arith.mului_extended
// LAZY:           arith.subi
// LAZY-NOT:       arith.cmpi
// LAZY:           arith.addi
// LAZY:           arith.mului_extended
// LAZY:           arith.subi
// LAZY-NOT:       arith.cmpi
// LAZY:           arith.addi
//   Third mont_mul: both inputs are lazy (bound=2), but reducing only one
//   suffices since (p - 1) * (2p - 1) < p * 2^w for BabyBear.
// LAZY:           arith.cmpi ult
// LAZY:           arith.select
// LAZY-NOT:       arith.cmpi ult
// LAZY:           arith.mului_extended

// EAGER-LABEL:    @test_redc_precondition_guard
// EAGER-COUNT-3:  arith.select
// EAGER-NOT:      arith.select
// EAGER:          return
func.func @test_redc_precondition_guard(%a : !Zp, %b : !Zp, %c : !Zp) -> !Zp {
  %r1 = mod_arith.mont_mul %a, %b : !Zp
  %r2 = mod_arith.mont_mul %a, %c : !Zp
  %r3 = mod_arith.mont_mul %r1, %r2 : !Zp
  return %r3 : !Zp
}

// -----

// BabyBear add pre-reduce: add(%a, %b) has bound=2, then add(bound=2, %c) has
// lhsBound + rhsBound = 3 > maxBound(2). Pre-reduce inserts cmpi + select
// before the second add. Both lazy and eager produce the same output here
// because the narrow headroom forces reduction anyway.

!Zp = !mod_arith.int<2013265921 : i32>

// LAZY-LABEL:     @test_add_pre_reduce
// LAZY:           arith.addi {{.*}} overflow<nuw>
// LAZY:           arith.cmpi ult
// LAZY:           arith.select
// LAZY:           arith.addi {{.*}} overflow<nuw>
// LAZY:           arith.cmpi ult
// LAZY:           arith.select
// LAZY:           return

// EAGER-LABEL:    @test_add_pre_reduce
// EAGER:          arith.addi {{.*}} overflow<nuw>
// EAGER:          arith.cmpi ult
// EAGER:          arith.select
// EAGER:          arith.addi {{.*}} overflow<nuw>
// EAGER:          arith.cmpi ult
// EAGER:          arith.select
// EAGER:          return
func.func @test_add_pre_reduce(%a : !Zp, %b : !Zp, %c : !Zp) -> !Zp {
  %ab = mod_arith.add %a, %b : !Zp
  %abc = mod_arith.add %ab, %c : !Zp
  return %abc : !Zp
}

// -----

// BabyBear sub pre-reduce: add(%a, %b) has bound=2, then sub(bound=2, %c)
// has lhsBound + rhsBound = 3 > maxBound(2). Pre-reduce inserts cmpi + select
// on the lazy lhs before the lazy sub computation.

!Zp = !mod_arith.int<2013265921 : i32>

// LAZY-LABEL:     @test_sub_pre_reduce
//   First add: lazy (no inline reduction).
// LAZY:           arith.addi {{.*}} overflow<nuw>
//   Pre-reduce the lazy add result before sub.
// LAZY:           arith.cmpi ult
// LAZY:           arith.select
//   Lazy sub: addi(lhs, correction) then subi.
// LAZY:           arith.addi
// LAZY:           arith.subi
//   Boundary reduction at return.
// LAZY:           arith.cmpi ult
// LAZY:           arith.select
// LAZY:           return

// EAGER-LABEL:    @test_sub_pre_reduce
// EAGER:          arith.addi {{.*}} overflow<nuw>
// EAGER:          arith.cmpi ult
// EAGER:          arith.select
// EAGER:          arith.subi
// EAGER:          arith.cmpi ult
// EAGER:          arith.addi
// EAGER:          arith.select
// EAGER:          return
func.func @test_sub_pre_reduce(%a : !Zp, %b : !Zp, %c : !Zp) -> !Zp {
  %ab = mod_arith.add %a, %b : !Zp
  %r = mod_arith.sub %ab, %c : !Zp
  return %r : !Zp
}

// -----

// BabyBear double pre-reduce: add(%a, %b) has bound=2, then double(bound=2)
// has 2 * inputBound = 4 > maxBound(2). Pre-reduce inserts cmpi + select
// before shli.

!Zp = !mod_arith.int<2013265921 : i32>

// LAZY-LABEL:     @test_double_pre_reduce
//   First add: lazy (no inline reduction).
// LAZY:           arith.addi {{.*}} overflow<nuw>
//   Pre-reduce the lazy add result before double.
// LAZY:           arith.cmpi ult
// LAZY:           arith.select
//   Double: shli.
// LAZY:           arith.shli {{.*}} overflow<nuw>
//   Boundary reduction at return.
// LAZY:           arith.cmpi ult
// LAZY:           arith.select
// LAZY:           return

// EAGER-LABEL:    @test_double_pre_reduce
// EAGER:          arith.addi {{.*}} overflow<nuw>
// EAGER:          arith.cmpi ult
// EAGER:          arith.select
// EAGER:          arith.shli {{.*}} overflow<nuw>
// EAGER:          arith.cmpi ult
// EAGER:          arith.select
// EAGER:          return
func.func @test_double_pre_reduce(%a : !Zp, %b : !Zp) -> !Zp {
  %ab = mod_arith.add %a, %b : !Zp
  %r = mod_arith.double %ab : !Zp
  return %r : !Zp
}

// -----

// BabyBear mont_square pre-reduce: mont_mul(%a, %b) has bound=2 (lazy REDC),
// then mont_square(bound=2) has inputBound² = 4 > maxBound(2). Pre-reduce
// inserts cmpi + select before the squaring multiply.

!Zp = !mod_arith.int<2013265921 : i32, true>

// LAZY-LABEL:     @test_mont_square_pre_reduce
//   First mont_mul: lazy REDC (subi + addi, no cmpi inside REDC).
// LAZY:           arith.mului_extended
// LAZY:           arith.subi
// LAZY-NOT:       arith.cmpi
// LAZY:           arith.addi
//   Pre-reduce before square: cmpi + select.
// LAZY:           arith.cmpi ult
// LAZY:           arith.select
//   Square: mului_extended (on reduced input).
// LAZY:           arith.mului_extended

// EAGER-LABEL:    @test_mont_square_pre_reduce
// EAGER-COUNT-2:  arith.select
// EAGER-NOT:      arith.select
// EAGER:          return
func.func @test_mont_square_pre_reduce(%a : !Zp, %b : !Zp) -> !Zp {
  %r1 = mod_arith.mont_mul %a, %b : !Zp
  %r2 = mod_arith.mont_square %r1 : !Zp
  return %r2 : !Zp
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

// -----

// BabyBear: p = 2013265921, i32, Montgomery. maxBound = 2.
// add(a, 1) has kp = 2, but exact umax = (p - 1) + (R mod p) = 2281701374.
// mont_mul(add(a,1), add(b,1)): kp product 2 * 2 = 4 > maxBound(2), but
// umax product 2281701374² < p * 2³², so REDC is safe without pre-reduction.

!Zp = !mod_arith.int<2013265921 : i32, true>

// LAZY-LABEL:     @test_add_const_then_mont_mul
// LAZY:           arith.addi
// LAZY:           arith.addi
// LAZY-NOT:       arith.cmpi ult
// LAZY-NOT:       arith.select
// LAZY:           arith.mului_extended

// EAGER-LABEL:    @test_add_const_then_mont_mul
// EAGER:          arith.addi
// EAGER:          arith.cmpi ult
// EAGER:          arith.select
// EAGER:          arith.addi
// EAGER:          arith.cmpi ult
// EAGER:          arith.select
// EAGER:          arith.mului_extended
func.func @test_add_const_then_mont_mul(%a : !Zp, %b : !Zp) -> !Zp {
  %one = mod_arith.constant 1 : !Zp
  %a_plus_1 = mod_arith.add %a, %one : !Zp
  %b_plus_1 = mod_arith.add %b, %one : !Zp
  %r = mod_arith.mont_mul %a_plus_1, %b_plus_1 : !Zp
  return %r : !Zp
}

// -----

// BabyBear: (a + b) * (c + d). umax of each add is 2p - 2 = 4026531840.
// Product (2p - 2)² > p * 2³², so pre-reduction IS needed.

!Zp = !mod_arith.int<2013265921 : i32, true>

// LAZY-LABEL:     @test_add_then_mont_mul_needs_reduce
// LAZY:           arith.addi
// LAZY:           arith.addi
//   Only one operand needs pre-reduction since (p - 1) * (2p - 2) < p * 2^w.
// LAZY:           arith.cmpi ult
// LAZY:           arith.select
// LAZY-NOT:       arith.cmpi ult
// LAZY:           arith.mului_extended

// EAGER-LABEL:    @test_add_then_mont_mul_needs_reduce
// EAGER:          arith.addi
// EAGER:          arith.cmpi ult
// EAGER:          arith.select
// EAGER:          arith.addi
// EAGER:          arith.cmpi ult
// EAGER:          arith.select
// EAGER:          arith.mului_extended
func.func @test_add_then_mont_mul_needs_reduce(%a : !Zp, %b : !Zp, %c : !Zp, %d : !Zp) -> !Zp {
  %ab = mod_arith.add %a, %b : !Zp
  %cd = mod_arith.add %c, %d : !Zp
  %r = mod_arith.mont_mul %ab, %cd : !Zp
  return %r : !Zp
}
