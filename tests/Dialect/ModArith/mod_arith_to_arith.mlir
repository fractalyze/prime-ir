// Copyright 2025 The PrimeIR Authors.
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

// RUN: prime-ir-opt -mod-arith-to-arith -split-input-file %s | FileCheck %s -enable-var-scope

!Zp = !mod_arith.int<65537 : i32>
!Zpv = tensor<4x!Zp>

!M3 = !mod_arith.int<7 : i32>

// CHECK-LABEL: @test_lower_constant
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_lower_constant() -> !Zp {
  // CHECK-NOT: mod_arith.constant
  // CHECK: %[[CVAL:.*]] = arith.constant 5 : [[T]]
  // CHECK: return %[[CVAL]] : [[T]]
  %res = mod_arith.constant 5:  !Zp
  return %res: !Zp
}

// CHECK-LABEL: @test_lower_constant_vec
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_lower_constant_vec() -> !Zpv {
  // CHECK-NOT: mod_arith.constant
  // CHECK: %[[CVAL:.*]] = arith.constant dense<[5, 10, 15, 20]> : [[T]]
  // CHECK: return %[[CVAL]] : [[T]]
  %res = mod_arith.constant dense<[5, 10, 15, 20]> :  !Zpv
  return %res: !Zpv
}

// CHECK-LABEL: @test_lower_negate
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_negate(%lhs : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.negate
  %res = mod_arith.negate %lhs: !Zp
  return %res : !Zp
}

// CHECK-LABEL: @test_lower_negate_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_negate_vec(%lhs : !Zpv) -> !Zpv {
  // CHECK-NOT: mod_arith.negate
  %res = mod_arith.negate %lhs: !Zpv
  return %res : !Zpv
}

// CHECK-LABEL: @test_lower_bitcast
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_bitcast(%lhs : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.bitcast
  // CHECK: return %[[LHS]] : [[T]]
  %res = mod_arith.bitcast %lhs: !Zp -> i32
  %res2 = mod_arith.bitcast %res: i32 -> !Zp
  return %res2 : !Zp
}

// CHECK-LABEL: @test_lower_bitcast_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_bitcast_vec(%lhs : !Zpv) -> !Zpv {
  // CHECK-NOT: mod_arith.bitcast
  // CHECK: return %[[LHS]] : [[T]]
  %res = mod_arith.bitcast %lhs: !Zpv -> tensor<4xi32>
  %res2 = mod_arith.bitcast %res: tensor<4xi32> -> !Zpv
  return %res2 : !Zpv
}

// CHECK-LABEL: @test_lower_inverse
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_inverse(%lhs : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.inverse
  %res = mod_arith.inverse %lhs : !Zp
  return %res : !Zp
}

// CHECK-LABEL: @test_lower_add
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_add(%lhs : !Zp, %rhs : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.add
  // CHECK: %[[ADD:.*]] = arith.addi %[[LHS]], %[[RHS]] overflow<nsw, nuw> : [[T]]
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[T]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: %[[REM:.*]] = arith.minui %[[SUB]], %[[ADD]] : [[T]]
  // CHECK: return %[[REM]] : [[T]]
  %res = mod_arith.add %lhs, %rhs : !Zp
  return %res : !Zp
}

// CHECK-LABEL: @test_lower_add_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_add_vec(%lhs : !Zpv, %rhs : !Zpv) -> !Zpv {
  // CHECK-NOT: mod_arith.add
  // CHECK: %[[ADD:.*]] = arith.addi %[[LHS]], %[[RHS]] overflow<nsw, nuw> : [[T]]
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[T]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: %[[REM:.*]] = arith.minui %[[SUB]], %[[ADD]] : [[T]]
  // CHECK: return %[[REM]] : [[T]]
  %res = mod_arith.add %lhs, %rhs : !Zpv
  return %res : !Zpv
}

// CHECK-LABEL: @test_lower_double
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_double(%input : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.double
  // CHECK: %[[ONE:.*]] = arith.constant 1 : [[T]]
  // CHECK: %[[SHL:.*]] = arith.shli %[[INPUT]], %[[ONE]] overflow<nsw, nuw> : [[T]]
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[T]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[SHL]], %[[CMOD]] : [[T]]
  // CHECK: %[[REM:.*]] = arith.minui %[[SUB]], %[[SHL]] : [[T]]
  // CHECK: return %[[REM]] : [[T]]
  %res = mod_arith.double %input : !Zp
  return %res : !Zp
}

// CHECK-LABEL: @test_lower_sub
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_sub(%lhs : !Zp, %rhs : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.sub
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[T]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[SUB]], %[[CMOD]] : [[T]]
  // CHECK: %[[MIN:.*]] = arith.minui %[[SUB]], %[[ADD]] : [[T]]
  // CHECK: return %[[MIN]] : [[T]]
  %res = mod_arith.sub %lhs, %rhs : !Zp
  return %res : !Zp
}

// CHECK-LABEL: @test_lower_sub_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_sub_vec(%lhs : !Zpv, %rhs : !Zpv) -> !Zpv {
  // CHECK-NOT: mod_arith.sub
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[T]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[SUB]], %[[CMOD]] : [[T]]
  // CHECK: %[[MIN:.*]] = arith.minui %[[SUB]], %[[ADD]] : [[T]]
  // CHECK: return %[[MIN]] : [[T]]
  %res = mod_arith.sub %lhs, %rhs : !Zpv
  return %res : !Zpv
}

// CHECK-LABEL: @test_lower_mul
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_mul(%lhs : !Zp, %rhs : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.mul
  %res = mod_arith.mul %lhs, %rhs : !Zp
  return %res : !Zp
}

// CHECK-LABEL: @test_lower_mul_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_mul_vec(%lhs : !Zpv, %rhs : !Zpv) -> !Zpv {
  // CHECK-NOT: mod_arith.mul
  %res = mod_arith.mul %lhs, %rhs : !Zpv
  return %res : !Zpv
}

// CHECK-LABEL: @test_lower_mul_mersenne
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_mul_mersenne(%lhs : !M3, %rhs : !M3) -> !M3 {
  // CHECK-NOT: mod_arith.mul
  %res = mod_arith.mul %lhs, %rhs : !M3
  return %res : !M3
}

// CHECK-LABEL: @test_lower_cmp
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) {
func.func @test_lower_cmp(%lhs : !Zp) {
  // CHECK: %[[RHS:.*]] = arith.constant 5 : [[T]]
  %rhs = mod_arith.constant 5:  !Zp
  // CHECK-NOT: mod_arith.cmp
  // %[[EQUAL:.*]] = arith.cmpi [[eq:.*]], %[[LHS]], %[[RHS]] : [[i32]]
  // %[[NOTEQUAL:.*]] = arith.cmpi [[ne:.*]], %[[LHS]], %[[RHS]] : [[i32]]
  // %[[LESSTHAN:.*]] = arith.cmpi [[ult:.*]], %[[LHS]], %[[RHS]] : [[i32]]
  // %[[LESSTHANOREQUALS:.*]] = arith.cmpi [[ule:.*]], %[[LHS]], %[[RHS]] : [[i32]]
  // %[[GREATERTHAN:.*]] = arith.cmpi [[ugt:.*]], %[[LHS]], %[[RHS]] : [[i32]]
  // %[[GREATERTHANOREQUALS:.*]] = arith.cmpi [[uge:.*]], %[[LHS]], %[[RHS]] : [[i32]]
  %equal = mod_arith.cmp eq, %lhs, %rhs : !Zp
  %notEqual = mod_arith.cmp ne, %lhs, %rhs : !Zp
  %lessThan = mod_arith.cmp ult, %lhs, %rhs : !Zp
  %lessThanOrEquals = mod_arith.cmp ule, %lhs, %rhs : !Zp
  %greaterThan = mod_arith.cmp ugt, %lhs, %rhs : !Zp
  %greaterThanOrEquals = mod_arith.cmp uge, %lhs, %rhs : !Zp
  return
}

// -----

// Babybear: modWidth = 31, storageWidth = 32
// storageWidth - modWidth = 1, so nsw is NOT safe (only nuw is safe)
!Babybear = !mod_arith.int<2013265921 : i32>

// CHECK-LABEL: @test_add_narrow_headroom
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_add_narrow_headroom(%lhs : !Babybear, %rhs : !Babybear) -> !Babybear {
  // CHECK-NOT: mod_arith.add
  // CHECK: %[[ADD:.*]] = arith.addi %[[LHS]], %[[RHS]] overflow<nuw> : [[T]]
  // CHECK-NOT: nsw
  %res = mod_arith.add %lhs, %rhs : !Babybear
  return %res : !Babybear
}

// CHECK-LABEL: @test_double_narrow_headroom
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_double_narrow_headroom(%input : !Babybear) -> !Babybear {
  // CHECK-NOT: mod_arith.double
  // CHECK: %[[ONE:.*]] = arith.constant 1 : [[T]]
  // CHECK: %[[SHL:.*]] = arith.shli %[[INPUT]], %[[ONE]] overflow<nuw> : [[T]]
  // CHECK-NOT: nsw
  %res = mod_arith.double %input : !Babybear
  return %res : !Babybear
}

// -----

// modWidth = 30, storageWidth = 32
// storageWidth - modWidth = 2, so both nuw and nsw are safe
!Zp30 = !mod_arith.int<1073741789 : i32>

// CHECK-LABEL: @test_add_wide_headroom
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_add_wide_headroom(%lhs : !Zp30, %rhs : !Zp30) -> !Zp30 {
  // CHECK-NOT: mod_arith.add
  // CHECK: %[[ADD:.*]] = arith.addi %[[LHS]], %[[RHS]] overflow<nsw, nuw> : [[T]]
  %res = mod_arith.add %lhs, %rhs : !Zp30
  return %res : !Zp30
}

// CHECK-LABEL: @test_double_wide_headroom
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_double_wide_headroom(%input : !Zp30) -> !Zp30 {
  // CHECK-NOT: mod_arith.double
  // CHECK: %[[ONE:.*]] = arith.constant 1 : [[T]]
  // CHECK: %[[SHL:.*]] = arith.shli %[[INPUT]], %[[ONE]] overflow<nsw, nuw> : [[T]]
  %res = mod_arith.double %input : !Zp30
  return %res : !Zp30
}

// -----

// BN254 Fr (Montgomery): 256-bit modulus, 64-bit limbs → numLimbs = 4.
// Signed multiply optimization must NOT apply to multi-limb types because
// reduceMultiLimb uses unsigned shifts that don't preserve sign information.
// sub → mont_mul must use mului_extended, not mulsi_extended.
!BN254m = !mod_arith.int<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256, true>

// CHECK-LABEL: @test_mont_mul_multi_limb_no_signed_opt
func.func @test_mont_mul_multi_limb_no_signed_opt(%a : !BN254m, %b : !BN254m, %c : !BN254m) -> !BN254m {
  // Both subs produce minui(x-y, x-y+p) which getSignedFormFromCanonical can
  // match. For multi-limb types, ConvertMontMul must NOT use the signed form.
  %lhs = mod_arith.sub %a, %b : !BN254m
  %rhs = mod_arith.sub %b, %c : !BN254m
  // CHECK-NOT: arith.mulsi_extended
  // CHECK: arith.mului_extended
  %res = mod_arith.mont_mul %lhs, %rhs : !BN254m
  return %res : !BN254m
}

// -----

// BabyBear (Montgomery): 32-bit modulus, 32-bit limbs → numLimbs = 1.
// Signed multiply optimization is safe for single-limb types because
// reduceSingleLimb handles signed inputs via isFromSignedMul.
!BBm = !mod_arith.int<2013265921 : i32, true>

// CHECK-LABEL: @test_mont_mul_single_limb_signed_opt
func.func @test_mont_mul_single_limb_signed_opt(%a : !BBm, %b : !BBm, %c : !BBm) -> !BBm {
  // Both subs produce minui(x-y, x-y+p). For single-limb types,
  // getSignedFormFromCanonical should match and use mulsi_extended.
  %lhs = mod_arith.sub %a, %b : !BBm
  %rhs = mod_arith.sub %b, %c : !BBm
  // CHECK: arith.mulsi_extended
  %res = mod_arith.mont_mul %lhs, %rhs : !BBm
  return %res : !BBm
}

// -----

// Goldilocks: p = 2⁶⁴ - 2³² + 1, p > 2⁶³ so getCanonicalDiff must use
// cmpi + select instead of minui (diff + p overflows i64).
!Gp = !mod_arith.int<18446744069414584321 : i64>

// CHECK-LABEL: @test_lower_sub_goldilocks
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_sub_goldilocks(%lhs : !Gp, %rhs : !Gp) -> !Gp {
  // CHECK-NOT: mod_arith.sub
  // CHECK: %[[CMOD:.*]] = arith.constant -4294967295 : [[T]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[SUB]], %[[CMOD]] : [[T]]
  // CHECK: %[[CMP:.*]] = arith.cmpi ult, %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: %[[SEL:.*]] = arith.select %[[CMP]], %[[ADD]], %[[SUB]] : [[T]]
  // CHECK: return %[[SEL]] : [[T]]
  %res = mod_arith.sub %lhs, %rhs : !Gp
  return %res : !Gp
}

// -----

// Goldilocks add is full-width (modWidth == storageWidth == 64), so it carries
// into an addui_extended overflow bit and canonicalizes via the overflow form
// of getCanonicalFromExtended. That folds the `>= p` case with `minui` (no
// `cmpi`/`ori`): subi + minui + select = 3 ALU ops. Byte-identical to the prior
// cmpi + ori + subi + select (4 ops). p = 0xffffffff00000001 prints as
// -4294967295 in signed i64.
!Gadd = !mod_arith.int<18446744069414584321 : i64>
// CHECK-LABEL: @test_lower_add_goldilocks
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_add_goldilocks(%lhs : !Gadd, %rhs : !Gadd) -> !Gadd {
  // CHECK-NOT: mod_arith.add
  // CHECK: %[[SUM:.*]], %[[OVF:.*]] = arith.addui_extended %[[LHS]], %[[RHS]] : [[T]], i1
  // CHECK: %[[CMOD:.*]] = arith.constant -4294967295 : [[T]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[SUM]], %[[CMOD]] : [[T]]
  // CHECK: %[[MIN:.*]] = arith.minui %[[SUB]], %[[SUM]] : [[T]]
  // CHECK: %[[RES:.*]] = arith.select %[[OVF]], %[[SUB]], %[[MIN]] : [[T]]
  // CHECK-NOT: arith.cmpi
  // CHECK: return %[[RES]] : [[T]]
  %res = mod_arith.add %lhs, %rhs : !Gadd
  return %res : !Gadd
}

// -----

// Goldilocks p = 2^64 - 2^32 + 1 takes a 64-bit Solinas reduction path so the
// CPU JIT runtime doesn't need libgcc's `__umodti3` (the generic
// `arith.remui i128, i128` fallback is unsupported on that runtime). The whole
// reduction stays in i64: the product comes from mului_extended and the
// borrow/carry corrections come from addui_extended — no i128.
!Goldilocks = !mod_arith.int<18446744069414584321 : i64>
// The full reducer chain is asserted via SSA capture: product halves from
// mului_extended, the lo - hi_hi borrow correction, the hi_lo·ε term, the
// addui_extended carry correction, and the final canonicalizing subtract.
// CHECK-LABEL: @test_lower_mul_goldilocks
func.func @test_lower_mul_goldilocks(%lhs : !Goldilocks, %rhs : !Goldilocks)
    -> !Goldilocks {
  // CHECK-NOT: mod_arith.mul
  // CHECK-NOT: arith.remui
  // CHECK-NOT: i128
  // CHECK:      %[[LO:.*]], %[[HI:.*]] = arith.mului_extended %{{.*}}, %{{.*}} : i64
  // CHECK:      %[[HIHI:.*]] = arith.shrui %[[HI]], %{{.*}} : i64
  // CHECK:      %[[HILO:.*]] = arith.andi %[[HI]], %{{.*}} : i64
  // CHECK:      %[[BORROW:.*]] = arith.cmpi ult, %[[LO]], %[[HIHI]] : i64
  // CHECK:      %[[T0RAW:.*]] = arith.subi %[[LO]], %[[HIHI]] : i64
  // CHECK:      %[[T0COR:.*]] = arith.subi %[[T0RAW]], %{{.*}} : i64
  // CHECK:      %[[T0:.*]] = arith.select %[[BORROW]], %[[T0COR]], %[[T0RAW]] : i64
  // CHECK:      %[[T1SHL:.*]] = arith.shli %[[HILO]], %{{.*}} : i64
  // CHECK:      %[[T1:.*]] = arith.subi %[[T1SHL]], %[[HILO]] : i64
  // CHECK:      %[[SUM:.*]], %[[CARRY:.*]] = arith.addui_extended %[[T0]], %[[T1]] : i64, i1
  // CHECK:      %[[T2COR:.*]] = arith.addi %[[SUM]], %{{.*}} : i64
  // CHECK:      %[[T2:.*]] = arith.select %[[CARRY]], %[[T2COR]], %[[SUM]] : i64
  // CHECK:      %[[PSUB:.*]] = arith.subi %[[T2]], %{{.*}} : i64
  // CHECK:      %[[RES:.*]] = arith.minui %[[PSUB]], %[[T2]] : i64
  // CHECK:      return %[[RES]] : i64
  %res = mod_arith.mul %lhs, %rhs : !Goldilocks
  return %res : !Goldilocks
}

// Squaring feeds the same reducer (via squareExtended's product halves), so it
// must lower to the identical i64 chain.
// CHECK-LABEL: @test_lower_square_goldilocks
func.func @test_lower_square_goldilocks(%lhs : !Goldilocks) -> !Goldilocks {
  // CHECK-NOT: mod_arith.square
  // CHECK-NOT: arith.remui
  // CHECK-NOT: i128
  // CHECK:      %[[LO:.*]], %[[HI:.*]] = arith.mului_extended %{{.*}}, %{{.*}} : i64
  // CHECK:      %[[HIHI:.*]] = arith.shrui %[[HI]], %{{.*}} : i64
  // CHECK:      %[[HILO:.*]] = arith.andi %[[HI]], %{{.*}} : i64
  // CHECK:      %[[BORROW:.*]] = arith.cmpi ult, %[[LO]], %[[HIHI]] : i64
  // CHECK:      %[[T0RAW:.*]] = arith.subi %[[LO]], %[[HIHI]] : i64
  // CHECK:      %[[T0COR:.*]] = arith.subi %[[T0RAW]], %{{.*}} : i64
  // CHECK:      %[[T0:.*]] = arith.select %[[BORROW]], %[[T0COR]], %[[T0RAW]] : i64
  // CHECK:      %[[T1SHL:.*]] = arith.shli %[[HILO]], %{{.*}} : i64
  // CHECK:      %[[T1:.*]] = arith.subi %[[T1SHL]], %[[HILO]] : i64
  // CHECK:      %[[SUM:.*]], %[[CARRY:.*]] = arith.addui_extended %[[T0]], %[[T1]] : i64, i1
  // CHECK:      %[[T2COR:.*]] = arith.addi %[[SUM]], %{{.*}} : i64
  // CHECK:      %[[T2:.*]] = arith.select %[[CARRY]], %[[T2COR]], %[[SUM]] : i64
  // CHECK:      %[[PSUB:.*]] = arith.subi %[[T2]], %{{.*}} : i64
  // CHECK:      %[[RES:.*]] = arith.minui %[[PSUB]], %[[T2]] : i64
  // CHECK:      return %[[RES]] : i64
  %res = mod_arith.square %lhs : !Goldilocks
  return %res : !Goldilocks
}

// -----

// Regression: a Goldilocks (full-width) sub must never take the lazy
// `lhs + p - rhs` path — lhs + p overflows i64, corrupting the residue. Even
// when the result feeds a multiply (a lazy-accepting consumer), the sub must
// canonicalize via getCanonicalDiff (subi + addi + cmpi + select).
!Gsub = !mod_arith.int<18446744069414584321 : i64>
// CHECK-LABEL: @test_lower_sub_goldilocks_feeds_mul
func.func @test_lower_sub_goldilocks_feeds_mul(%a: !Gsub, %b: !Gsub, %c: !Gsub) -> !Gsub {
  // CHECK:      %[[SUB:.*]] = arith.subi %[[A:.*]], %[[B:.*]] : i64
  // CHECK:      %[[ADD:.*]] = arith.addi %[[SUB]], %{{.*}} : i64
  // CHECK:      %[[CMP:.*]] = arith.cmpi ult, %[[A]], %[[B]] : i64
  // CHECK:      %[[DIFF:.*]] = arith.select %[[CMP]], %[[ADD]], %[[SUB]] : i64
  // CHECK:      arith.mului_extended %[[DIFF]], %{{.*}}
  %s = mod_arith.sub %a, %b : !Gsub
  %r = mod_arith.mul %s, %c : !Gsub
  return %r : !Gsub
}
