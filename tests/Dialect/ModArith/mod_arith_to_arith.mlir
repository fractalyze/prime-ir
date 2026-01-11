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

// CHECK-LABEL: @test_lower_inverse_tensor
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_inverse_tensor(%input : !Zpv) -> !Zpv {
  // CHECK-NOT: mod_arith.inverse
  %res = mod_arith.inverse %input : !Zpv
  return %res : !Zpv
}

// CHECK-LABEL: @test_lower_add
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_add(%lhs : !Zp, %rhs : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.add
  // CHECK: %[[ADD:.*]] = arith.addi %[[LHS]], %[[RHS]] overflow<nsw, nuw> : [[T]]
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[T]]
  // CHECK: %[[IFLT:.*]] = arith.cmpi ult, %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: %[[REM:.*]] = arith.select %[[IFLT]], %[[ADD]], %[[SUB]] : [[T]]
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
  // CHECK: %[[IFLT:.*]] = arith.cmpi ult, %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: %[[REM:.*]] = arith.select %[[IFLT]], %[[ADD]], %[[SUB]] : tensor<4xi1>, [[T]]
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
  // CHECK: %[[IFLT:.*]] = arith.cmpi ult, %[[SHL]], %[[CMOD]] : [[T]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[SHL]], %[[CMOD]] : [[T]]
  // CHECK: %[[REM:.*]] = arith.select %[[IFLT]], %[[SHL]], %[[SUB]] : [[T]]
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
  // CHECK: %[[IFLT:.*]] = arith.cmpi ult, %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[SUB]], %[[CMOD]] : [[T]]
  // CHECK: %[[SELECT:.*]] = arith.select %[[IFLT]], %[[ADD]], %[[SUB]] : [[T]]
  // CHECK: return %[[SELECT]] : [[T]]
  %res = mod_arith.sub %lhs, %rhs : !Zp
  return %res : !Zp
}

// CHECK-LABEL: @test_lower_sub_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_sub_vec(%lhs : !Zpv, %rhs : !Zpv) -> !Zpv {
  // CHECK-NOT: mod_arith.sub
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[T]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: %[[IFLT:.*]] = arith.cmpi ult, %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[SUB]], %[[CMOD]] : [[T]]
  // CHECK: %[[SELECT:.*]] = arith.select %[[IFLT]], %[[ADD]], %[[SUB]] : tensor<4xi1>, [[T]]
  // CHECK: return %[[SELECT]] : [[T]]
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
