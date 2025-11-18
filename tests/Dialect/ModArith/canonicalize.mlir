// Copyright 2025 The ZKIR Authors.
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

// RUN: zkir-opt -canonicalize %s | FileCheck %s

!Zp = !mod_arith.int<37 : i32>
!Zpm = !mod_arith.int<37 : i32, true>

//===----------------------------------------------------------------------===//
// Constant Folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_bitcast_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_bitcast_fold() -> !Zp {
  // CHECK: %[[C:.*]] = mod_arith.constant 3 : [[T]]
  // CHECK: return %[[C]] : [[T]]
  %c1 = arith.constant 1: index
  %0 = arith.constant dense<[2, 3]> : tensor<2xi32>
  %1 = mod_arith.bitcast %0: tensor<2xi32> -> tensor<2x!Zp>
  %2 = tensor.extract %1[%c1] : tensor<2x!Zp>
  return %2 : !Zp
}

// CHECK-LABEL: @test_double_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_double_fold() -> !Zp {
  // CHECK: %[[C:.*]] = mod_arith.constant 4 : [[T]]
  %0 = mod_arith.constant 2 : !Zp
  %1 = mod_arith.double %0 : !Zp
  // CHECK-NOT: mod_arith.double
  // CHECK: return %[[C]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_square_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_square_fold() -> !Zp {
  // CHECK: %[[C:.*]] = mod_arith.constant 4 : [[T]]
  %0 = mod_arith.constant 2 : !Zp
  %1 = mod_arith.square %0 : !Zp
  // CHECK-NOT: mod_arith.square
  // CHECK: return %[[C]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_negate_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_negate_fold() -> !Zp {
  // CHECK: %[[C:.*]] = mod_arith.constant 35 : [[T]]
  %0 = mod_arith.constant 2 : !Zp
  %1 = mod_arith.negate %0 : !Zp
  // CHECK-NOT: mod_arith.negate
  // CHECK: return %[[C]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_from_mont_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_from_mont_fold() -> !Zp {
  // CHECK: %[[C:.*]] = mod_arith.constant 32 : [[T]]
  %0 = mod_arith.constant 2 : !Zpm
  %1 = mod_arith.from_mont %0 : !Zp
  // CHECK-NOT: mod_arith.from_mont
  // CHECK: return %[[C]] : [[T]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_from_mont_tensor_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_from_mont_tensor_fold() -> tensor<4x!Zp> {
  // CHECK: %[[C:.*]] = mod_arith.constant dense<[16, 32, 11, 27]> : [[T]]
  %0 = mod_arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32> : tensor<4x!Zpm>
  %1 = mod_arith.from_mont %0 : tensor<4x!Zp>
  // CHECK-NOT: mod_arith.from_mont
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<4x!Zp>
}


// CHECK-LABEL: @test_to_mont_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_to_mont_fold() -> !Zpm {
  // CHECK: %[[C:.*]] = mod_arith.constant 2 : [[T]]
  %0 = mod_arith.constant 32 : !Zp
  %1 = mod_arith.to_mont %0 : !Zpm
  // CHECK-NOT: mod_arith.from_mont
  // CHECK: return %[[C]] : [[T]]
  return %1 : !Zpm
}

// CHECK-LABEL: @test_to_mont_tensor_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_to_mont_tensor_fold() -> tensor<4x!Zpm> {
  // CHECK: %[[C:.*]] = mod_arith.constant dense<[1, 2, 3, 4]> : [[T]]
  %0 = mod_arith.constant dense<[16, 32, 11, 27]> : tensor<4xi32> : tensor<4x!Zp>
  %1 = mod_arith.to_mont %0 : tensor<4x!Zpm>
  // CHECK-NOT: mod_arith.to_mont
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<4x!Zpm>
}

// CHECK-LABEL: @test_cmp_fold
// CHECK-SAME: () -> i1 {
func.func @test_cmp_fold() -> i1 {
  // CHECK: %[[C:.*]] = arith.constant true
  %0 = mod_arith.constant 2 : !Zp
  %1 = mod_arith.constant 1 : !Zp
  %2 = mod_arith.cmp ugt, %0, %1 : !Zp
  // CHECK-NOT: mod_arith.cmp
  // CHECK: return %[[C]]
  return %2 : i1
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_add_zero_is_self
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_add_zero_is_self(%arg0: !Zp) -> !Zp {
  %0 = mod_arith.constant 0 : !Zp
  %1 = mod_arith.add %arg0, %0 : !Zp
  // CHECK: return %[[ARG0]] : [[T]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_add_constant_twice
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_add_constant_twice(%arg0: !Zp) -> !Zp {
  %c1 = mod_arith.constant 1 : !Zp
  %c2 = mod_arith.constant 2 : !Zp
  %0 = mod_arith.add %arg0, %c1 : !Zp
  %1 = mod_arith.add %0, %c2 : !Zp
  // CHECK: %[[C3:.*]] = mod_arith.constant 3 : [[T]]
  // CHECK: %[[RES:.*]] = mod_arith.add %[[ARG0]], %[[C3]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_add_constant_to_sub_lhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_add_constant_to_sub_lhs(%arg0: !Zp) -> !Zp {
  %c1 = mod_arith.constant 1 : !Zp
  %c2 = mod_arith.constant 2 : !Zp
  %0 = mod_arith.sub %c1, %arg0 : !Zp
  %1 = mod_arith.add %0, %c2 : !Zp
  // CHECK: %[[C3:.*]] = mod_arith.constant 3 : [[T]]
  // CHECK: %[[RES:.*]] = mod_arith.sub %[[C3]], %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_add_constant_to_sub_rhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_add_constant_to_sub_rhs(%arg0: !Zp) -> !Zp {
  %c1 = mod_arith.constant 1 : !Zp
  %c2 = mod_arith.constant 2 : !Zp
  %0 = mod_arith.sub %arg0, %c1 : !Zp
  %1 = mod_arith.add %0, %c2 : !Zp
  // CHECK: %[[C1:.*]] = mod_arith.constant 1 : [[T]]
  // CHECK: %[[RES:.*]] = mod_arith.add %[[ARG0]], %[[C1]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_add_self_is_double
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_add_self_is_double(%arg0: !Zp) -> !Zp {
  %0 = mod_arith.add %arg0, %arg0 : !Zp
  // CHECK: %[[RES:.*]] = mod_arith.double %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %0 : !Zp
}

// CHECK-LABEL: @test_add_after_neg
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_add_after_neg(%arg0: !Zp, %arg1: !Zp) -> !Zp {
  %0 = mod_arith.negate %arg0 : !Zp
  %1 = mod_arith.negate %arg1 : !Zp
  %2 = mod_arith.add %0, %1 : !Zp
  // CHECK: %[[ADD:.*]] = mod_arith.add %[[ARG0]], %[[ARG1]] : [[T]]
  // CHECK: %[[RES:.*]] = mod_arith.negate %[[ADD]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %2 : !Zp
}

// CHECK-LABEL: @test_add_after_sub
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_add_after_sub(%arg0: !Zp, %arg1: !Zp) -> !Zp {
  %0 = mod_arith.sub %arg0, %arg1 : !Zp
  %1 = mod_arith.add %0, %arg1 : !Zp
  // CHECK: return %[[ARG0]] : [[T]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_add_after_neg_lhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_add_after_neg_lhs(%arg0: !Zp, %arg1: !Zp) -> !Zp {
  %neg_arg0 = mod_arith.negate %arg0 : !Zp
  %0 = mod_arith.add %neg_arg0, %arg1 : !Zp
  // CHECK: %[[RES:.*]] = mod_arith.sub %[[ARG1]], %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %0 : !Zp
}

// CHECK-LABEL: @test_add_after_neg_rhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_add_after_neg_rhs(%arg0: !Zp, %arg1: !Zp) -> !Zp {
  %neg_arg1 = mod_arith.negate %arg1 : !Zp
  %0 = mod_arith.add %arg0, %neg_arg1 : !Zp
  // CHECK: %[[RES:.*]] = mod_arith.sub %[[ARG0]], %[[ARG1]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %0 : !Zp
}

// CHECK-LABEL: @test_factor_mul_add
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]], [[ARG2:%.*]]: [[T]]) -> [[T]]
func.func @test_factor_mul_add(%arg0: !Zp, %arg1: !Zp, %arg2: !Zp) -> !Zp {
  %0 = mod_arith.mul %arg0, %arg1 : !Zp
  %1 = mod_arith.mul %arg0, %arg2 : !Zp
  %2 = mod_arith.add %0, %1 : !Zp
  // CHECK: %[[ADD:.*]] = mod_arith.add %[[ARG1]], [[ARG2]] : [[T]]
  // CHECK: %[[RES:.*]] = mod_arith.mul %[[ARG0]], %[[ADD]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %2 : !Zp
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_sub_zero_is_self
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_sub_zero_is_self(%arg0: !Zp) -> !Zp {
  %0 = mod_arith.constant 0 : !Zp
  %1 = mod_arith.sub %arg0, %0 : !Zp
  // CHECK: return %[[ARG0]] : [[T]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_sub_constant_from_add
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_sub_constant_from_add(%arg0: !Zp) -> !Zp {
  %c1 = mod_arith.constant 1 : !Zp
  %c2 = mod_arith.constant 2 : !Zp
  %0 = mod_arith.add %arg0, %c1 : !Zp
  %1 = mod_arith.sub %0, %c2 : !Zp
  // CHECK: %[[C36:.*]] = mod_arith.constant 36 : [[T]]
  // CHECK: %[[RES:.*]] = mod_arith.add %[[ARG0]], %[[C36]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_sub_constant_twice_lhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_sub_constant_twice_lhs(%arg0: !Zp) -> !Zp {
  %c1 = mod_arith.constant 1 : !Zp
  %c2 = mod_arith.constant 2 : !Zp
  %0 = mod_arith.sub %c1, %arg0 : !Zp
  %1 = mod_arith.sub %0, %c2 : !Zp
  // CHECK: %[[C36:.*]] = mod_arith.constant 36 : [[T]]
  // CHECK: %[[RES:.*]] = mod_arith.sub %[[C36]], %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_sub_constant_twice_rhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_sub_constant_twice_rhs(%arg0: !Zp) -> !Zp {
  %c1 = mod_arith.constant 1 : !Zp
  %c2 = mod_arith.constant 2 : !Zp
  %0 = mod_arith.sub %arg0, %c1 : !Zp
  %1 = mod_arith.sub %0, %c2 : !Zp
  // CHECK: %[[C3:.*]] = mod_arith.constant 3 : [[T]]
  // CHECK: %[[RES:.*]] = mod_arith.sub %[[ARG0]], %[[C3]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_sub_add_from_constant
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_sub_add_from_constant(%arg0: !Zp) -> !Zp {
  %c1 = mod_arith.constant 1 : !Zp
  %c2 = mod_arith.constant 2 : !Zp
  %0 = mod_arith.add %arg0, %c1 : !Zp
  %1 = mod_arith.sub %c2, %0 : !Zp
  // CHECK: %[[C1:.*]] = mod_arith.constant 1 : [[T]]
  // CHECK: %[[RES:.*]] = mod_arith.sub %[[C1]], %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_sub_sub_from_constant_lhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_sub_sub_from_constant_lhs(%arg0: !Zp) -> !Zp {
  %c1 = mod_arith.constant 1 : !Zp
  %c2 = mod_arith.constant 2 : !Zp
  %0 = mod_arith.sub %c1, %arg0 : !Zp
  %1 = mod_arith.sub %c2, %0 : !Zp
  // CHECK: %[[C1:.*]] = mod_arith.constant 1 : [[T]]
  // CHECK: %[[RES:.*]] = mod_arith.add %[[ARG0]], %[[C1]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_sub_sub_from_constant_rhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_sub_sub_from_constant_rhs(%arg0: !Zp) -> !Zp {
  %c1 = mod_arith.constant 1 : !Zp
  %c2 = mod_arith.constant 2 : !Zp
  %0 = mod_arith.sub %arg0, %c1 : !Zp
  %1 = mod_arith.sub %c2, %0 : !Zp
  // CHECK: %[[C3:.*]] = mod_arith.constant 3 : [[T]]
  // CHECK: %[[RES:.*]] = mod_arith.sub %[[C3]], %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_sub_self_is_zero
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_sub_self_is_zero(%arg0: !Zp) -> !Zp {
  %0 = mod_arith.sub %arg0, %arg0 : !Zp
  // CHECK: %[[C:.*]] = mod_arith.constant 0 : [[T]]
  // CHECK: return %[[C]] : [[T]]
  return %0 : !Zp
}

// CHECK-LABEL: @test_sub_lhs_after_add
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_lhs_after_add(%arg0: !Zp, %arg1: !Zp) -> !Zp {
  %0 = mod_arith.add %arg0, %arg1 : !Zp
  %1 = mod_arith.sub %0, %arg0 : !Zp
  // CHECK: return %[[ARG1]] : [[T]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_sub_rhs_after_add
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_rhs_after_add(%arg0: !Zp, %arg1: !Zp) -> !Zp {
  %0 = mod_arith.add %arg0, %arg1 : !Zp
  %1 = mod_arith.sub %0, %arg1 : !Zp
  // CHECK: return %[[ARG0]] : [[T]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_sub_lhs_after_sub
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_lhs_after_sub(%arg0: !Zp, %arg1: !Zp) -> !Zp {
  %0 = mod_arith.sub %arg0, %arg1 : !Zp
  %1 = mod_arith.sub %0, %arg0 : !Zp
  // CHECK: %[[RES:.*]] = mod_arith.negate %[[ARG1]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_sub_after_neg_lhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_after_neg_lhs(%arg0: !Zp, %arg1: !Zp) -> !Zp {
  %neg_arg0 = mod_arith.negate %arg0 : !Zp
  %0 = mod_arith.sub %neg_arg0, %arg1 : !Zp
  // CHECK: %[[SUM:.*]] = mod_arith.add %[[ARG0]], %[[ARG1]] : [[T]]
  // CHECK: %[[RES:.*]] = mod_arith.negate %[[SUM]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %0 : !Zp
}

// CHECK-LABEL: @test_sub_after_neg_rhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_after_neg_rhs(%arg0: !Zp, %arg1: !Zp) -> !Zp {
  %neg_arg1 = mod_arith.negate %arg1 : !Zp
  %0 = mod_arith.sub %arg0, %neg_arg1 : !Zp
  // CHECK: %[[RES:.*]] = mod_arith.add %[[ARG0]], %[[ARG1]] : [[T]]
  // CHECK: return %[[RES]]
  return %0 : !Zp
}

// CHECK-LABEL: @test_sub_after_neg_both
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_after_neg_both(%arg0: !Zp, %arg1: !Zp) -> !Zp {
  %neg_arg0 = mod_arith.negate %arg0 : !Zp
  %neg_arg1 = mod_arith.negate %arg1 : !Zp
  %0 = mod_arith.sub %neg_arg0, %neg_arg1 : !Zp
  // CHECK: %[[RES:.*]] = mod_arith.sub %[[ARG1]], %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]]
  return %0 : !Zp
}

// CHECK-LABEL: @test_sub_after_square_both
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_after_square_both(%arg0: !Zp, %arg1: !Zp) -> !Zp {
  %sq0 = mod_arith.square %arg0 : !Zp
  %sq1 = mod_arith.square %arg1 : !Zp
  %0 = mod_arith.sub %sq0, %sq1 : !Zp
  // CHECK: %[[SUB:.*]] = mod_arith.sub %[[ARG0]], %[[ARG1]] : [[T]]
  // CHECK: %[[ADD:.*]] = mod_arith.add %[[ARG0]], %[[ARG1]] : [[T]]
  // CHECK: %[[RES:.*]] = mod_arith.mul %[[SUB]], %[[ADD]] : [[T]]
  // CHECK: return %[[RES]]
  return %0 : !Zp
}

// CHECK-LABEL: @test_factor_mul_sub
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]], [[ARG2:%.*]]: [[T]]) -> [[T]]
func.func @test_factor_mul_sub(%arg0: !Zp, %arg1: !Zp, %arg2: !Zp) -> !Zp {
  %0 = mod_arith.mul %arg0, %arg1 : !Zp
  %1 = mod_arith.mul %arg0, %arg2 : !Zp
  %2 = mod_arith.sub %0, %1 : !Zp
  // CHECK: %[[SUB:.*]] = mod_arith.sub %[[ARG1]], [[ARG2]] : [[T]]
  // CHECK: %[[RES:.*]] = mod_arith.mul %[[ARG0]], %[[SUB]] : [[T]]
  // CHECK: return %[[RES]]
  return %2 : !Zp
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_mul_by_zero_is_zero
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_by_zero_is_zero(%arg0: !Zp) -> !Zp {
  %0 = mod_arith.constant 0 : !Zp
  %1 = mod_arith.mul %arg0, %0 : !Zp
  // CHECK: %[[C:.*]] = mod_arith.constant 0 : [[T]]
  // CHECK: return %[[C]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_mul_by_one_is_self
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_by_one_is_self(%arg0: !Zp) -> !Zp {
  %0 = mod_arith.constant 1 : !Zp
  %1 = mod_arith.mul %arg0, %0 : !Zp
  // CHECK: return %[[ARG0]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_mul_by_two_is_double
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_by_two_is_double(%arg0: !Zp) -> !Zp {
  %0 = mod_arith.constant 2 : !Zp
  %1 = mod_arith.mul %arg0, %0 : !Zp
  // CHECK: %[[RES:.*]] = mod_arith.double %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_mul_self_is_square
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_self_is_square(%arg0: !Zp) -> !Zp {
  %0 = mod_arith.mul %arg0, %arg0 : !Zp
  // CHECK: %[[RES:.*]] = mod_arith.square %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]]
  return %0 : !Zp
}

// CHECK-LABEL: @test_mul_by_neg_one
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_by_neg_one(%arg0: !Zp) -> !Zp {
  %0 = mod_arith.constant 36 : !Zp // -1
  %1 = mod_arith.mul %arg0, %0 : !Zp
  // CHECK: %[[RES:.*]] = mod_arith.negate %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_mul_by_neg_two
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_by_neg_two(%arg0: !Zp) -> !Zp {
  %0 = mod_arith.constant 35 : !Zp // -2
  %1 = mod_arith.mul %arg0, %0 : !Zp
  // CHECK: %[[DOUBLE:.*]] = mod_arith.double %[[ARG0]] : [[T]]
  // CHECK: %[[RES:.*]] = mod_arith.negate %[[DOUBLE]] : [[T]]
  // CHECK: return %[[RES]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_mul_by_neg_three
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_by_neg_three(%arg0: !Zp) -> !Zp {
  %0 = mod_arith.constant 34 : !Zp // -3
  %1 = mod_arith.mul %arg0, %0 : !Zp
  // CHECK: %[[DOUBLE:.*]] = mod_arith.double %[[ARG0]] : [[T]]
  // CHECK: %[[ADD:.*]] = mod_arith.add %[[DOUBLE]], %[[ARG0]] : [[T]]
  // CHECK: %[[RES:.*]] = mod_arith.negate %[[ADD]] : [[T]]
  // CHECK: return %[[RES]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_mul_by_neg_four
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_by_neg_four(%arg0: !Zp) -> !Zp {
  %0 = mod_arith.constant 33 : !Zp // -4
  %1 = mod_arith.mul %arg0, %0 : !Zp
  // CHECK: %[[DOUBLE1:.*]] = mod_arith.double %[[ARG0]] : [[T]]
  // CHECK: %[[DOUBLE2:.*]] = mod_arith.double %[[DOUBLE1]] : [[T]]
  // CHECK: %[[RES:.*]] = mod_arith.negate %[[DOUBLE2]] : [[T]]
  // CHECK: return %[[RES]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_mul_constant_twice
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_constant_twice(%arg0: !Zp) -> !Zp {
  %c3 = mod_arith.constant 3 : !Zp
  %c4 = mod_arith.constant 4 : !Zp
  %0 = mod_arith.mul %arg0, %c3 : !Zp
  %1 = mod_arith.mul %0, %c4 : !Zp
  // CHECK: %[[C12:.*]] = mod_arith.constant 12 : [[T]]
  // CHECK: %[[RES:.*]] = mod_arith.mul %[[ARG0]], %[[C12]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_mul_of_mul_by_constant
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_mul_of_mul_by_constant(%arg0: !Zp, %arg1: !Zp) -> !Zp {
  %c3 = mod_arith.constant 3 : !Zp
  %c4 = mod_arith.constant 4 : !Zp
  %0 = mod_arith.mul %arg0, %c3 : !Zp
  %1 = mod_arith.mul %arg1, %c4 : !Zp
  %2 = mod_arith.mul %0, %1 : !Zp
  // CHECK: %[[C12:.*]] = mod_arith.constant 12 : [[T]]
  // CHECK: %[[PROD:.*]] = mod_arith.mul %[[ARG0]], %[[ARG1]] : [[T]]
  // CHECK: %[[RES:.*]] = mod_arith.mul %[[PROD]], %[[C12]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %2 : !Zp
}

// CHECK-LABEL: @test_mul_add_distribute_constant
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_add_distribute_constant(%arg0: !Zp) -> !Zp {
  %c3 = mod_arith.constant 3 : !Zp
  %c4 = mod_arith.constant 4 : !Zp
  %0 = mod_arith.add %arg0, %c3 : !Zp
  %1 = mod_arith.mul %0, %c4 : !Zp
  // CHECK: %[[C12:.*]] = mod_arith.constant 12 : [[T]]
  // CHECK: %[[C4:.*]] = mod_arith.constant 4 : [[T]]
  // CHECK: %[[PROD:.*]] = mod_arith.mul %[[ARG0]], %[[C4]] : [[T]]
  // CHECK: %[[RES:.*]] = mod_arith.add %[[PROD]], %[[C12]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_mul_sub_distribute_constant_rhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_sub_distribute_constant_rhs(%arg0: !Zp) -> !Zp {
  %c3 = mod_arith.constant 3 : !Zp
  %c4 = mod_arith.constant 4 : !Zp
  %0 = mod_arith.sub %arg0, %c3 : !Zp
  %1 = mod_arith.mul %0, %c4 : !Zp
  // CHECK: %[[C12:.*]] = mod_arith.constant 12 : [[T]]
  // CHECK: %[[C4:.*]] = mod_arith.constant 4 : [[T]]
  // CHECK: %[[PROD:.*]] = mod_arith.mul %[[ARG0]], %[[C4]] : [[T]]
  // CHECK: %[[RES:.*]] = mod_arith.sub %[[PROD]], %[[C12]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !Zp
}

// CHECK-LABEL: @test_mul_sub_distribute_constant_lhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_sub_distribute_constant_lhs(%arg0: !Zp) -> !Zp {
  %c3 = mod_arith.constant 3 : !Zp
  %c4 = mod_arith.constant 4 : !Zp
  %0 = mod_arith.sub %c3, %arg0 : !Zp
  %1 = mod_arith.mul %0, %c4 : !Zp
  // CHECK: %[[C12:.*]] = mod_arith.constant 12 : [[T]]
  // CHECK: %[[C4:.*]] = mod_arith.constant 4 : [[T]]
  // CHECK: %[[PROD:.*]] = mod_arith.mul %[[ARG0]], %[[C4]] : [[T]]
  // CHECK: %[[RES:.*]] = mod_arith.sub %[[C12]], %[[PROD]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !Zp
}
