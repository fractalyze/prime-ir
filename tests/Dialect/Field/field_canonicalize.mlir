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

// RUN: prime-ir-opt -canonicalize %s | FileCheck %s -enable-var-scope

!PF17 = !field.pf<17:i32>

//===----------------------------------------------------------------------===//
// Constant Folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_fold_negate_scalar
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_fold_negate_scalar() -> !PF17 {
  // CHECK: %[[C:.*]] = field.constant 12 : [[T]]
  // -5 mod 17 = 12
  %0 = field.constant 5 : !PF17
  %1 = field.negate %0 : !PF17
  // CHECK-NOT: field.negate
  // CHECK: return %[[C]] : [[T]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_fold_negate_tensor
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_fold_negate_tensor() -> tensor<2x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<[14, 12]> : [[T]]
  // -[3, 5] mod 17 = [14, 12]
  %0 = field.constant dense<[3, 5]> : tensor<2x!PF17>
  %1 = field.negate %0 : tensor<2x!PF17>
  // CHECK-NOT: field.negate
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!PF17>
}

// CHECK-LABEL: @test_fold_double_scalar
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_fold_double_scalar() -> !PF17 {
  // CHECK: %[[C:.*]] = field.constant 14 : [[T]]
  // 2 * 7 mod 17 = 14
  %0 = field.constant 7 : !PF17
  %1 = field.double %0 : !PF17
  // CHECK-NOT: field.double
  // CHECK: return %[[C]] : [[T]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_fold_double_tensor
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_fold_double_tensor() -> tensor<2x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<[6, 16]> : [[T]]
  // 2 * [3, 8] mod 17 = [6, 16]
  %0 = field.constant dense<[3, 8]> : tensor<2x!PF17>
  %1 = field.double %0 : tensor<2x!PF17>
  // CHECK-NOT: field.double
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!PF17>
}

// CHECK-LABEL: @test_fold_square_scalar
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_fold_square_scalar() -> !PF17 {
  // CHECK: %[[C:.*]] = field.constant 9 : [[T]]
  // 3² mod 17 = 9
  %0 = field.constant 3 : !PF17
  %1 = field.square %0 : !PF17
  // CHECK-NOT: field.square
  // CHECK: return %[[C]] : [[T]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_fold_square_tensor
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_fold_square_tensor() -> tensor<2x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<[4, 15]> : [[T]]
  // [2, 7]² mod 17 = [4, 49 mod 17] = [4, 15]
  %0 = field.constant dense<[2, 7]> : tensor<2x!PF17>
  %1 = field.square %0 : tensor<2x!PF17>
  // CHECK-NOT: field.square
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!PF17>
}

// CHECK-LABEL: @test_fold_inverse_scalar
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_fold_inverse_scalar() -> !PF17 {
  // CHECK: %[[C:.*]] = field.constant 6 : [[T]]
  // 3⁻¹ mod 17: 3 * 6 = 18 = 1 mod 17, so 3⁻¹ = 6
  %0 = field.constant 3 : !PF17
  %1 = field.inverse %0 : !PF17
  // CHECK-NOT: field.inverse
  // CHECK: return %[[C]] : [[T]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_fold_inverse_tensor
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_fold_inverse_tensor() -> tensor<2x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<[9, 6]> : [[T]]
  // 2⁻¹ mod 17: 2 * 9 = 18 = 1 mod 17, so 2⁻¹ = 9
  // 3⁻¹ mod 17: 3 * 6 = 18 = 1 mod 17, so 3⁻¹ = 6
  %0 = field.constant dense<[2, 3]> : tensor<2x!PF17>
  %1 = field.inverse %0 : tensor<2x!PF17>
  // CHECK-NOT: field.inverse
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!PF17>
}

//===----------------------------------------------------------------------===//
// AddOp patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_add_self_is_double
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_add_self_is_double(%arg0: !PF17) -> !PF17 {
  %add = field.add %arg0, %arg0 : !PF17
  // CHECK: %[[RES:.*]] = field.double %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %add : !PF17
}

// CHECK-LABEL: @test_add_both_negated
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_add_both_negated(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  %neg0 = field.negate %arg0 : !PF17
  %neg1 = field.negate %arg1 : !PF17
  %add = field.add %neg0, %neg1 : !PF17
  // CHECK: %[[ADD:.*]] = field.add %[[ARG0]], %[[ARG1]] : [[T]]
  // CHECK: %[[RES:.*]] = field.negate %[[ADD]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %add : !PF17
}

// CHECK-LABEL: @test_add_after_sub
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_add_after_sub(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  %sub = field.sub %arg0, %arg1 : !PF17
  %add = field.add %sub, %arg1 : !PF17
  // CHECK: return %[[ARG0]] : [[T]]
  return %add : !PF17
}

// CHECK-LABEL: @test_add_after_neg_lhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_add_after_neg_lhs(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  %neg = field.negate %arg0 : !PF17
  %add = field.add %neg, %arg1 : !PF17
  // CHECK: %[[RES:.*]] = field.sub %[[ARG1]], %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %add : !PF17
}

// CHECK-LABEL: @test_add_after_neg_rhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_add_after_neg_rhs(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  %neg = field.negate %arg1 : !PF17
  %add = field.add %arg0, %neg : !PF17
  // CHECK: %[[RES:.*]] = field.sub %[[ARG0]], %[[ARG1]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %add : !PF17
}

// CHECK-LABEL: @test_factor_mul_add
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]], %[[ARG2:.*]]: [[T]]) -> [[T]]
func.func @test_factor_mul_add(%arg0: !PF17, %arg1: !PF17, %arg2: !PF17) -> !PF17 {
  %mul1 = field.mul %arg0, %arg1 : !PF17
  %mul2 = field.mul %arg0, %arg2 : !PF17
  %add = field.add %mul1, %mul2 : !PF17
  // CHECK: %[[ADD:.*]] = field.add %[[ARG1]], %[[ARG2]] : [[T]]
  // CHECK: %[[RES:.*]] = field.mul %[[ARG0]], %[[ADD]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %add : !PF17
}

//===----------------------------------------------------------------------===//
// SubOp patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_sub_lhs_after_add
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_lhs_after_add(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  %add = field.add %arg0, %arg1 : !PF17
  %sub = field.sub %add, %arg0 : !PF17
  // CHECK: return %[[ARG1]] : [[T]]
  return %sub : !PF17
}

// CHECK-LABEL: @test_sub_rhs_after_add
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_rhs_after_add(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  %add = field.add %arg0, %arg1 : !PF17
  %sub = field.sub %add, %arg1 : !PF17
  // CHECK: return %[[ARG0]] : [[T]]
  return %sub : !PF17
}

// CHECK-LABEL: @test_sub_lhs_after_sub
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_lhs_after_sub(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  %sub1 = field.sub %arg0, %arg1 : !PF17
  %sub2 = field.sub %sub1, %arg0 : !PF17
  // CHECK: %[[RES:.*]] = field.negate %[[ARG1]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %sub2 : !PF17
}

// CHECK-LABEL: @test_sub_after_neg_lhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_after_neg_lhs(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  %neg = field.negate %arg0 : !PF17
  %sub = field.sub %neg, %arg1 : !PF17
  // CHECK: %[[SUM:.*]] = field.add %[[ARG0]], %[[ARG1]] : [[T]]
  // CHECK: %[[RES:.*]] = field.negate %[[SUM]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %sub : !PF17
}

// CHECK-LABEL: @test_sub_after_neg_rhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_after_neg_rhs(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  %neg = field.negate %arg1 : !PF17
  %sub = field.sub %arg0, %neg : !PF17
  // CHECK: %[[RES:.*]] = field.add %[[ARG0]], %[[ARG1]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %sub : !PF17
}

// CHECK-LABEL: @test_sub_both_negated
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_both_negated(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  %neg0 = field.negate %arg0 : !PF17
  %neg1 = field.negate %arg1 : !PF17
  %sub = field.sub %neg0, %neg1 : !PF17
  // CHECK: %[[RES:.*]] = field.sub %[[ARG1]], %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %sub : !PF17
}

// CHECK-LABEL: @test_sub_after_square_both
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_after_square_both(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  %sq0 = field.square %arg0 : !PF17
  %sq1 = field.square %arg1 : !PF17
  %sub = field.sub %sq0, %sq1 : !PF17
  // CHECK: %[[SUB:.*]] = field.sub %[[ARG0]], %[[ARG1]] : [[T]]
  // CHECK: %[[ADD:.*]] = field.add %[[ARG0]], %[[ARG1]] : [[T]]
  // CHECK: %[[RES:.*]] = field.mul %[[SUB]], %[[ADD]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %sub : !PF17
}

// CHECK-LABEL: @test_factor_mul_sub
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]], %[[ARG2:.*]]: [[T]]) -> [[T]]
func.func @test_factor_mul_sub(%arg0: !PF17, %arg1: !PF17, %arg2: !PF17) -> !PF17 {
  %mul1 = field.mul %arg0, %arg1 : !PF17
  %mul2 = field.mul %arg0, %arg2 : !PF17
  %sub = field.sub %mul1, %mul2 : !PF17
  // CHECK: %[[SUB:.*]] = field.sub %[[ARG1]], %[[ARG2]] : [[T]]
  // CHECK: %[[RES:.*]] = field.mul %[[ARG0]], %[[SUB]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %sub : !PF17
}

//===----------------------------------------------------------------------===//
// MulOp patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_mul_self_is_square
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_self_is_square(%arg0: !PF17) -> !PF17 {
  %mul = field.mul %arg0, %arg0 : !PF17
  // CHECK: %[[RES:.*]] = field.square %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %mul : !PF17
}

//===----------------------------------------------------------------------===//
// Tensor operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_tensor_from_elements
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_from_elements() -> tensor<2x!PF17> {
  %0 = field.constant 1 : !PF17
  %1 = field.constant 2 : !PF17
  %2 = tensor.from_elements %0, %1 : tensor<2x!PF17>
  // CHECK: %[[C:.*]] = field.constant dense<[1, 2]> : [[T]]
  // CHECK-NOT: tensor.from_elements
  // CHECK: return %[[C:.*]] : [[T]]
  return %2 : tensor<2x!PF17>
}

// CHECK-LABEL: @test_tensor_extract
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_extract() -> !PF17 {
  // CHECK: %[[C:.*]] = field.constant 3 : [[T]]
  // CHECK-NOT: tensor.extract
  // CHECK: return %[[C]] : [[T]]
  %c1 = arith.constant 1: index
  %0 = field.constant dense<[2, 3]> : tensor<2x!PF17>
  %1 = tensor.extract %0[%c1] : tensor<2x!PF17>
  return %1 : !PF17
}

//===----------------------------------------------------------------------===//
// Vector operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_vector_from_elements
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_vector_from_elements() -> vector<2x!PF17> {
  %0 = field.constant 1 : !PF17
  %1 = field.constant 2 : !PF17
  %2 = vector.from_elements %0, %1 : vector<2x!PF17>
  // CHECK: %[[FROM_ELEMENTS:.*]] = field.constant dense<[1, 2]> : [[T]]
  // CHECK-NOT: vector.from_elements
  // CHECK: return %[[FROM_ELEMENTS:.*]] : [[T]]
  return %2 : vector<2x!PF17>
}

// CHECK-LABEL: @test_vector_extract
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_vector_extract() -> !PF17 {
  // CHECK: %[[C:.*]] = field.constant 3 : [[T]]
  // CHECK-NOT: vector.extract
  // CHECK: return %[[C]] : [[T]]
  %0 = field.constant dense<[2, 3]> : vector<2x!PF17>
  %1 = vector.extract %0[1] : !PF17 from vector<2x!PF17>
  return %1 : !PF17
}
