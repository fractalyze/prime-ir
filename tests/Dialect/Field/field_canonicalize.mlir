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

// CHECK-LABEL: @test_negate_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_negate_fold() -> !PF17 {
  // CHECK: %[[C:.*]] = field.constant 12 : [[T]]
  // -5 mod 17 = 12
  %0 = field.constant 5 : !PF17
  %1 = field.negate %0 : !PF17
  // CHECK-NOT: field.negate
  // CHECK: return %[[C]] : [[T]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_negate_tensor_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_negate_tensor_fold() -> tensor<2x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<[14, 12]> : [[T]]
  // -[3, 5] mod 17 = [14, 12]
  %0 = field.constant dense<[3, 5]> : tensor<2x!PF17>
  %1 = field.negate %0 : tensor<2x!PF17>
  // CHECK-NOT: field.negate
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!PF17>
}

// CHECK-LABEL: @test_double_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_double_fold() -> !PF17 {
  // CHECK: %[[C:.*]] = field.constant 14 : [[T]]
  // 2 * 7 mod 17 = 14
  %0 = field.constant 7 : !PF17
  %1 = field.double %0 : !PF17
  // CHECK-NOT: field.double
  // CHECK: return %[[C]] : [[T]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_double_tensor_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_double_tensor_fold() -> tensor<2x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<[6, 16]> : [[T]]
  // 2 * [3, 8] mod 17 = [6, 16]
  %0 = field.constant dense<[3, 8]> : tensor<2x!PF17>
  %1 = field.double %0 : tensor<2x!PF17>
  // CHECK-NOT: field.double
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!PF17>
}

// CHECK-LABEL: @test_square_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_square_fold() -> !PF17 {
  // CHECK: %[[C:.*]] = field.constant 9 : [[T]]
  // 3² mod 17 = 9
  %0 = field.constant 3 : !PF17
  %1 = field.square %0 : !PF17
  // CHECK-NOT: field.square
  // CHECK: return %[[C]] : [[T]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_square_tensor_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_square_tensor_fold() -> tensor<2x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<[4, 15]> : [[T]]
  // [2, 7]² mod 17 = [4, 49 mod 17] = [4, 15]
  %0 = field.constant dense<[2, 7]> : tensor<2x!PF17>
  %1 = field.square %0 : tensor<2x!PF17>
  // CHECK-NOT: field.square
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!PF17>
}

// CHECK-LABEL: @test_inverse_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_inverse_fold() -> !PF17 {
  // CHECK: %[[C:.*]] = field.constant 6 : [[T]]
  // 3⁻¹ mod 17: 3 * 6 = 18 = 1 mod 17, so 3⁻¹ = 6
  %0 = field.constant 3 : !PF17
  %1 = field.inverse %0 : !PF17
  // CHECK-NOT: field.inverse
  // CHECK: return %[[C]] : [[T]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_inverse_tensor_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_inverse_tensor_fold() -> tensor<2x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<[9, 6]> : [[T]]
  // 2⁻¹ mod 17: 2 * 9 = 18 = 1 mod 17, so 2⁻¹ = 9
  // 3⁻¹ mod 17: 3 * 6 = 18 = 1 mod 17, so 3⁻¹ = 6
  %0 = field.constant dense<[2, 3]> : tensor<2x!PF17>
  %1 = field.inverse %0 : tensor<2x!PF17>
  // CHECK-NOT: field.inverse
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!PF17>
}

// CHECK-LABEL: @test_inverse_splat_tensor_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_inverse_splat_tensor_fold() -> tensor<2x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<9> : [[T]]
  // 2⁻¹ mod 17: 2 * 9 = 18 = 1 mod 17, so 2⁻¹ = 9
  %0 = field.constant dense<2> : tensor<2x!PF17>
  %1 = field.inverse %0 : tensor<2x!PF17>
  // CHECK-NOT: field.inverse
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!PF17>
}

// CHECK-LABEL: @test_add_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_add_fold() -> !PF17 {
  // CHECK: %[[C:.*]] = field.constant 5 : [[T]]
  // 2 + 3 mod 17 = 5
  %0 = field.constant 2 : !PF17
  %1 = field.constant 3 : !PF17
  %2 = field.add %0, %1 : !PF17
  // CHECK-NOT: field.add
  // CHECK: return %[[C]] : [[T]]
  return %2 : !PF17
}

// CHECK-LABEL: @test_add_tensor_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_add_tensor_fold() -> tensor<2x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<[6, 5]> : [[T]]
  // [2, 3] + [4, 2] mod 17 = [6, 5]
  %0 = field.constant dense<[2, 3]> : tensor<2x!PF17>
  %1 = field.constant dense<[4, 2]> : tensor<2x!PF17>
  %2 = field.add %0, %1 : tensor<2x!PF17>
  // CHECK-NOT: field.add
  // CHECK: return %[[C]] : [[T]]
  return %2 : tensor<2x!PF17>
}

// CHECK-LABEL: @test_add_splat_tensor_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_add_splat_tensor_fold() -> tensor<2x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<6> : [[T]]
  // [2] + [4] mod 17 = [6]
  %0 = field.constant dense<2> : tensor<2x!PF17>
  %1 = field.constant dense<4> : tensor<2x!PF17>
  %2 = field.add %0, %1 : tensor<2x!PF17>
  // CHECK-NOT: field.add
  // CHECK: return %[[C]] : [[T]]
  return %2 : tensor<2x!PF17>
}

// CHECK-LABEL: @test_sub_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_sub_fold() -> !PF17 {
  // CHECK: %[[C:.*]] = field.constant 16 : [[T]]
  // 2 - 3 mod 17 = 16
  %0 = field.constant 2 : !PF17
  %1 = field.constant 3 : !PF17
  %2 = field.sub %0, %1 : !PF17
  // CHECK-NOT: field.sub
  // CHECK: return %[[C]] : [[T]]
  return %2 : !PF17
}

// CHECK-LABEL: @test_sub_tensor_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_sub_tensor_fold() -> tensor<2x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<[15, 1]> : [[T]]
  // [2, 3] - [4, 2] mod 17 = [15, 1]
  %0 = field.constant dense<[2, 3]> : tensor<2x!PF17>
  %1 = field.constant dense<[4, 2]> : tensor<2x!PF17>
  %2 = field.sub %0, %1 : tensor<2x!PF17>
  // CHECK-NOT: field.sub
  // CHECK: return %[[C]] : [[T]]
  return %2 : tensor<2x!PF17>
}

// CHECK-LABEL: @test_sub_splat_tensor_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_sub_splat_tensor_fold() -> tensor<2x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<15> : [[T]]
  // [2] - [4] mod 17 = [15]
  %0 = field.constant dense<2> : tensor<2x!PF17>
  %1 = field.constant dense<4> : tensor<2x!PF17>
  %2 = field.sub %0, %1 : tensor<2x!PF17>
  // CHECK-NOT: field.sub
  // CHECK: return %[[C]] : [[T]]
  return %2 : tensor<2x!PF17>
}

// CHECK-LABEL: @test_double_splat_tensor_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_double_splat_tensor_fold() -> tensor<2x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<4> : [[T]]
  // 2 * 2 mod 17 = 4
  %0 = field.constant dense<2> : tensor<2x!PF17>
  %1 = field.double %0 : tensor<2x!PF17>
  // CHECK-NOT: field.double
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!PF17>
}

// CHECK-LABEL: @test_square_splat_tensor_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_square_splat_tensor_fold() -> tensor<2x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<4> : [[T]]
  // 2² mod 17 = 4
  %0 = field.constant dense<2> : tensor<2x!PF17>
  %1 = field.square %0 : tensor<2x!PF17>
  // CHECK-NOT: field.square
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!PF17>
}

// CHECK-LABEL: @test_mul_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_mul_fold() -> !PF17 {
  // CHECK: %[[C:.*]] = field.constant 6 : [[T]]
  // 2 * 3 mod 17 = 6
  %0 = field.constant 2 : !PF17
  %1 = field.constant 3 : !PF17
  %2 = field.mul %0, %1 : !PF17
  // CHECK-NOT: field.mul
  // CHECK: return %[[C]] : [[T]]
  return %2 : !PF17
}

// CHECK-LABEL: @test_mul_tensor_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_mul_tensor_fold() -> tensor<2x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<[8, 6]> : [[T]]
  // [2, 3] * [4, 2] mod 17 = [8, 6]
  %0 = field.constant dense<[2, 3]> : tensor<2x!PF17>
  %1 = field.constant dense<[4, 2]> : tensor<2x!PF17>
  %2 = field.mul %0, %1 : tensor<2x!PF17>
  // CHECK-NOT: field.mul
  // CHECK: return %[[C]] : [[T]]
  return %2 : tensor<2x!PF17>
}

// CHECK-LABEL: @test_mul_splat_tensor_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_mul_splat_tensor_fold() -> tensor<2x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<8> : [[T]]
  // [2] * [4] mod 17 = [8]
  %0 = field.constant dense<2> : tensor<2x!PF17>
  %1 = field.constant dense<4> : tensor<2x!PF17>
  %2 = field.mul %0, %1 : tensor<2x!PF17>
  // CHECK-NOT: field.mul
  // CHECK: return %[[C]] : [[T]]
  return %2 : tensor<2x!PF17>
}

// CHECK-LABEL: @test_negate_splat_tensor_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_negate_splat_tensor_fold() -> tensor<2x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<15> : [[T]]
  // -2 mod 17 = 15
  %0 = field.constant dense<2> : tensor<2x!PF17>
  %1 = field.negate %0 : tensor<2x!PF17>
  // CHECK-NOT: field.negate
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!PF17>
}

// CHECK-LABEL: @test_negate_tensor_zero_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_negate_tensor_zero_fold() -> tensor<2x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<0> : [[T]]
  // -0 mod 17 = 0
  %0 = field.constant dense<0> : tensor<2x!PF17>
  %1 = field.negate %0 : tensor<2x!PF17>
  // CHECK-NOT: field.negate
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!PF17>
}

//===----------------------------------------------------------------------===//
// AddOp patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_add_zero_is_self
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_add_zero_is_self(%arg0: !PF17) -> !PF17 {
  %0 = field.constant 0 : !PF17
  %1 = field.add %arg0, %0 : !PF17
  // CHECK: return %[[ARG0]] : [[T]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_add_tensor_zero_is_self
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_add_tensor_zero_is_self(%arg0: tensor<2x!PF17>) -> tensor<2x!PF17> {
  %0 = field.constant dense<0> : tensor<2x!PF17>
  %1 = field.add %arg0, %0 : tensor<2x!PF17>
  // CHECK: return %[[ARG0]] : [[T]]
  return %1 : tensor<2x!PF17>
}

// CHECK-LABEL: @test_add_constant_twice
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_add_constant_twice(%arg0: !PF17) -> !PF17 {
  %c1 = field.constant 1 : !PF17
  %c2 = field.constant 2 : !PF17
  %0 = field.add %arg0, %c1 : !PF17
  %1 = field.add %0, %c2 : !PF17
  // CHECK: %[[C3:.*]] = field.constant 3 : [[T]]
  // CHECK: %[[RES:.*]] = field.add %[[ARG0]], %[[C3]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_add_constant_to_sub_lhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_add_constant_to_sub_lhs(%arg0: !PF17) -> !PF17 {
  %c1 = field.constant 1 : !PF17
  %c2 = field.constant 2 : !PF17
  %0 = field.sub %c1, %arg0 : !PF17
  %1 = field.add %0, %c2 : !PF17
  // CHECK: %[[C3:.*]] = field.constant 3 : [[T]]
  // CHECK: %[[RES:.*]] = field.sub %[[C3]], %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_add_constant_to_sub_rhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_add_constant_to_sub_rhs(%arg0: !PF17) -> !PF17 {
  %c1 = field.constant 1 : !PF17
  %c2 = field.constant 2 : !PF17
  %0 = field.sub %arg0, %c1 : !PF17
  %1 = field.add %0, %c2 : !PF17
  // CHECK: %[[C1:.*]] = field.constant 1 : [[T]]
  // CHECK: %[[RES:.*]] = field.add %[[ARG0]], %[[C1]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !PF17
}

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

// CHECK-LABEL: @test_sub_zero_is_self
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_sub_zero_is_self(%arg0: !PF17) -> !PF17 {
  %0 = field.constant 0 : !PF17
  %1 = field.sub %arg0, %0 : !PF17
  // CHECK: return %[[ARG0]] : [[T]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_sub_tensor_zero_is_self
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_sub_tensor_zero_is_self(%arg0: tensor<2x!PF17>) -> tensor<2x!PF17> {
  %0 = field.constant dense<0> : tensor<2x!PF17>
  %1 = field.sub %arg0, %0 : tensor<2x!PF17>
  // CHECK: return %[[ARG0]] : [[T]]
  return %1 : tensor<2x!PF17>
}

// CHECK-LABEL: @test_sub_constant_from_add
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_sub_constant_from_add(%arg0: !PF17) -> !PF17 {
  %c1 = field.constant 1 : !PF17
  %c2 = field.constant 2 : !PF17
  %0 = field.add %arg0, %c1 : !PF17
  %1 = field.sub %0, %c2 : !PF17
  // CHECK: %[[C16:.*]] = field.constant 16 : [[T]]
  // CHECK: %[[RES:.*]] = field.add %[[ARG0]], %[[C16]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_sub_constant_twice_lhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_sub_constant_twice_lhs(%arg0: !PF17) -> !PF17 {
  %c1 = field.constant 1 : !PF17
  %c2 = field.constant 2 : !PF17
  %0 = field.sub %c1, %arg0 : !PF17
  %1 = field.sub %0, %c2 : !PF17
  // CHECK: %[[C16:.*]] = field.constant 16 : [[T]]
  // CHECK: %[[RES:.*]] = field.sub %[[C16]], %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_sub_constant_twice_rhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_sub_constant_twice_rhs(%arg0: !PF17) -> !PF17 {
  %c1 = field.constant 1 : !PF17
  %c2 = field.constant 2 : !PF17
  %0 = field.sub %arg0, %c1 : !PF17
  %1 = field.sub %0, %c2 : !PF17
  // CHECK: %[[C3:.*]] = field.constant 3 : [[T]]
  // CHECK: %[[RES:.*]] = field.sub %[[ARG0]], %[[C3]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_sub_add_from_constant
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_sub_add_from_constant(%arg0: !PF17) -> !PF17 {
  %c1 = field.constant 1 : !PF17
  %c2 = field.constant 2 : !PF17
  %0 = field.add %arg0, %c1 : !PF17
  %1 = field.sub %c2, %0 : !PF17
  // CHECK: %[[C1:.*]] = field.constant 1 : [[T]]
  // CHECK: %[[RES:.*]] = field.sub %[[C1]], %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_sub_sub_from_constant_lhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_sub_sub_from_constant_lhs(%arg0: !PF17) -> !PF17 {
  %c1 = field.constant 1 : !PF17
  %c2 = field.constant 2 : !PF17
  %0 = field.sub %c1, %arg0 : !PF17
  %1 = field.sub %c2, %0 : !PF17
  // CHECK: %[[C1:.*]] = field.constant 1 : [[T]]
  // CHECK: %[[RES:.*]] = field.add %[[ARG0]], %[[C1]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_sub_sub_from_constant_rhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_sub_sub_from_constant_rhs(%arg0: !PF17) -> !PF17 {
  %c1 = field.constant 1 : !PF17
  %c2 = field.constant 2 : !PF17
  %0 = field.sub %arg0, %c1 : !PF17
  %1 = field.sub %c2, %0 : !PF17
  // CHECK: %[[C3:.*]] = field.constant 3 : [[T]]
  // CHECK: %[[RES:.*]] = field.sub %[[C3]], %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_sub_self_is_zero
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_sub_self_is_zero(%arg0: !PF17) -> !PF17 {
  %0 = field.sub %arg0, %arg0 : !PF17
  // CHECK: %[[C:.*]] = field.constant 0 : [[T]]
  // CHECK: return %[[C]] : [[T]]
  return %0 : !PF17
}

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

// CHECK-LABEL: @test_mul_by_zero_is_zero
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_by_zero_is_zero(%arg0: !PF17) -> !PF17 {
  %0 = field.constant 0 : !PF17
  %1 = field.mul %arg0, %0 : !PF17
  // CHECK: %[[C:.*]] = field.constant 0 : [[T]]
  // CHECK: return %[[C]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_mul_tensor_by_zero_is_zero
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_tensor_by_zero_is_zero(%arg0: tensor<2x!PF17>) -> tensor<2x!PF17> {
  %0 = field.constant dense<0> : tensor<2x!PF17>
  %1 = field.mul %arg0, %0 : tensor<2x!PF17>
  // CHECK: %[[C:.*]] = field.constant dense<0> : [[T]]
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!PF17>
}

// CHECK-LABEL: @test_mul_by_one_is_self
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_by_one_is_self(%arg0: !PF17) -> !PF17 {
  %0 = field.constant 1 : !PF17
  %1 = field.mul %arg0, %0 : !PF17
  // CHECK: return %[[ARG0]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_mul_tensor_by_one_is_self
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_tensor_by_one_is_self(%arg0: tensor<2x!PF17>) -> tensor<2x!PF17> {
  %0 = field.constant dense<1> : tensor<2x!PF17>
  %1 = field.mul %arg0, %0 : tensor<2x!PF17>
  // CHECK: return %[[ARG0]] : [[T]]
  return %1 : tensor<2x!PF17>
}

// CHECK-LABEL: @test_mul_by_two_is_double
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_by_two_is_double(%arg0: !PF17) -> !PF17 {
  %0 = field.constant 2 : !PF17
  %1 = field.mul %arg0, %0 : !PF17
  // CHECK: %[[RES:.*]] = field.double %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_mul_self_is_square
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_self_is_square(%arg0: !PF17) -> !PF17 {
  %mul = field.mul %arg0, %arg0 : !PF17
  // CHECK: %[[RES:.*]] = field.square %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %mul : !PF17
}

// CHECK-LABEL: @test_mul_by_neg_one
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_by_neg_one(%arg0: !PF17) -> !PF17 {
  %0 = field.constant 16 : !PF17 // -1 mod 17
  %1 = field.mul %arg0, %0 : !PF17
  // CHECK: %[[RES:.*]] = field.negate %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_mul_by_neg_two
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_by_neg_two(%arg0: !PF17) -> !PF17 {
  %0 = field.constant 15 : !PF17 // -2 mod 17
  %1 = field.mul %arg0, %0 : !PF17
  // CHECK: %[[DOUBLE:.*]] = field.double %[[ARG0]] : [[T]]
  // CHECK: %[[RES:.*]] = field.negate %[[DOUBLE]] : [[T]]
  // CHECK: return %[[RES]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_mul_by_neg_three
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_by_neg_three(%arg0: !PF17) -> !PF17 {
  %0 = field.constant 14 : !PF17 // -3 mod 17
  %1 = field.mul %arg0, %0 : !PF17
  // CHECK: %[[DOUBLE:.*]] = field.double %[[ARG0]] : [[T]]
  // CHECK: %[[ADD:.*]] = field.add %[[DOUBLE]], %[[ARG0]] : [[T]]
  // CHECK: %[[RES:.*]] = field.negate %[[ADD]] : [[T]]
  // CHECK: return %[[RES]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_mul_by_neg_four
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_by_neg_four(%arg0: !PF17) -> !PF17 {
  %0 = field.constant 13 : !PF17 // -4 mod 17
  %1 = field.mul %arg0, %0 : !PF17
  // CHECK: %[[DOUBLE1:.*]] = field.double %[[ARG0]] : [[T]]
  // CHECK: %[[DOUBLE2:.*]] = field.double %[[DOUBLE1]] : [[T]]
  // CHECK: %[[RES:.*]] = field.negate %[[DOUBLE2]] : [[T]]
  // CHECK: return %[[RES]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_mul_constant_twice
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_constant_twice(%arg0: !PF17) -> !PF17 {
  %c3 = field.constant 3 : !PF17
  %c4 = field.constant 4 : !PF17
  %0 = field.mul %arg0, %c3 : !PF17
  %1 = field.mul %0, %c4 : !PF17
  // CHECK: %[[C12:.*]] = field.constant 12 : [[T]]
  // CHECK: %[[RES:.*]] = field.mul %[[ARG0]], %[[C12]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_mul_of_mul_by_constant
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_mul_of_mul_by_constant(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  %c3 = field.constant 3 : !PF17
  %c4 = field.constant 4 : !PF17
  %0 = field.mul %arg0, %c3 : !PF17
  %1 = field.mul %arg1, %c4 : !PF17
  %2 = field.mul %0, %1 : !PF17
  // CHECK: %[[C12:.*]] = field.constant 12 : [[T]]
  // CHECK: %[[PROD:.*]] = field.mul %[[ARG0]], %[[ARG1]] : [[T]]
  // CHECK: %[[RES:.*]] = field.mul %[[PROD]], %[[C12]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %2 : !PF17
}

// CHECK-LABEL: @test_mul_add_distribute_constant
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_add_distribute_constant(%arg0: !PF17) -> !PF17 {
  %c3 = field.constant 3 : !PF17
  %c4 = field.constant 4 : !PF17
  %0 = field.add %arg0, %c3 : !PF17
  %1 = field.mul %0, %c4 : !PF17
  // CHECK: %[[C12:.*]] = field.constant 12 : [[T]]
  // CHECK: %[[C4:.*]] = field.constant 4 : [[T]]
  // CHECK: %[[PROD:.*]] = field.mul %[[ARG0]], %[[C4]] : [[T]]
  // CHECK: %[[RES:.*]] = field.add %[[PROD]], %[[C12]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_mul_sub_distribute_constant_rhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_sub_distribute_constant_rhs(%arg0: !PF17) -> !PF17 {
  %c3 = field.constant 3 : !PF17
  %c4 = field.constant 4 : !PF17
  %0 = field.sub %arg0, %c3 : !PF17
  %1 = field.mul %0, %c4 : !PF17
  // CHECK: %[[C12:.*]] = field.constant 12 : [[T]]
  // CHECK: %[[C4:.*]] = field.constant 4 : [[T]]
  // CHECK: %[[PROD:.*]] = field.mul %[[ARG0]], %[[C4]] : [[T]]
  // CHECK: %[[RES:.*]] = field.sub %[[PROD]], %[[C12]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !PF17
}

// CHECK-LABEL: @test_mul_sub_distribute_constant_lhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_sub_distribute_constant_lhs(%arg0: !PF17) -> !PF17 {
  %c3 = field.constant 3 : !PF17
  %c4 = field.constant 4 : !PF17
  %0 = field.sub %c3, %arg0 : !PF17
  %1 = field.mul %0, %c4 : !PF17
  // CHECK: %[[C12:.*]] = field.constant 12 : [[T]]
  // CHECK: %[[C4:.*]] = field.constant 4 : [[T]]
  // CHECK: %[[PROD:.*]] = field.mul %[[ARG0]], %[[C4]] : [[T]]
  // CHECK: %[[RES:.*]] = field.sub %[[C12]], %[[PROD]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %1 : !PF17
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

// CHECK-LABEL: @test_tensor_splat_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_splat_fold() -> tensor<4x!PF17> {
  %0 = field.constant 9 : !PF17
  %1 = tensor.splat %0 : tensor<4x!PF17>
  // CHECK: %[[C:.*]] = field.constant dense<9> : [[T]]
  // CHECK-NOT: tensor.splat
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<4x!PF17>
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

// CHECK-LABEL: @test_splat_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_splat_fold() -> vector<2x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<1> : [[T]]
  // CHECK-NOT: vector.splat
  // CHECK: return %[[C]] : [[T]]
  %0 = field.constant 1 : !PF17
  %1 = vector.splat %0 : vector<2x!PF17>
  return %1 : vector<2x!PF17>
}

// CHECK-LABEL: @test_shuffle_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_shuffle_fold() -> vector<4x!PF17> {
  %v1 = field.constant dense<[10, 3, 13, 16]> : vector<4x!PF17>
  %v2 = field.constant dense<[15, 8]> : vector<2x!PF17>
  %shuffled = vector.shuffle %v1, %v2 [0, 4, 1, 5] : vector<4x!PF17>, vector<2x!PF17>
  // CHECK: %[[C:.*]] = field.constant dense<[10, 15, 3, 8]> : [[T]]
  // CHECK-NOT: vector.shuffle
  // CHECK: return %[[C]] : [[T]]
  return %shuffled : vector<4x!PF17>
}

// CHECK-LABEL: @test_extract_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_extract_fold() -> vector<2x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<[3, 4]> : [[T]]
  // CHECK-NOT: vector.extract
  // CHECK: return %[[C]] : [[T]]
  %0 = field.constant dense<[[1, 2],[3, 4]]> : vector<2x2x!PF17>
  %1 = vector.extract %0[1] : vector<2x!PF17> from vector<2x2x!PF17>
  return %1 : vector<2x!PF17>
}

// CHECK-LABEL: @test_extract_fold_from_splat
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_extract_fold_from_splat() -> vector<2x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<1> : [[T]]
  // CHECK-NOT: vector.extract
  // CHECK: return %[[C]] : [[T]]
  %0 = field.constant dense<1> : vector<2x2x!PF17>
  %1 = vector.extract %0[1] : vector<2x!PF17> from vector<2x2x!PF17>
  return %1 : vector<2x!PF17>
}

// CHECK-LABEL: @test_broadcast_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_broadcast_fold() -> vector<2x2x!PF17> {
  %0 = field.constant 5 : !PF17
  %1 = vector.broadcast %0 : !PF17 to vector<2x2x!PF17>
  // CHECK: %[[C:.*]] = field.constant dense<5> : [[T]]
  // CHECK-NOT: vector.broadcast
  // CHECK: return %[[C]] : [[T]]
  return %1 : vector<2x2x!PF17>
}

// CHECK-LABEL: @test_broadcast_splat_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_broadcast_splat_fold() -> vector<2x2x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<5> : [[T]]
  // CHECK-NOT: vector.broadcast
  // CHECK: return %[[C]] : [[T]]
  %1 = field.constant dense<5> : vector<2x!PF17>
  %2 = vector.broadcast %1 : vector<2x!PF17> to vector<2x2x!PF17>
  return %2 : vector<2x2x!PF17>
}

// CHECK-LABEL: @test_insert_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_insert_fold() -> vector<2x!PF17> {
  %0 = field.constant dense<[1, 3]> : vector<2x!PF17>
  %1 = field.constant 2 : !PF17
  %result = vector.insert %1, %0[1] : !PF17 into vector<2x!PF17>
  // CHECK: %[[C:.*]] = field.constant dense<[1, 2]> : [[T]]
  // CHECK-NOT: vector.insert
  // CHECK: return %[[C]] : [[T]]
  return %result : vector<2x!PF17>
}

// CHECK-LABEL: @test_insert_strided_slice_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_insert_strided_slice_fold() -> vector<4x!PF17> {
  // CHECK: %[[C:.*]] = field.constant dense<[3, 1, 2, 3]> : [[T]]
  // CHECK-NOT: vector.insert_strided_slice
  // CHECK: return %[[C]] : [[T]]
  %source = field.constant dense<[1, 2]> : vector<2x!PF17>
  %dest = field.constant dense<3> : vector<4x!PF17>
  %result = vector.insert_strided_slice %source, %dest
            {offsets = [1], strides = [1]}
            : vector<2x!PF17> into vector<4x!PF17>
  return %result : vector<4x!PF17>
}


// CHECK-LABEL: @test_extract_strided_slice_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_extract_strided_slice_fold() -> vector<2x!PF17> {
  %0 = field.constant dense<[1, 2, 3, 4]> : vector<4x!PF17>
  %slice = vector.extract_strided_slice %0 {offsets = [0], sizes = [2], strides = [1]} : vector<4x!PF17> to vector<2x!PF17>
  // CHECK: %[[C:.*]] = field.constant dense<[1, 2]> : [[T]]
  // CHECK-NOT: vector.extract_strided_slice
  // CHECK: return %[[C]] : [[T]]
  return %slice : vector<2x!PF17>
}

// CHECK-LABEL: @test_splat_extract_strided_slice_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_splat_extract_strided_slice_fold() -> vector<2x!PF17> {
  %0 = field.constant dense<5> : vector<4x!PF17>
  %slice = vector.extract_strided_slice %0 {offsets = [0], sizes = [2], strides = [1]} : vector<4x!PF17> to vector<2x!PF17>
  // CHECK: %[[C:.*]] = field.constant dense<5> : [[T]]
  // CHECK-NOT: vector.extract_strided_slice
  // CHECK: return %[[C]] : [[T]]
  return %slice : vector<2x!PF17>
}
