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

// RUN: prime-ir-opt --canonicalize %s | FileCheck %s

// Test constant folding for binary field operations

!BF8 = !field.bf<3>  // GF(2^8)

// CHECK-LABEL: @fold_binary_add
func.func @fold_binary_add() -> !BF8 {
  %a = field.constant 5 : !BF8
  %b = field.constant 3 : !BF8
  // CHECK: field.constant 6 : !field.bf<3>
  // 5 XOR 3 = 0b101 XOR 0b011 = 0b110 = 6
  %c = field.add %a, %b : !BF8
  return %c : !BF8
}

// CHECK-LABEL: @fold_binary_sub
func.func @fold_binary_sub() -> !BF8 {
  %a = field.constant 5 : !BF8
  %b = field.constant 3 : !BF8
  // CHECK: field.constant 6 : !field.bf<3>
  // In char 2, sub = add = XOR: 5 XOR 3 = 6
  %c = field.sub %a, %b : !BF8
  return %c : !BF8
}

// CHECK-LABEL: @fold_binary_negate
func.func @fold_binary_negate() -> !BF8 {
  %a = field.constant 42 : !BF8
  // CHECK: field.constant 42 : !field.bf<3>
  // In char 2, -a = a (negation is identity)
  %c = field.negate %a : !BF8
  return %c : !BF8
}

// CHECK-LABEL: @fold_binary_double_is_zero
func.func @fold_binary_double_is_zero() -> !BF8 {
  %a = field.constant 5 : !BF8
  // CHECK: field.constant 0 : !field.bf<3>
  // In char 2, 2a = a + a = a XOR a = 0
  %c = field.double %a : !BF8
  return %c : !BF8
}

// CHECK-LABEL: @fold_binary_square
func.func @fold_binary_square() -> !BF8 {
  %a = field.constant 3 : !BF8
  // CHECK: field.constant 2 : !field.bf<3>
  // In GF(2^8) tower: 3² = 2
  %c = field.square %a : !BF8
  return %c : !BF8
}

// CHECK-LABEL: @fold_binary_mul
func.func @fold_binary_mul() -> !BF8 {
  %a = field.constant 3 : !BF8
  %b = field.constant 5 : !BF8
  // CHECK: field.constant 15 : !field.bf<3>
  // In GF(2^8) tower: 3 * 5 = 15
  %c = field.mul %a, %b : !BF8
  return %c : !BF8
}

// CHECK-LABEL: @fold_binary_inverse
func.func @fold_binary_inverse() -> !BF8 {
  %a = field.constant 1 : !BF8
  // CHECK: field.constant 1 : !field.bf<3>
  // 1⁻¹ = 1 in any field
  %c = field.inverse %a : !BF8
  return %c : !BF8
}

// CHECK-LABEL: @fold_add_identity
func.func @fold_add_identity(%a: !BF8) -> !BF8 {
  %zero = field.constant 0 : !BF8
  // CHECK: return %arg0 : !field.bf<3>
  %c = field.add %a, %zero : !BF8
  return %c : !BF8
}

// CHECK-LABEL: @fold_mul_identity
func.func @fold_mul_identity(%a: !BF8) -> !BF8 {
  %one = field.constant 1 : !BF8
  // CHECK: return %arg0 : !field.bf<3>
  %c = field.mul %a, %one : !BF8
  return %c : !BF8
}

// CHECK-LABEL: @fold_mul_zero
func.func @fold_mul_zero(%a: !BF8) -> !BF8 {
  %zero = field.constant 0 : !BF8
  // CHECK: field.constant 0 : !field.bf<3>
  %c = field.mul %a, %zero : !BF8
  return %c : !BF8
}

//===----------------------------------------------------------------------===//
// Tensor/Vector constant folding tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_tensor_add
func.func @fold_tensor_add() -> tensor<2x!BF8> {
  %a = field.constant dense<[5, 3]> : tensor<2x!BF8>
  %b = field.constant dense<[3, 5]> : tensor<2x!BF8>
  // CHECK: field.constant dense<6> : tensor<2x!field.bf<3>>
  // [5 XOR 3, 3 XOR 5] = [6, 6]
  %c = field.add %a, %b : tensor<2x!BF8>
  return %c : tensor<2x!BF8>
}

// CHECK-LABEL: @fold_tensor_sub
func.func @fold_tensor_sub() -> tensor<2x!BF8> {
  %a = field.constant dense<[5, 3]> : tensor<2x!BF8>
  %b = field.constant dense<[3, 5]> : tensor<2x!BF8>
  // CHECK: field.constant dense<6> : tensor<2x!field.bf<3>>
  // In char 2, sub = add = XOR
  %c = field.sub %a, %b : tensor<2x!BF8>
  return %c : tensor<2x!BF8>
}

// CHECK-LABEL: @fold_tensor_negate
func.func @fold_tensor_negate() -> tensor<2x!BF8> {
  %a = field.constant dense<[42, 7]> : tensor<2x!BF8>
  // CHECK: field.constant dense<[42, 7]> : tensor<2x!field.bf<3>>
  // In char 2, -a = a (negation is identity)
  %c = field.negate %a : tensor<2x!BF8>
  return %c : tensor<2x!BF8>
}

// CHECK-LABEL: @fold_tensor_double
func.func @fold_tensor_double() -> tensor<2x!BF8> {
  %a = field.constant dense<[5, 100]> : tensor<2x!BF8>
  // CHECK: field.constant dense<0> : tensor<2x!field.bf<3>>
  // In char 2, 2a = 0
  %c = field.double %a : tensor<2x!BF8>
  return %c : tensor<2x!BF8>
}

// CHECK-LABEL: @fold_tensor_square
func.func @fold_tensor_square() -> tensor<2x!BF8> {
  %a = field.constant dense<[3, 1]> : tensor<2x!BF8>
  // CHECK: field.constant dense<[2, 1]> : tensor<2x!field.bf<3>>
  // In GF(2^8) tower: 3² = 2, 1² = 1
  %c = field.square %a : tensor<2x!BF8>
  return %c : tensor<2x!BF8>
}

// CHECK-LABEL: @fold_tensor_mul
func.func @fold_tensor_mul() -> tensor<2x!BF8> {
  %a = field.constant dense<[3, 2]> : tensor<2x!BF8>
  %b = field.constant dense<[5, 3]> : tensor<2x!BF8>
  // CHECK: field.constant dense<[15, 1]> : tensor<2x!field.bf<3>>
  // In GF(2^8) tower: 3 * 5 = 15, 2 * 3 = 1
  %c = field.mul %a, %b : tensor<2x!BF8>
  return %c : tensor<2x!BF8>
}

// CHECK-LABEL: @fold_tensor_add_identity
func.func @fold_tensor_add_identity(%a: tensor<2x!BF8>) -> tensor<2x!BF8> {
  %zero = field.constant dense<0> : tensor<2x!BF8>
  // CHECK: return %arg0 : tensor<2x!field.bf<3>>
  %c = field.add %a, %zero : tensor<2x!BF8>
  return %c : tensor<2x!BF8>
}

// CHECK-LABEL: @fold_tensor_mul_identity
func.func @fold_tensor_mul_identity(%a: tensor<2x!BF8>) -> tensor<2x!BF8> {
  %one = field.constant dense<1> : tensor<2x!BF8>
  // CHECK: return %arg0 : tensor<2x!field.bf<3>>
  %c = field.mul %a, %one : tensor<2x!BF8>
  return %c : tensor<2x!BF8>
}

// CHECK-LABEL: @fold_tensor_mul_zero
func.func @fold_tensor_mul_zero(%a: tensor<2x!BF8>) -> tensor<2x!BF8> {
  %zero = field.constant dense<0> : tensor<2x!BF8>
  // CHECK: field.constant dense<0> : tensor<2x!field.bf<3>>
  %c = field.mul %a, %zero : tensor<2x!BF8>
  return %c : tensor<2x!BF8>
}

//===----------------------------------------------------------------------===//
// Vector constant folding tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_vector_add
func.func @fold_vector_add() -> vector<4x!BF8> {
  %a = field.constant dense<[1, 2, 3, 4]> : vector<4x!BF8>
  %b = field.constant dense<[4, 3, 2, 1]> : vector<4x!BF8>
  // CHECK: field.constant dense<[5, 1, 1, 5]> : vector<4x!field.bf<3>>
  // 1^4=5, 2^3=1, 3^2=1, 4^1=5
  %c = field.add %a, %b : vector<4x!BF8>
  return %c : vector<4x!BF8>
}

// CHECK-LABEL: @fold_vector_negate
func.func @fold_vector_negate() -> vector<4x!BF8> {
  %a = field.constant dense<[10, 20, 30, 40]> : vector<4x!BF8>
  // CHECK: field.constant dense<[10, 20, 30, 40]> : vector<4x!field.bf<3>>
  %c = field.negate %a : vector<4x!BF8>
  return %c : vector<4x!BF8>
}

// CHECK-LABEL: @fold_vector_double
func.func @fold_vector_double() -> vector<4x!BF8> {
  %a = field.constant dense<[1, 2, 3, 4]> : vector<4x!BF8>
  // CHECK: field.constant dense<0> : vector<4x!field.bf<3>>
  %c = field.double %a : vector<4x!BF8>
  return %c : vector<4x!BF8>
}
