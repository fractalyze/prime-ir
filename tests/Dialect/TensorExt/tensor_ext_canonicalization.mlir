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

// CHECK-LABEL: @test_constant_folding
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_constant_folding() -> tensor<8xi32> {
  // CHECK: %[[C:.*]] = arith.constant dense<[1, 5, 3, 7, 2, 6, 4, 8]> : [[T]]
  // CHECK-NOT: tensor_ext.bit_reverse
  %const = arith.constant dense<[1, 2, 3, 4, 5, 6, 7, 8]> : tensor<8xi32>
  // CHECK: return %[[C]] : [[T]]
  %const_reversed = tensor_ext.bit_reverse %const into %const {dimension = 0 : i64} : tensor<8xi32>
  return %const_reversed : tensor<8xi32>
}

// Test that bit_reverse(bit_reverse(x)) is simplified to x.
// Both bit_reverse operations are eliminated since they cancel each other out.
// CHECK-LABEL: @test_involution
// CHECK-SAME: (%arg0: [[T:.*]]) -> [[T]] {
func.func @test_involution(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  // CHECK-NOT: tensor_ext.bit_reverse
  // CHECK: return %arg0 : [[T]]
  %reversed = tensor_ext.bit_reverse %arg0 into %arg0 {dimension = 0 : i64} : tensor<8xi32>
  %reversed_reversed = tensor_ext.bit_reverse %reversed into %reversed {dimension = 0 : i64} : tensor<8xi32>
  return %reversed_reversed : tensor<8xi32>
}

!PF = !field.pf<7:i32>
!PFTensor = tensor<8x!PF>

// Test that bit_reverse(mul(bit_reverse(x), y)) -> mul(x, bit_reverse(y))
// This optimization is useful for NTT algorithms.
// CHECK-LABEL: @test_bit_reverse_mul_bit_reverse
// CHECK-SAME: (%[[X:.*]]: [[T:.*]], %[[Y:.*]]: [[T]], %[[TMP:.*]]: [[T]]) -> [[T]] {
func.func @test_bit_reverse_mul_bit_reverse(%x: !PFTensor, %y: !PFTensor, %tmp: !PFTensor) -> !PFTensor {
  // CHECK: %[[BR_Y:.*]] = tensor_ext.bit_reverse %[[Y]] into %[[TMP]]
  // CHECK: %[[MUL:.*]] = field.mul %[[X]], %[[BR_Y]] : [[T]]
  // CHECK: return %[[MUL]] : [[T]]
  %br_x = tensor_ext.bit_reverse %x into %tmp {dimension = 0 : i64} : !PFTensor
  %mul = field.mul %br_x, %y : !PFTensor
  %result = tensor_ext.bit_reverse %mul into %tmp {dimension = 0 : i64} : !PFTensor
  return %result : !PFTensor
}
