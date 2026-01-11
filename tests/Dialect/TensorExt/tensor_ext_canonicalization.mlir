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
  %const_reversed = tensor_ext.bit_reverse %const into %const : tensor<8xi32>
  return %const_reversed : tensor<8xi32>
}

// CHECK-LABEL: @test_involution
// CHECK-SAME: (%arg0: [[T:.*]]) -> [[T]] {
func.func @test_involution(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  // CHECK-NOT: tensor_ext.bit_reverse
  // CHECK: return %arg0 : [[T]]
  %reversed = tensor_ext.bit_reverse %arg0 into %arg0 : tensor<8xi32>
  %reversed_reversed = tensor_ext.bit_reverse %reversed into %reversed : tensor<8xi32>
  return %reversed_reversed : tensor<8xi32>
}
