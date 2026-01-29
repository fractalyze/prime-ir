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

// RUN: prime-ir-opt -fold-field-linalg-contraction %s | FileCheck %s
// RUN: prime-ir-opt -fold-field-linalg-contraction="max-unroll-size=2" %s | FileCheck %s --check-prefix=CHECK-SMALL

!PF = !field.pf<97:i32>

// -----
// Test 1: Basic matvec - should fold constant matrix into scalar operations

// CHECK-LABEL: @test_matvec_basic
// CHECK-NOT: linalg.matvec
// CHECK-DAG: tensor.extract
// CHECK-DAG: field.mul
// CHECK-DAG: field.add
// CHECK: tensor.from_elements
func.func @test_matvec_basic(%vec: tensor<3x!PF>) -> tensor<2x!PF> {
  %matrix = field.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3x!PF>
  %init = bufferization.alloc_tensor() : tensor<2x!PF>
  %result = linalg.matvec ins(%matrix, %vec : tensor<2x3x!PF>, tensor<3x!PF>)
                          outs(%init : tensor<2x!PF>) -> tensor<2x!PF>
  return %result : tensor<2x!PF>
}

// -----
// Test 2: Basic dot - should fold constant vector into scalar operations

// CHECK-LABEL: @test_dot_basic
// CHECK-NOT: linalg.dot
// CHECK-DAG: tensor.extract
// CHECK-DAG: field.mul
// CHECK-DAG: field.add
// CHECK: tensor.from_elements
func.func @test_dot_basic(%vec: tensor<3x!PF>) -> tensor<!PF> {
  %const_vec = field.constant dense<[1, 2, 3]> : tensor<3x!PF>
  %init = bufferization.alloc_tensor() : tensor<!PF>
  %result = linalg.dot ins(%const_vec, %vec : tensor<3x!PF>, tensor<3x!PF>)
                       outs(%init : tensor<!PF>) -> tensor<!PF>
  return %result : tensor<!PF>
}

// -----
// Test 3: Dot with constant on RHS - should also fold

// CHECK-LABEL: @test_dot_const_rhs
// CHECK-NOT: linalg.dot
// CHECK-DAG: tensor.extract
// CHECK-DAG: field.mul
// CHECK-DAG: field.add
// CHECK: tensor.from_elements
func.func @test_dot_const_rhs(%vec: tensor<3x!PF>) -> tensor<!PF> {
  %const_vec = field.constant dense<[7, 8, 9]> : tensor<3x!PF>
  %init = bufferization.alloc_tensor() : tensor<!PF>
  %result = linalg.dot ins(%vec, %const_vec : tensor<3x!PF>, tensor<3x!PF>)
                       outs(%init : tensor<!PF>) -> tensor<!PF>
  return %result : tensor<!PF>
}

// -----
// Test 4: Size limit (with default limit=16) - should fold since 10 <= 16

// CHECK-LABEL: @test_size_within_limit
// CHECK-NOT: linalg.dot
// CHECK: tensor.from_elements
func.func @test_size_within_limit(%vec: tensor<10x!PF>) -> tensor<!PF> {
  %const_vec = field.constant dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : tensor<10x!PF>
  %init = bufferization.alloc_tensor() : tensor<!PF>
  %result = linalg.dot ins(%const_vec, %vec : tensor<10x!PF>, tensor<10x!PF>)
                       outs(%init : tensor<!PF>) -> tensor<!PF>
  return %result : tensor<!PF>
}

// -----
// Test 5: Size limit exceeded with small option - should NOT fold

// CHECK-SMALL-LABEL: @test_exceeds_small_limit
// CHECK-SMALL: linalg.dot
func.func @test_exceeds_small_limit(%vec: tensor<3x!PF>) -> tensor<!PF> {
  %const_vec = field.constant dense<[1, 2, 3]> : tensor<3x!PF>
  %init = bufferization.alloc_tensor() : tensor<!PF>
  %result = linalg.dot ins(%const_vec, %vec : tensor<3x!PF>, tensor<3x!PF>)
                       outs(%init : tensor<!PF>) -> tensor<!PF>
  return %result : tensor<!PF>
}

// -----
// Test 6: Variable matrix - should NOT fold

// CHECK-LABEL: @test_variable_matrix
// CHECK: linalg.matvec
func.func @test_variable_matrix(%matrix: tensor<2x3x!PF>, %vec: tensor<3x!PF>) -> tensor<2x!PF> {
  %init = bufferization.alloc_tensor() : tensor<2x!PF>
  %result = linalg.matvec ins(%matrix, %vec : tensor<2x3x!PF>, tensor<3x!PF>)
                          outs(%init : tensor<2x!PF>) -> tensor<2x!PF>
  return %result : tensor<2x!PF>
}

// -----
// Test 7: Both vectors variable in dot - should NOT fold

// CHECK-LABEL: @test_both_variable
// CHECK: linalg.dot
func.func @test_both_variable(%vec1: tensor<3x!PF>, %vec2: tensor<3x!PF>) -> tensor<!PF> {
  %init = bufferization.alloc_tensor() : tensor<!PF>
  %result = linalg.dot ins(%vec1, %vec2 : tensor<3x!PF>, tensor<3x!PF>)
                       outs(%init : tensor<!PF>) -> tensor<!PF>
  return %result : tensor<!PF>
}

// -----
// Test 8: Zero coefficient optimization - zeros should be skipped

// CHECK-LABEL: @test_zero_coefficients
// CHECK-NOT: linalg.dot
// Note: The number of field.mul ops should be less than 3 since zeros are skipped
// Coefficient at index 1 is zero, so we skip c₁ * x[1]
// CHECK-COUNT-2: field.mul
// CHECK-NOT: field.mul
func.func @test_zero_coefficients(%vec: tensor<3x!PF>) -> tensor<!PF> {
  %const_vec = field.constant dense<[5, 0, 7]> : tensor<3x!PF>
  %init = bufferization.alloc_tensor() : tensor<!PF>
  %result = linalg.dot ins(%const_vec, %vec : tensor<3x!PF>, tensor<3x!PF>)
                       outs(%init : tensor<!PF>) -> tensor<!PF>
  return %result : tensor<!PF>
}

// -----
// Test 9: One coefficient optimization - no mul for coefficient=1

// CHECK-LABEL: @test_one_coefficient
// CHECK-NOT: linalg.dot
// Only 2 multiplications (for coefficients 2 and 3), the coefficient 1 is optimized away
// CHECK-COUNT-2: field.mul
// CHECK-NOT: field.mul
func.func @test_one_coefficient(%vec: tensor<3x!PF>) -> tensor<!PF> {
  %const_vec = field.constant dense<[1, 2, 3]> : tensor<3x!PF>
  %init = bufferization.alloc_tensor() : tensor<!PF>
  %result = linalg.dot ins(%const_vec, %vec : tensor<3x!PF>, tensor<3x!PF>)
                       outs(%init : tensor<!PF>) -> tensor<!PF>
  return %result : tensor<!PF>
}

// -----
// Test 10: All zeros - should generate a zero constant

// CHECK-LABEL: @test_all_zeros
// CHECK-NOT: linalg.dot
// CHECK-NOT: field.mul
// CHECK-NOT: field.add
// CHECK: field.constant dense<0>
func.func @test_all_zeros(%vec: tensor<3x!PF>) -> tensor<!PF> {
  %const_vec = field.constant dense<[0, 0, 0]> : tensor<3x!PF>
  %init = bufferization.alloc_tensor() : tensor<!PF>
  %result = linalg.dot ins(%const_vec, %vec : tensor<3x!PF>, tensor<3x!PF>)
                       outs(%init : tensor<!PF>) -> tensor<!PF>
  return %result : tensor<!PF>
}
