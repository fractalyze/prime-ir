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

// -----
// Test 11: Extension field matvec - should fold constant matrix

!QF = !field.ef<2x!field.pf<97:i32>, 5:i32>

// CHECK-LABEL: @test_matvec_ext_field
// CHECK-NOT: linalg.matvec
// 2x2 matrix folded into scalar ops (muls/adds interleaved, so use CHECK-DAG)
// CHECK-DAG: field.mul
// CHECK-DAG: field.add
// CHECK: tensor.from_elements
func.func @test_matvec_ext_field(%vec: tensor<2x!QF>) -> tensor<2x!QF> {
  // Matrix 2x2 of quadratic extension field elements
  // Each element is [c₀, c₁] representing c₀ + c₁v
  %matrix = field.constant dense<[[[1, 0], [2, 1]], [[3, 2], [4, 3]]]> : tensor<2x2x!QF>
  %init = bufferization.alloc_tensor() : tensor<2x!QF>
  %result = linalg.matvec ins(%matrix, %vec : tensor<2x2x!QF>, tensor<2x!QF>)
                          outs(%init : tensor<2x!QF>) -> tensor<2x!QF>
  return %result : tensor<2x!QF>
}

// -----
// Test 12: Extension field dot - should fold constant vector

!QF2 = !field.ef<2x!field.pf<97:i32>, 5:i32>

// CHECK-LABEL: @test_dot_ext_field
// CHECK-NOT: linalg.dot
// 2 non-zero/non-one coefficients: 2 muls, 1 add
// CHECK-COUNT-2: field.mul
// CHECK-NOT: field.mul
// CHECK-COUNT-1: field.add
// CHECK-NOT: field.add
func.func @test_dot_ext_field(%vec: tensor<2x!QF2>) -> tensor<!QF2> {
  %const_vec = field.constant dense<[[1, 2], [3, 4]]> : tensor<2x!QF2>
  %init = bufferization.alloc_tensor() : tensor<!QF2>
  %result = linalg.dot ins(%const_vec, %vec : tensor<2x!QF2>, tensor<2x!QF2>)
                       outs(%init : tensor<!QF2>) -> tensor<!QF2>
  return %result : tensor<!QF2>
}

// -----
// Test 13: Extension field with zero element optimization
// [0, 0] is the additive identity

!QF3 = !field.ef<2x!field.pf<97:i32>, 5:i32>

// CHECK-LABEL: @test_ext_field_zero_coeff
// CHECK-NOT: linalg.dot
// Only one mul since [0, 0] is zero
// CHECK-COUNT-1: field.mul
// CHECK-NOT: field.mul
func.func @test_ext_field_zero_coeff(%vec: tensor<2x!QF3>) -> tensor<!QF3> {
  %const_vec = field.constant dense<[[1, 2], [0, 0]]> : tensor<2x!QF3>
  %init = bufferization.alloc_tensor() : tensor<!QF3>
  %result = linalg.dot ins(%const_vec, %vec : tensor<2x!QF3>, tensor<2x!QF3>)
                       outs(%init : tensor<!QF3>) -> tensor<!QF3>
  return %result : tensor<!QF3>
}

// -----
// Test 14: Extension field with one element optimization
// [1, 0] is the multiplicative identity

!QF4 = !field.ef<2x!field.pf<97:i32>, 5:i32>

// CHECK-LABEL: @test_ext_field_one_coeff
// CHECK-NOT: linalg.dot
// Only one mul since [1, 0] doesn't need multiplication
// CHECK-COUNT-1: field.mul
// CHECK-NOT: field.mul
func.func @test_ext_field_one_coeff(%vec: tensor<2x!QF4>) -> tensor<!QF4> {
  %const_vec = field.constant dense<[[1, 0], [2, 3]]> : tensor<2x!QF4>
  %init = bufferization.alloc_tensor() : tensor<!QF4>
  %result = linalg.dot ins(%const_vec, %vec : tensor<2x!QF4>, tensor<2x!QF4>)
                       outs(%init : tensor<!QF4>) -> tensor<!QF4>
  return %result : tensor<!QF4>
}

// Note: Extension field "all zeros" test is omitted because
// maybeConvertPrimeIRToBuiltinType in prime_ir/IR/Attributes.cpp doesn't handle
// ExtensionFieldType yet. Once added, tensor.from_elements fold will work for
// extension fields when all elements are constant.

// -----
// Test 15: Tower extension field dot - basic case
// Fp6 = (Fp2)³ where Fp2 = Fp[v]/(v² - 6), Fp6 = Fp2[w]/(w³ - 2)
// Each Fp6 element has 6 prime field coefficients
// Two non-zero/non-one coefficients → 2 muls, 1 add

!PF_tower = !field.pf<7:i32>
!QF_tower = !field.ef<2x!PF_tower, 6:i32>
!TowerF6 = !field.ef<3x!QF_tower, 2:i32>

// CHECK-LABEL: @test_dot_tower_ext_field
// CHECK-NOT: linalg.dot
// CHECK-COUNT-2: field.mul
// CHECK-NOT: field.mul
// CHECK-COUNT-1: field.add
// CHECK-NOT: field.add
func.func @test_dot_tower_ext_field(%vec: tensor<2x!TowerF6>) -> tensor<!TowerF6> {
  // Two non-zero, non-one coefficients (flattened to 6 prime field elements)
  // [1, 2, 0, 0, 0, 0] = (1 + 2v) in Fp6
  // [3, 4, 0, 0, 0, 0] = (3 + 4v) in Fp6
  %const_vec = field.constant dense<[
    [1, 2, 0, 0, 0, 0],
    [3, 4, 0, 0, 0, 0]
  ]> : tensor<2x!TowerF6>
  %init = bufferization.alloc_tensor() : tensor<!TowerF6>
  %result = linalg.dot ins(%const_vec, %vec : tensor<2x!TowerF6>, tensor<2x!TowerF6>)
                       outs(%init : tensor<!TowerF6>) -> tensor<!TowerF6>
  return %result : tensor<!TowerF6>
}

// -----
// Test 16: Tower extension field with zero coefficient optimization
// One zero coefficient → 1 mul, 0 adds

!PF_tower2 = !field.pf<7:i32>
!QF_tower2 = !field.ef<2x!PF_tower2, 6:i32>
!TowerF6_2 = !field.ef<3x!QF_tower2, 2:i32>

// CHECK-LABEL: @test_tower_zero_coeff
// CHECK-NOT: linalg.dot
// CHECK-COUNT-1: field.mul
// CHECK-NOT: field.mul
// CHECK-NOT: field.add
func.func @test_tower_zero_coeff(%vec: tensor<2x!TowerF6_2>) -> tensor<!TowerF6_2> {
  // First element is zero [0,0,0,0,0,0], second is non-zero [2,3,0,0,0,0]
  %const_vec = field.constant dense<[
    [0, 0, 0, 0, 0, 0],
    [2, 3, 0, 0, 0, 0]
  ]> : tensor<2x!TowerF6_2>
  %init = bufferization.alloc_tensor() : tensor<!TowerF6_2>
  %result = linalg.dot ins(%const_vec, %vec : tensor<2x!TowerF6_2>, tensor<2x!TowerF6_2>)
                       outs(%init : tensor<!TowerF6_2>) -> tensor<!TowerF6_2>
  return %result : tensor<!TowerF6_2>
}

// -----
// Test 17: Tower extension field with identity coefficient optimization
// Identity [1,0,0,0,0,0] skips multiplication

!PF_tower3 = !field.pf<7:i32>
!QF_tower3 = !field.ef<2x!PF_tower3, 6:i32>
!TowerF6_3 = !field.ef<3x!QF_tower3, 2:i32>

// CHECK-LABEL: @test_tower_one_coeff
// CHECK-NOT: linalg.dot
// First coeff is identity [1,0,0,0,0,0] (no mul), second is [2,0,0,0,0,0] (1 mul)
// CHECK-COUNT-1: field.mul
// CHECK-NOT: field.mul
// CHECK-COUNT-1: field.add
// CHECK-NOT: field.add
func.func @test_tower_one_coeff(%vec: tensor<2x!TowerF6_3>) -> tensor<!TowerF6_3> {
  // First element is identity [1,0,0,0,0,0], second is non-trivial [2,0,0,0,0,0]
  %const_vec = field.constant dense<[
    [1, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0]
  ]> : tensor<2x!TowerF6_3>
  %init = bufferization.alloc_tensor() : tensor<!TowerF6_3>
  %result = linalg.dot ins(%const_vec, %vec : tensor<2x!TowerF6_3>, tensor<2x!TowerF6_3>)
                       outs(%init : tensor<!TowerF6_3>) -> tensor<!TowerF6_3>
  return %result : tensor<!TowerF6_3>
}
