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

// RUN: zkir-opt -canonicalize %s | FileCheck %s -enable-var-scope

!PF = !field.pf<7:i32>
!QF = !field.f2<!PF, 6:i32>

//===----------------------------------------------------------------------===//
// ExtToCoeffsOp and ExtFromCoeffsOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_ext_to_coeffs_of_ext_from_coeffs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_ext_to_coeffs_of_ext_from_coeffs(%arg0: !QF) -> !QF {
  %0:2 = field.ext_to_coeffs %arg0 : (!QF) -> (!PF, !PF)
  %1 = field.ext_from_coeffs %0#0, %0#1 : (!PF, !PF) -> !QF
  // CHECK-NOT: field.ext_from_coeffs
  // CHECK: return %[[ARG0]] : [[T]]
  return %1 : !QF
}

// CHECK-LABEL: @test_ext_to_coeffs_of_swapped_ext_from_coeffs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_ext_to_coeffs_of_swapped_ext_from_coeffs(%arg0: !QF) -> !QF {
  %0:2 = field.ext_to_coeffs %arg0 : (!QF) -> (!PF, !PF)
  %1 = field.ext_from_coeffs %0#1, %0#0 : (!PF, !PF) -> !QF
  // CHECK: %[[EXT_TO_COEFFS:.*]]:2 = field.ext_to_coeffs %[[ARG0]] : ([[T]]) -> ([[T2:.*]], [[T2]])
  // CHECK: %[[EXT_FROM_COEFFS:.*]] = field.ext_from_coeffs %[[EXT_TO_COEFFS]]#1, %[[EXT_TO_COEFFS]]#0 : ([[T2]], [[T2]]) -> [[T]]
  // CHECK: return %[[EXT_FROM_COEFFS]] : [[T]]
  return %1 : !QF
}

// CHECK-LABEL: @test_ext_from_coeffs_of_ext_to_coeffs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> ([[T]], [[T]]) {
func.func @test_ext_from_coeffs_of_ext_to_coeffs(%arg0: !PF, %arg1: !PF) -> (!PF, !PF) {
  %0 = field.ext_from_coeffs %arg0, %arg1 : (!PF, !PF) -> (!QF)
  %1:2 = field.ext_to_coeffs %0 : (!QF) -> (!PF, !PF)
  // CHECK-NOT: field.ext_to_coeffs
  // CHECK: return %[[ARG0]], %[[ARG1]] : [[T]], [[T]]
  return %1#0, %1#1 : !PF, !PF
}

//===----------------------------------------------------------------------===//
// Tensor operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_tensor_pf_from_elements
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_pf_from_elements() -> tensor<2x!PF> {
  %0 = field.constant 1 : !PF
  %1 = field.constant 2 : !PF
  %2 = tensor.from_elements %0, %1 : tensor<2x!PF>
  // CHECK: %[[C:.*]] = field.constant dense<[1, 2]> : [[T]]
  // CHECK-NOT: tensor.from_elements
  // CHECK: return %[[C:.*]] : [[T]]
  return %2 : tensor<2x!PF>
}

// CHECK-LABEL: @test_tensor_pf_extract
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_pf_extract() -> !PF {
  // CHECK: %[[C:.*]] = field.constant 3 : [[T]]
  // CHECK-NOT: tensor.extract
  // CHECK: return %[[C]] : [[T]]
  %c1 = arith.constant 1: index
  %0 = field.constant dense<[2, 3]> : tensor<2x!PF>
  %1 = tensor.extract %0[%c1] : tensor<2x!PF>
  return %1 : !PF
}

//===----------------------------------------------------------------------===//
// Vector operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_vector_pf_from_elements
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_vector_pf_from_elements() -> vector<2x!PF> {
  %0 = field.constant 1 : !PF
  %1 = field.constant 2 : !PF
  %2 = vector.from_elements %0, %1 : vector<2x!PF>
  // CHECK: %[[FROM_ELEMENTS:.*]] = field.constant dense<[1, 2]> : [[T]]
  // CHECK-NOT: vector.from_elements
  // CHECK: return %[[FROM_ELEMENTS:.*]] : [[T]]
  return %2 : vector<2x!PF>
}

// CHECK-LABEL: @test_vector_pf_extract
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_vector_pf_extract() -> !PF {
  // CHECK: %[[C:.*]] = field.constant 3 : [[T]]
  // CHECK-NOT: vector.extract
  // CHECK: return %[[C]] : [[T]]
  %0 = field.constant dense<[2, 3]> : vector<2x!PF>
  %1 = vector.extract %0[1] : !PF from vector<2x!PF>
  return %1 : !PF
}
