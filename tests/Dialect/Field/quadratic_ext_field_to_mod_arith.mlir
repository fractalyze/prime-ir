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

// RUN: zkir-opt -field-to-mod-arith -split-input-file %s | FileCheck %s -enable-var-scope
!PF = !field.pf<7:i32>
!PFm = !field.pf<7:i32, true>
!QF = !field.f2<!PF, 6:i32>
!QFm = !field.f2<!PFm, 6:i32>

// CHECK-LABEL: @test_lower_inverse
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_inverse(%arg0: !QF) -> !QF {
    // CHECK-NOT: field.inverse
    %0 = field.inverse %arg0 : !QF
    return %0 : !QF
}

// CHECK-LABEL: @test_lower_double
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_double(%arg0: !QF) -> !QF {
    // CHECK-NOT: field.double
    %0 = field.double %arg0 : !QF
    return %0 : !QF
}

// CHECK-LABEL: @test_lower_square
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_square(%arg0: !QF) -> !QF {
    // CHECK-NOT: field.square
    %0 = field.square %arg0 : !QF
    return %0 : !QF
}

// CHECK-LABEL: @test_lower_add
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_add(%arg0: !QF, %arg1: !QF) -> !QF {
    // CHECK: %[[LHS:.*]]:2 = field.ext_to_coeffs %[[ARG0]] : ([[T]]) -> (!z7_i32, !z7_i32)
    // CHECK: %[[RHS:.*]]:2 = field.ext_to_coeffs %[[ARG1]] : ([[T]]) -> (!z7_i32, !z7_i32)
    // CHECK: %[[C0:.*]] = mod_arith.add %[[LHS]]#0, %[[RHS]]#0 : !z7_i32
    // CHECK: %[[C1:.*]] = mod_arith.add %[[LHS]]#1, %[[RHS]]#1 : !z7_i32
    // CHECK: %[[RESULT:.*]] = field.ext_from_coeffs %[[C0]], %[[C1]] : (!z7_i32, !z7_i32) -> [[T]]
    // CHECK: return %[[RESULT]] : [[T]]
    %0 = field.add %arg0, %arg1 : !QF
    return %0 : !QF
}

// CHECK-LABEL: @test_lower_mul
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_mul(%arg0: !QF, %arg1: !QF) -> !QF {
    // CHECK: %[[BETA:.*]] = mod_arith.constant 6 : !z7_i32
    // CHECK: %[[LHS:.*]]:2 = field.ext_to_coeffs %[[ARG0]] : ([[T]]) -> (!z7_i32, !z7_i32)
    // CHECK: %[[RHS:.*]]:2 = field.ext_to_coeffs %[[ARG1]] : ([[T]]) -> (!z7_i32, !z7_i32)
    // CHECK: %[[V0:.*]] = mod_arith.mul %[[LHS]]#0, %[[RHS]]#0 : !z7_i32
    // CHECK: %[[V1:.*]] = mod_arith.mul %[[LHS]]#1, %[[RHS]]#1 : !z7_i32
    // CHECK: %[[BETATIMESV1:.*]] = mod_arith.mul %[[BETA]], %[[V1]] : !z7_i32
    // CHECK: %[[C0:.*]] = mod_arith.add %[[V0]], %[[BETATIMESV1]] : !z7_i32
    // CHECK: %[[SUMLHS:.*]] = mod_arith.add %[[LHS]]#0, %[[LHS]]#1 : !z7_i32
    // CHECK: %[[SUMRHS:.*]] = mod_arith.add %[[RHS]]#0, %[[RHS]]#1 : !z7_i32
    // CHECK: %[[SUMPRODUCT:.*]] = mod_arith.mul %[[SUMLHS]], %[[SUMRHS]] : !z7_i32
    // CHECK: %[[TMP:.*]] = mod_arith.sub %[[SUMPRODUCT]], %[[V0]] : !z7_i32
    // CHECK: %[[C1:.*]] = mod_arith.sub %[[TMP]], %[[V1]] : !z7_i32
    // CHECK: %[[RESULT:.*]] = field.ext_from_coeffs %[[C0]], %[[C1]] : (!z7_i32, !z7_i32) -> [[T]]
    // CHECK: return %[[RESULT]] : [[T]]
    %0 = field.mul %arg0, %arg1 : !QF
    return %0 : !QF
}

// CHECK-LABEL: @test_lower_from_elements
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> tensor<2x[[T]]> {
func.func @test_lower_from_elements(%arg0: !QF, %arg1: !QF) -> tensor<2x!QF> {
    // CHECK: %[[TENSOR:.*]] = tensor.from_elements %[[ARG0]], %[[ARG1]] : tensor<2x[[T]]>
    %2 = tensor.from_elements %arg0, %arg1 : tensor<2x!QF>
    // CHECK: return %[[TENSOR]] : tensor<2x[[T]]>
    return %2 : tensor<2x!QF>
}

// CHECK-LABEL: @test_lower_from_mont
// CHECK-SAME: (%[[ARG0:.*]]: [[Tm:.*]]) -> [[T:.*]] {
func.func @test_lower_from_mont(%arg0: !QFm) -> !QF {
    // CHECK-NOT: field.from_mont
    %0 = field.from_mont %arg0 : !QF
    return %0 : !QF
}

// CHECK-LABEL: @test_lower_to_mont
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[Tm:.*]] {
func.func @test_lower_to_mont(%arg0: !QF) -> !QFm {
    // CHECK-NOT: field.to_mont
    %0 = field.to_mont %arg0 : !QFm
    return %0 : !QFm
}

// CHECK-LABEL: @test_lower_tensor_extract
// CHECK-SAME: (%[[ARG0:.*]]: tensor<3x2x[[T:.*]]>) -> [[T]] {
func.func @test_lower_tensor_extract(%arg0: tensor<3x2x!QF>) -> !QF {
    // CHECK: %[[I1:.*]] = arith.constant 1 : index
    %i1 = arith.constant 1 : index
    // CHECK: %[[VALUE:.*]] = tensor.extract %[[ARG0]][%[[I1]], %[[I1]]] : tensor<3x2x[[T]]>
    %1 = tensor.extract %arg0[%i1, %i1] : tensor<3x2x!QF>
    // CHECK: return %[[VALUE]] : [[T]]
    return %1 : !QF
}

// CHECK-LABEL: @test_lower_memref
// CHECK-SAME: (%[[ARG0:.*]]: memref<3x2x[[T:.*]]>) -> [[T]] {
func.func @test_lower_memref(%arg0: memref<3x2x!QF>) -> !QF {
    %t = bufferization.to_tensor %arg0 : memref<3x2x!QF> to tensor<3x2x!QF>
    %i1 = arith.constant 1 : index
    %1 = tensor.extract %t[%i1, %i1] : tensor<3x2x!QF>
    return %1 : !QF
}
