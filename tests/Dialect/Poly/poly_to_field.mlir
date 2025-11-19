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

// RUN: zkir-opt -poly-to-field -split-input-file %s | FileCheck %s -enable-var-scope

!PF1 = !field.pf<7:i255>
!poly_ty1 = !poly.polynomial<!PF1, 3>
!poly_ty2 = !poly.polynomial<!PF1, 4>
#elem = #field.pf.elem<6:i255>  : !PF1
#root_of_unity = #field.root_of_unity<#elem, 2:i255>

// CHECK-LABEL: @test_lower_to_tensor
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_to_tensor(%arg0 : !poly_ty1) -> tensor<4x!PF1> {
  // CHECK-NOT: poly.to_tensor
  // CHECK: return %[[ARG0]] : [[T]]
  %res = poly.to_tensor %arg0 : !poly_ty1 -> tensor<4x!PF1>
  return %res : tensor<4x!PF1>
}

// CHECK-LABEL: @test_lower_from_tensor
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_from_tensor(%t : tensor<4x!PF1>) -> !poly_ty1 {
  // CHECK-NOT: poly.from_tensor
  // CHECK: return %[[LHS]] : [[T]]
  %res = poly.from_tensor %t : tensor<4x!PF1> -> !poly_ty1
  return %res : !poly_ty1
}

// CHECK-LABEL: @test_lower_ntt
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_ntt(%input : tensor<2x!PF1>) -> tensor<2x!PF1> {
  // CHECK-NOT: poly.ntt
  %res = poly.ntt %input into %input {root=#root_of_unity}: tensor<2x!PF1>
  return %res: tensor<2x!PF1>
}

// CHECK-LABEL: @test_lower_ntt_with_twiddles
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_ntt_with_twiddles(%input : tensor<2x!PF1>, %twiddles : tensor<2x!PF1>) -> tensor<2x!PF1> {
  // CHECK-NOT: poly.ntt
  // CHECK-NOT: arith.constant dense
  %res = poly.ntt %input into %input with %twiddles: tensor<2x!PF1>
  return %res: tensor<2x!PF1>
}

// CHECK-LABEL: @test_lower_intt
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[P:.*]] {
func.func @test_lower_intt(%input : tensor<2x!PF1>) -> tensor<2x!PF1> {
  // CHECK-NOT: poly.ntt
  %res = poly.ntt %input into %input {root=#root_of_unity} inverse=true : tensor<2x!PF1>
  return %res: tensor<2x!PF1>
}

// CHECK-LABEL: @test_lower_intt_with_twiddles
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_intt_with_twiddles(%input : tensor<2x!PF1>, %twiddles : tensor<2x!PF1>) -> tensor<2x!PF1> {
  // CHECK-NOT: poly.ntt
  // CHECK-NOT: arith.constant dense
  %res = poly.ntt %input into %input with %twiddles inverse=true: tensor<2x!PF1>
  return %res: tensor<2x!PF1>
}
