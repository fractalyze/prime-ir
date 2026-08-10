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

// RUN: prime-ir-opt -poly-to-field -split-input-file %s | FileCheck %s -enable-var-scope

!PF1 = !field.pf<7:i255>
!poly_ty1 = !poly.polynomial<!PF1, 3>

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
