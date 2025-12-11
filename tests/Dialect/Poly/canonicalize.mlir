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

// RUN: zkir-opt %s -canonicalize | FileCheck %s

!coeff_ty = !field.pf<7681:i32>
#root_of_unity = #field.root_of_unity<3383:i32, 4:i32> : !coeff_ty
!poly_ty = !poly.polynomial<!coeff_ty, 3>
!tensor_ty = tensor<4x!coeff_ty>

// CHECK-LABEL: @test_canonicalize_intt_after_ntt
// CHECK: (%[[P:.*]]: [[T:.*]]) -> [[T]]
func.func @test_canonicalize_intt_after_ntt(%p0 : !poly_ty) -> !poly_ty {
  // CHECK-NOT: poly.ntt
  %coeffs = poly.to_tensor %p0 : !poly_ty -> !tensor_ty
  %evals = poly.ntt %coeffs into %coeffs {root=#root_of_unity} : !tensor_ty
  %coeffs1 = poly.ntt %evals into %evals {root=#root_of_unity} inverse=true : !tensor_ty
  %evals2 = poly.ntt %coeffs1 into %coeffs1 {root=#root_of_unity} bit_reverse=false : !tensor_ty
  %coeffs2 = poly.ntt %evals2 into %evals2 {root=#root_of_unity} inverse=true bit_reverse=false : !tensor_ty
  %p1 = poly.from_tensor %coeffs1 : !tensor_ty -> !poly_ty
  // CHECK: return %[[P]] : [[T]]
  return %p1 : !poly_ty
}

// CHECK-LABEL: @test_canonicalize_ntt_after_intt
// CHECK: (%[[X:.*]]: [[T:.*]]) -> [[T]]
func.func @test_canonicalize_ntt_after_intt(%t0 : !tensor_ty) -> !tensor_ty {
  // CHECK-NOT: poly.ntt
  // CHECK: %[[RESULT:.*]] = field.double %[[X]] : [[T]]
  %coeffs = poly.ntt %t0 into %t0 {root=#root_of_unity} inverse=true : !tensor_ty
  %evals = poly.ntt %coeffs into %coeffs {root=#root_of_unity} : !tensor_ty
  %coeffs1 = poly.ntt %evals into %evals {root=#root_of_unity} inverse=true bit_reverse=false : !tensor_ty
  %evals2 = poly.ntt %coeffs1 into %coeffs1 {root=#root_of_unity} bit_reverse=false : !tensor_ty
  %evals3 = field.add %evals2, %evals2 : !tensor_ty
  // CHECK: return %[[RESULT]] : [[T]]
  return %evals3 : !tensor_ty
}
