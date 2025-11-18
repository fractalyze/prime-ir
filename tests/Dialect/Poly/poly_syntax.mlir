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

// RUN: zkir-opt -split-input-file %s | FileCheck %s -enable-var-scope

!PF1 = !field.pf<7:i32>
!PF2 = !field.pf<13:i32>
!poly_ty1 = !poly.polynomial<!PF1, 32>
!poly_ty2 = !poly.polynomial<!PF2, 32>
#elem = #field.pf.elem<2:i32>  : !PF1
#root_of_unity = #field.root_of_unity<#elem, 3:i32>
#root = #poly.primitive_root<#root_of_unity>

// CHECK-LABEL: @test_poly_syntax
// CHECK-SAME: (%[[ARG0:.*]]: [[P:.*]], %[[ARG1:.*]]: [[P]]) -> [[P]] {
func.func @test_poly_syntax(%arg0 : !poly_ty1, %arg1 : !poly_ty1) -> !poly_ty1 {
  // CHECK: %[[RES0:.*]] = poly.add %[[ARG0]], %[[ARG1]] : [[P]]
  %0 = poly.add %arg0, %arg1 : !poly_ty1
  // CHECK: %[[RES1:.*]] = poly.sub %[[ARG0]], %[[ARG1]] : [[P]]
  %1 = poly.sub %arg0, %arg1 : !poly_ty1
  // CHECK: %[[RES2:.*]] = poly.mul %[[RES0]], %[[RES1]] : [[P]]
  %2 = poly.mul %0, %1 : !poly_ty1
  // CHECK: %[[RES3:.*]] = poly.to_tensor %[[RES2]] : [[P]] -> [[T:.*]]
  %3 = poly.to_tensor %2 : !poly_ty1 -> tensor<4x!PF1>
  // CHECK: %[[RES4:.*]] = poly.from_tensor %[[RES3]] : [[T]] -> [[P]]
  %4 = poly.from_tensor %3 : tensor<4x!PF1> -> !poly_ty1
  // CHECK: return %[[RES4]] : [[P]]
  return %4 : !poly_ty1
}
