// RUN: zkir-opt -poly-to-field -split-input-file %s | FileCheck %s -enable-var-scope

!PF1 = !field.pf<7:i255>
!poly_ty1 = !poly.polynomial<!PF1, 3>
!poly_ty2 = !poly.polynomial<!PF1, 4>
#elem = #field.pf.elem<6:i255>  : !PF1
#root_of_unity = #field.root_of_unity<#elem, 2:i255>

// CHECK-LABEL: @test_lower_add
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_add(%lhs : !poly_ty1, %rhs : !poly_ty1) -> !poly_ty1 {
  // CHECK-NOT: poly.add
  // CHECK: %[[RES:.*]] = field.add %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  %res = poly.add %lhs, %rhs : !poly_ty1
  return %res : !poly_ty1
}

// CHECK-LABEL: @test_lower_sub
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_sub(%lhs : !poly_ty1, %rhs : !poly_ty1) -> !poly_ty1 {
  // CHECK-NOT: poly.sub
  // CHECK: %[[RES:.*]] = field.sub %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  %res = poly.sub %lhs, %rhs : !poly_ty1
  return %res : !poly_ty1
}

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
  %res = poly.ntt %input {root=#root_of_unity}: tensor<2x!PF1>
  return %res: tensor<2x!PF1>
}

// CHECK-LABEL: @test_lower_ntt_with_twiddles
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_ntt_with_twiddles(%input : tensor<2x!PF1>, %twiddles : tensor<2x!PF1>) -> tensor<2x!PF1> {
  // CHECK-NOT: poly.ntt
  // CHECK-NOT: arith.constant dense
  %res = poly.ntt %input, %twiddles: tensor<2x!PF1>
  return %res: tensor<2x!PF1>
}

// CHECK-LABEL: @test_lower_intt
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[P:.*]] {
func.func @test_lower_intt(%input : tensor<2x!PF1>) -> tensor<2x!PF1> {
  // CHECK-NOT: poly.ntt
  %res = poly.ntt %input {root=#root_of_unity} inverse=true : tensor<2x!PF1>
  return %res: tensor<2x!PF1>
}

// CHECK-LABEL: @test_lower_intt_with_twiddles
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_intt_with_twiddles(%input : tensor<2x!PF1>, %twiddles : tensor<2x!PF1>) -> tensor<2x!PF1> {
  // CHECK-NOT: poly.ntt
  // CHECK-NOT: arith.constant dense
  %res = poly.ntt %input, %twiddles inverse=true: tensor<2x!PF1>
  return %res: tensor<2x!PF1>
}
