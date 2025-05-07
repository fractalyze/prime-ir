// RUN: zkir-opt %s -canonicalize | FileCheck %s

!coeff_ty = !field.pf<7681:i32>
#elem = #field.pf_elem<3383:i32>  : !coeff_ty
#root = #poly.primitive_root<root=#elem, degree=4 :i32>
!poly_ty = !poly.polynomial<!coeff_ty, 3>
!tensor_ty = tensor<4x!coeff_ty>

// CHECK-LABEL: @test_canonicalize_intt_after_ntt
// CHECK: (%[[P:.*]]: [[T:.*]]) -> [[T]]
func.func @test_canonicalize_intt_after_ntt(%p0 : !poly_ty) -> !poly_ty {
  // CHECK-NOT: poly.ntt
  // CHECK-NOT: poly.intt
  // CHECK: %[[RESULT:.*]] = poly.add %[[P]], %[[P]]  : [[T]]
  %coeffs = poly.to_tensor %p0 : !poly_ty -> !tensor_ty
  %evals = poly.ntt %coeffs {root=#root} : !tensor_ty
  %coeffs1 = poly.intt %evals {root=#root} : !tensor_ty
  %p1 = poly.from_tensor %coeffs1 : !tensor_ty -> !poly_ty
  %p2 = poly.add %p1, %p1 : !poly_ty
  // CHECK: return %[[RESULT]] : [[T]]
  return %p2 : !poly_ty
}

// CHECK-LABEL: @test_canonicalize_ntt_after_intt
// CHECK: (%[[X:.*]]: [[T:.*]]) -> [[T]]
func.func @test_canonicalize_ntt_after_intt(%t0 : !tensor_ty) -> !tensor_ty {
  // CHECK-NOT: poly.intt
  // CHECK-NOT: poly.ntt
  // CHECK: %[[RESULT:.*]] = field.pf.add %[[X]], %[[X]] : [[T]]
  %coeffs = poly.intt %t0 {root=#root} : !tensor_ty
  %evals = poly.ntt %coeffs {root=#root} : !tensor_ty
  %evals2 = field.pf.add %evals, %evals : !tensor_ty
  // CHECK: return %[[RESULT]] : [[T]]
  return %evals2 : !tensor_ty
}
