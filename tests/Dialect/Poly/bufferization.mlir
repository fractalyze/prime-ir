// RUN: zkir-opt -poly-to-field %s | FileCheck %s -enable-var-scope

!PF = !field.pf<7:i32, true>
!poly_ty = !poly.polynomial<!PF, 3>

// CHECK-LABEL: @test_bufferization_alloc_tensor
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_bufferization_alloc_tensor() -> tensor<2x!poly_ty> {
  // CHECK: %[[TENSOR:.*]] = bufferization.alloc_tensor() : [[T]]
  %tensor = bufferization.alloc_tensor() : tensor<2x!poly_ty>
  // CHECK: return %[[TENSOR]] : [[T]]
  return %tensor : tensor<2x!poly_ty>
}
