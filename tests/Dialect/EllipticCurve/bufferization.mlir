// RUN: zkir-opt -elliptic-curve-to-field %s | FileCheck %s -enable-var-scope

!PF = !field.pf<97:i32>

#1 = #field.pf.elem<1:i32> : !PF
#2 = #field.pf.elem<2:i32> : !PF
#3 = #field.pf.elem<3:i32> : !PF
#4 = #field.pf.elem<4:i32> : !PF

#curve = #elliptic_curve.sw<#1, #2, (#3, #4)>
!affine = !elliptic_curve.affine<#curve>

// CHECK-LABEL: @test_bufferization_alloc_tensor
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_bufferization_alloc_tensor() -> tensor<2x!affine> {
  // CHECK: %[[TENSOR:.*]] = bufferization.alloc_tensor() : [[T]]
  %tensor = bufferization.alloc_tensor() : tensor<2x!affine>
  // CHECK: return %[[TENSOR]] : [[T]]
  return %tensor : tensor<2x!affine>
}

// CHECK-LABEL: @test_bufferization_materialize_in_destination
// CHECK-SAME: (%[[TENSOR:.*]]: [[TENSOR_TYPE:.*]], %[[MEMREF:.*]]: [[MEMREF_TYPE:.*]]) {
func.func @test_bufferization_materialize_in_destination(%tensor : tensor<2x!affine>, %memref : memref<2x!affine>) {
  // CHECK: bufferization.materialize_in_destination %[[TENSOR]] in restrict writable %[[MEMREF]] : ([[TENSOR_TYPE]], [[MEMREF_TYPE]]) -> ()
  bufferization.materialize_in_destination %tensor in restrict writable %memref : (tensor<2x!affine>, memref<2x!affine>) -> ()
  return
}

// CHECK-LABEL: @test_bufferization_to_memref
// CHECK-SAME: (%[[TENSOR:.*]]: [[TENSOR_TYPE:.*]]) -> [[T:.*]] {
func.func @test_bufferization_to_memref(%tensor : tensor<2x!affine>) -> memref<2x!affine> {
  // CHECK: %[[MEMREF:.*]] = bufferization.to_memref %[[TENSOR]] : [[TENSOR_TYPE]] to [[T]]
  %memref = bufferization.to_memref %tensor : tensor<2x!affine> to memref<2x!affine>
  // CHECK: return %[[MEMREF]] : [[T]]
  return %memref : memref<2x!affine>
}

// CHECK-LABEL: @test_bufferization_to_tensor
// CHECK-SAME: (%[[MEMREF:.*]]: [[MEMREF_TYPE:.*]]) -> [[T:.*]] {
func.func @test_bufferization_to_tensor(%memref : memref<2x!affine>) -> tensor<2x!affine> {
  // CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[MEMREF]] : [[MEMREF_TYPE]] to [[T]]
  %tensor = bufferization.to_tensor %memref : memref<2x!affine> to tensor<2x!affine>
  // CHECK: return %[[TENSOR]] : [[T]]
  return %tensor : tensor<2x!affine>
}
