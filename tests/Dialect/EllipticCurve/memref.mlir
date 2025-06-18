// RUN: zkir-opt -elliptic-curve-to-field %s | FileCheck %s -enable-var-scope

!PF = !field.pf<97:i32>

#1 = #field.pf.elem<1:i32> : !PF
#2 = #field.pf.elem<2:i32> : !PF
#3 = #field.pf.elem<3:i32> : !PF
#4 = #field.pf.elem<4:i32> : !PF

#curve = #elliptic_curve.sw<#1, #2, (#3, #4)>
!affine = !elliptic_curve.affine<#curve>

// CHECK-LABEL: @test_memref_cast
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_memref_cast(%input : memref<4x!affine>) -> memref<?x!affine> {
  // CHECK: %[[CAST:.*]] = memref.cast %[[INPUT]] : [[INPUT_TYPE]] to [[T]]
  %cast = memref.cast %input : memref<4x!affine> to memref<?x!affine>
  // CHECK: return %[[CAST]] : [[T]]
  return %cast : memref<?x!affine>
}

// CHECK-LABEL: @test_memref_unranked_cast
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_memref_unranked_cast(%input : memref<4x!affine>) -> memref<*x!affine> {
  // CHECK: %[[CAST:.*]] = memref.cast %[[INPUT]] : [[INPUT_TYPE]] to [[T]]
  %cast = memref.cast %input : memref<4x!affine> to memref<*x!affine>
  // CHECK: return %[[CAST]] : [[T]]
  return %cast : memref<*x!affine>
}

// CHECK-LABEL: @test_memref_subview
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_memref_subview(%input : memref<8x!affine>) -> memref<4x!affine, strided<[1], offset: 2>> {
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %[[INPUT]][2, 0] [4, 2] [1, 1] : [[INPUT_TYPE]] to [[T]]
  %subview = memref.subview %input[2] [4] [1] : memref<8x!affine> to memref<4x!affine, strided<[1], offset: 2>>
  // CHECK: return %[[SUBVIEW]] : [[T]]
  return %subview : memref<4x!affine, strided<[1], offset: 2>>
}
