// RUN: zkir-opt -elliptic-curve-to-field %s | FileCheck %s -enable-var-scope

!PF = !field.pf<97:i32>

#1 = #field.pf.elem<1:i32> : !PF
#2 = #field.pf.elem<2:i32> : !PF
#3 = #field.pf.elem<3:i32> : !PF
#4 = #field.pf.elem<4:i32> : !PF

#curve = #elliptic_curve.sw<#1, #2, (#3, #4)>
!affine = !elliptic_curve.affine<#curve>

// CHECK-LABEL: @test_memref_alloc
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_memref_alloc() -> memref<4x!affine> {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : [[T]]
  %alloc = memref.alloc() : memref<4x!affine>
  // CHECK: return %[[ALLOC]] : [[T]]
  return %alloc : memref<4x!affine>
}

// CHECK-LABEL: @test_memref_alloca
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_memref_alloca() -> memref<4x!affine> {
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() : [[T]]
  %alloca = memref.alloca() : memref<4x!affine>
  // CHECK: return %[[ALLOCA]] : [[T]]
  return %alloca : memref<4x!affine>
}

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

// CHECK-LABEL: @test_memref_dim
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_memref_dim(%input : memref<?x!affine>) -> index {
  %c0 = arith.constant 0 : index
  // CHECK: %[[DIM:.*]] = memref.dim %[[INPUT]], %[[C0:.*]] : [[INPUT_TYPE]]
  %dim = memref.dim %input, %c0 : memref<?x!affine>
  // CHECK: return %[[DIM]] : [[T]]
  return %dim : index
}

// CHECK-LABEL: @test_memref_load
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> ([[T:.*]], [[T:.*]]) {
func.func @test_memref_load(%input : memref<4x!affine>) -> !affine {
  %c0 = arith.constant 0 : index
  // CHECK: %[[LOAD0:.*]] = memref.load %[[INPUT]][%[[C0:.*]], %[[C0_0:.*]]] : [[INPUT_TYPE]]
  // CHECK: %[[LOAD1:.*]] = memref.load %[[INPUT]][%[[C0:.*]], %[[C1:.*]]] : [[INPUT_TYPE]]
  %load = memref.load %input[%c0] : memref<4x!affine>
  // CHECK: return %[[LOAD0]], %[[LOAD1]] : [[T]], [[T]]
  return %load : !affine
}

// CHECK-LABEL: @test_memref_store
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]], %[[ELEM0:.*]]: [[ELEM_TYPE:.*]], %[[ELEM1:.*]]: [[ELEM_TYPE:.*]]) {
func.func @test_memref_store(%input : memref<4x!affine>, %point : !affine) {
  %c0 = arith.constant 0 : index
  // CHECK: memref.store %[[ELEM0]], %[[INPUT]][%[[C0:.*]], %[[C0_0:.*]]] : [[INPUT_TYPE]]
  // CHECK: memref.store %[[ELEM1]], %[[INPUT]][%[[C0:.*]], %[[C1:.*]]] : [[INPUT_TYPE]]
  memref.store %point, %input[%c0] : memref<4x!affine>
  return
}

// CHECK-LABEL: @test_memref_subview
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_memref_subview(%input : memref<8x!affine>) -> memref<4x!affine, strided<[1], offset: 2>> {
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %[[INPUT]][2, 0] [4, 2] [1, 1] : [[INPUT_TYPE]] to [[T]]
  %subview = memref.subview %input[2] [4] [1] : memref<8x!affine> to memref<4x!affine, strided<[1], offset: 2>>
  // CHECK: return %[[SUBVIEW]] : [[T]]
  return %subview : memref<4x!affine, strided<[1], offset: 2>>
}
