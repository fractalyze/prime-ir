// RUN: zkir-opt -field-to-mod-arith %s | FileCheck %s -enable-var-scope

!PF = !field.pf<7:i32, true>

// CHECK-LABEL: @test_memref_alloc
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_memref_alloc() -> memref<4x!PF> {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : [[T]]
  %alloc = memref.alloc() : memref<4x!PF>
  // CHECK: return %[[ALLOC]] : [[T]]
  return %alloc : memref<4x!PF>
}

// CHECK-LABEL: @test_memref_cast
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_memref_cast(%input : memref<4x!PF>) -> memref<?x!PF> {
  // CHECK: %[[CAST:.*]] = memref.cast %[[INPUT]] : [[INPUT_TYPE]] to [[T]]
  %cast = memref.cast %input : memref<4x!PF> to memref<?x!PF>
  // CHECK: return %[[CAST]] : [[T]]
  return %cast : memref<?x!PF>
}

// CHECK-LABEL: @test_memref_unranked_cast
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_memref_unranked_cast(%input : memref<4x!PF>) -> memref<*x!PF> {
  // CHECK: %[[CAST:.*]] = memref.cast %[[INPUT]] : [[INPUT_TYPE]] to [[T]]
  %cast = memref.cast %input : memref<4x!PF> to memref<*x!PF>
  // CHECK: return %[[CAST]] : [[T]]
  return %cast : memref<*x!PF>
}

// CHECK-LABEL: @test_memref_copy
// CHECK-SAME: (%[[SRC:.*]]: [[SRC_TYPE:.*]], %[[DST:.*]]: [[DST_TYPE:.*]]) {
func.func @test_memref_copy(%src : memref<4x!PF>, %dst : memref<4x!PF>) {
  // CHECK: memref.copy %[[SRC]], %[[DST]] : [[SRC_TYPE]] to [[DST_TYPE]]
  memref.copy %src, %dst : memref<4x!PF> to memref<4x!PF>
  return
}

// CHECK-LABEL: @test_memref_subview
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_memref_subview(%input : memref<8x!PF>) -> memref<4x!PF, strided<[1], offset: 2>> {
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %[[INPUT]][2] [4] [1] : [[INPUT_TYPE]] to [[T]]
  %subview = memref.subview %input[2] [4] [1] : memref<8x!PF> to memref<4x!PF, strided<[1], offset: 2>>
  // CHECK: return %[[SUBVIEW]] : [[T]]
  return %subview : memref<4x!PF, strided<[1], offset: 2>>
}

// CHECK-LABEL: @test_memref_view
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]], %[[OFFSET:.*]]: [[OFFSET_TYPE:.*]]) -> [[T:.*]] {
func.func @test_memref_view(%input : memref<4x!PF>, %offset : index) -> memref<2x!PF, strided<[1], offset: ?>> {
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %[[INPUT]][%[[OFFSET]]] [2] [1] : [[INPUT_TYPE]] to [[T]]
  %subview = memref.subview %input[%offset] [2] [1] : memref<4x!PF> to memref<2x!PF, strided<[1], offset: ?>>
  // CHECK: return %[[SUBVIEW]] : [[T]]
  return %subview : memref<2x!PF, strided<[1], offset: ?>>
}
