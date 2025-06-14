// RUN: zkir-opt -field-to-mod-arith %s | FileCheck %s -enable-var-scope

!PF1 = !field.pf<7:i32, true>

// CHECK-LABEL: @test_memref_alloc
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_memref_alloc() -> memref<4x!PF1> {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : [[T]]
  %alloc = memref.alloc() : memref<4x!PF1>
  // CHECK: return %[[ALLOC]] : [[T]]
  return %alloc : memref<4x!PF1>
}

// CHECK-LABEL: @test_memref_cast
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_memref_cast(%input : memref<4x!PF1>) -> memref<?x!PF1> {
  // CHECK: %[[CAST:.*]] = memref.cast %[[INPUT]] : [[INPUT_TYPE]] to [[T]]
  %cast = memref.cast %input : memref<4x!PF1> to memref<?x!PF1>
  // CHECK: return %[[CAST]] : [[T]]
  return %cast : memref<?x!PF1>
}

// CHECK-LABEL: @test_memref_copy
// CHECK-SAME: (%[[SRC:.*]]: [[SRC_TYPE:.*]], %[[DST:.*]]: [[DST_TYPE:.*]]) {
func.func @test_memref_copy(%src : memref<4x!PF1>, %dst : memref<4x!PF1>) {
  // CHECK: memref.copy %[[SRC]], %[[DST]] : [[SRC_TYPE]] to [[DST_TYPE]]
  memref.copy %src, %dst : memref<4x!PF1> to memref<4x!PF1>
  return
}

// TODO(chokobole): add test_memref_subview and test_memref_view
