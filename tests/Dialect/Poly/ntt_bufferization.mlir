// RUN: zkir-opt %s -poly-to-field -field-to-mod-arith -mod-arith-to-arith -tensor-ext-to-tensor \
// RUN:    -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -canonicalize | FileCheck %s

!PF = !field.pf<7681:i32>
#elem = #field.pf.elem<3383:i32>  : !PF
#root_of_unity = #field.root_of_unity<#elem, 4:i32>

// CHECK-LABEL: @ntt_in_place
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @ntt_in_place(%arg0: tensor<4x!PF>) -> tensor<4x!PF> {
  // CHECK-NOT: poly.ntt
  // CHECK-NOT: memref.copy
  %1 = poly.ntt %arg0 into %arg0 {root=#root_of_unity} : tensor<4x!PF>
  // CHECK: return %[[ARG0]] : [[T]]
  return %1 : tensor<4x!PF>
}

// CHECK-LABEL: @ntt_out_of_place
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @ntt_out_of_place(%arg0: tensor<4x!PF> {bufferization.writable = false}) -> tensor<4x!PF> {
  // CHECK-NOT: poly.ntt
  // CHECK-NOT: memref.copy
  %temp = bufferization.alloc_tensor() : tensor<4x!PF>
  %1 = poly.ntt %arg0 into %temp {root=#root_of_unity} : tensor<4x!PF>
  return %1 : tensor<4x!PF>
}

// CHECK-LABEL: @intt_in_place
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @intt_in_place(%arg0: tensor<4x!PF>) -> tensor<4x!PF> {
  // CHECK-NOT: poly.ntt
  // CHECK-NOT: memref.copy
  %1 = poly.ntt %arg0 into %arg0 {root=#root_of_unity} inverse=true : tensor<4x!PF>
  // CHECK: return %[[ARG0]] : [[T]]
  return %1 : tensor<4x!PF>
}

// CHECK-LABEL: @intt_out_of_place
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @intt_out_of_place(%arg0: tensor<4x!PF> {bufferization.writable = false}) -> tensor<4x!PF> {
  // CHECK-NOT: poly.ntt
  // CHECK-NOT: memref.copy
  %temp = bufferization.alloc_tensor() : tensor<4x!PF>
  %1 = poly.ntt %arg0 into %temp {root=#root_of_unity} inverse=true : tensor<4x!PF>
  return %1 : tensor<4x!PF>
}
