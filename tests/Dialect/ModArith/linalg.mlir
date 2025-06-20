// RUN: zkir-opt -mod-arith-to-arith %s | FileCheck %s -enable-var-scope

!Zp = !mod_arith.int<65537 : i32>

// CHECK-LABEL: @test_linalg_broadcast
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]], %[[INIT:.*]]: [[INIT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_linalg_broadcast(%input : tensor<!Zp>, %init: tensor<4x!Zp>) -> tensor<4x!Zp> {
  // CHECK: %[[RES:.*]] = linalg.broadcast ins(%[[INPUT]] : [[INPUT_TYPE]]) outs(%[[INIT]] : [[INIT_TYPE]]) dimensions = [0]
  %res = linalg.broadcast ins(%input:tensor<!Zp>) outs(%init:tensor<4x!Zp>) dimensions = [0]
  // CHECK: return %[[RES]] : [[T]]
  return %res : tensor<4x!Zp>
}

// CHECK-LABEL: @test_linalg_generic
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]], %[[OUTPUT:.*]]: [[OUTPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_linalg_generic(%input : tensor<4x!Zp>, %output: tensor<4x!Zp>) -> tensor<4x!Zp> {
  // CHECK: %[[RES:.*]] = linalg.generic {indexing_maps = [#[[MAP0:.*]], #[[MAP1:.*]]], iterator_types = ["parallel"]} ins(%[[INPUT]] : [[INPUT_TYPE]]) outs(%[[OUTPUT]] : [[OUTPUT_TYPE]]) {
  // CHECK: ^bb0(%[[ARG0:.*]]: [[ARITH_TYPE:.*]], %[[ARG1:.*]]: [[ARITH_TYPE:.*]]):
  // CHECK: } -> [[T]]
  %res = linalg.generic {
    indexing_maps = [
      affine_map<(i) -> (i)>,
      affine_map<(i) -> (i)>
    ],
    iterator_types = ["parallel"]
  } ins(%input : tensor<4x!Zp>) outs(%output : tensor<4x!Zp>) {
  ^bb0(%arg0: !Zp, %arg1: !Zp):
    %sum = mod_arith.add %arg0, %arg1 : !Zp
    linalg.yield %sum : !Zp
  } -> tensor<4x!Zp>
  // CHECK: return %[[RES]] : [[T]]
  return %res : tensor<4x!Zp>
}

// CHECK-LABEL: @test_linalg_map
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]], %[[OUTPUT:.*]]: [[OUTPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_linalg_map(%input : tensor<4x!Zp>, %output: tensor<4x!Zp>) -> tensor<4x!Zp> {
  // CHECK: %[[RES:.*]] = linalg.map ins(%[[INPUT]] : [[INPUT_TYPE]]) outs(%[[OUTPUT]] : [[OUTPUT_TYPE]])
  %res = linalg.map ins(%input : tensor<4x!Zp>) outs(%output : tensor<4x!Zp>) (%arg0: !Zp) {
    %square = mod_arith.double %arg0 : !Zp
    linalg.yield %square : !Zp
  }
  // CHECK: return %[[RES]] : [[T]]
  return %res : tensor<4x!Zp>
}
