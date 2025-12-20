// Copyright 2025 The ZKIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

// RUN: zkir-opt -field-to-mod-arith %s | FileCheck %s -enable-var-scope

!PF = !field.pf<7:i32, true>

// CHECK-LABEL: @test_linalg_broadcast
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]], %[[INIT:.*]]: [[INIT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_linalg_broadcast(%input : tensor<!PF>, %init: tensor<4x!PF>) -> tensor<4x!PF> {
  // CHECK: %[[RES:.*]] = linalg.broadcast ins(%[[INPUT]] : [[INPUT_TYPE]]) outs(%[[INIT]] : [[INIT_TYPE]]) dimensions = [0]
  %res = linalg.broadcast ins(%input:tensor<!PF>) outs(%init:tensor<4x!PF>) dimensions = [0]
  // CHECK: return %[[RES]] : [[T]]
  return %res : tensor<4x!PF>
}

// CHECK-LABEL: @test_linalg_generic
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]], %[[OUTPUT:.*]]: [[OUTPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_linalg_generic(%input : tensor<4x!PF>, %output: tensor<4x!PF>) -> tensor<4x!PF> {
  // CHECK: %[[RES:.*]] = linalg.generic {indexing_maps = [#[[MAP0:.*]], #[[MAP1:.*]]], iterator_types = ["parallel"]} ins(%[[INPUT]] : [[INPUT_TYPE]]) outs(%[[OUTPUT]] : [[OUTPUT_TYPE]]) {
  // CHECK: ^bb0(%[[ARG0:.*]]: [[MOD_ARITH_TYPE:.*]], %[[ARG1:.*]]: [[MOD_ARITH_TYPE:.*]]):
  // CHECK:   %[[SUM:.*]] = mod_arith.add %[[ARG0]], %[[ARG1]] : [[MOD_ARITH_TYPE]]
  // CHECK:   linalg.yield %[[SUM]] : [[MOD_ARITH_TYPE]]
  // CHECK: } -> [[T]]
  %res = linalg.generic {
    indexing_maps = [
      affine_map<(i) -> (i)>,
      affine_map<(i) -> (i)>
    ],
    iterator_types = ["parallel"]
  } ins(%input : tensor<4x!PF>) outs(%output : tensor<4x!PF>) {
  ^bb0(%arg0: !PF, %arg1: !PF):
    %sum = field.add %arg0, %arg1 : !PF
    linalg.yield %sum : !PF
  } -> tensor<4x!PF>
  // CHECK: return %[[RES]] : [[T]]
  return %res : tensor<4x!PF>
}

// CHECK-LABEL: @test_linalg_map
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]], %[[OUTPUT:.*]]: [[OUTPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_linalg_map(%input : tensor<4x!PF>, %output: tensor<4x!PF>) -> tensor<4x!PF> {
  // CHECK: %[[RES:.*]] = linalg.map { mod_arith.double } ins(%[[INPUT]] : [[INPUT_TYPE]]) outs(%[[OUTPUT]] : [[OUTPUT_TYPE]])
  %res = linalg.map ins(%input : tensor<4x!PF>) outs(%output : tensor<4x!PF>) (%arg0: !PF) {
    %double = field.double %arg0 : !PF
    linalg.yield %double : !PF
  }
  // CHECK: return %[[RES]] : [[T]]
  return %res : tensor<4x!PF>
}

// CHECK-LABEL: @test_linalg_reduce
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]], %[[INIT:.*]]: [[INIT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_linalg_reduce(%input: tensor<4x!PF>, %init: tensor<!PF>) -> tensor<!PF> {
  // CHECK: %[[RES:.*]] = linalg.reduce { mod_arith.add } ins(%[[INPUT]] : [[INPUT_TYPE]]) outs(%[[INIT]] : [[INIT_TYPE]]) dimensions = [0]
  %res = linalg.reduce { field.add } ins(%input : tensor<4x!PF>) outs(%init : tensor<!PF>) dimensions = [0]
  // CHECK: return %[[RES]] : [[T]]
  return %res : tensor<!PF>
}

// CHECK-LABEL: @test_linalg_transpose
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[OUTPUT_TYPE:.*]] {
func.func @test_linalg_transpose(%input : tensor<2x3x!PF>) -> tensor<3x2x!PF> {
  // CHECK: %[[OUTPUT:.*]] = tensor.empty() : [[OUTPUT_TYPE]]
  %output = tensor.empty() : tensor<3x2x!PF>
  // CHECK: %[[RES:.*]] = linalg.transpose ins(%[[INPUT]] : [[INPUT_TYPE]]) outs(%[[OUTPUT]] : [[OUTPUT_TYPE]]) permutation = [1, 0]
  %res = linalg.transpose ins(%input : tensor<2x3x!PF>) outs(%output : tensor<3x2x!PF>) permutation = [1, 0]
  // CHECK: return %[[RES]] : [[OUTPUT_TYPE]]
  return %res : tensor<3x2x!PF>
}
