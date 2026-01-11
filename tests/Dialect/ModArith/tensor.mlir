// Copyright 2025 The PrimeIR Authors.
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

// RUN: prime-ir-opt -mod-arith-to-arith %s | FileCheck %s -enable-var-scope

!Zp = !mod_arith.int<65537 : i32>

// CHECK-LABEL: @test_tensor_cast
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_tensor_cast(%input : tensor<4x!Zp>) -> tensor<?x!Zp> {
  // CHECK: %[[CAST:.*]] = tensor.cast %[[INPUT]] : [[INPUT_TYPE]] to [[T]]
  %cast = tensor.cast %input : tensor<4x!Zp> to tensor<?x!Zp>
  // CHECK: return %[[CAST]] : [[T]]
  return %cast : tensor<?x!Zp>
}

// CHECK-LABEL: @test_tensor_concat
// CHECK-SAME: (%[[INPUT1:.*]]: [[INPUT1_TYPE:.*]], %[[INPUT2:.*]]: [[INPUT2_TYPE:.*]]) -> [[T:.*]] {
func.func @test_tensor_concat(%input1 : tensor<2x!Zp>, %input2 : tensor<3x!Zp>) -> tensor<5x!Zp> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[CONCAT:.*]] = tensor.concat dim(0) %[[INPUT1]], %[[INPUT2]] : ([[INPUT1_TYPE]], [[INPUT2_TYPE]]) -> [[T]]
  %concat = tensor.concat dim(0) %input1, %input2 : (tensor<2x!Zp>, tensor<3x!Zp>) -> tensor<5x!Zp>
  // CHECK: return %[[CONCAT]] : [[T]]
  return %concat : tensor<5x!Zp>
}

// CHECK-LABEL: @test_tensor_dim
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_tensor_dim(%input : tensor<?x!Zp>) -> index {
  %c0 = arith.constant 0 : index
  // CHECK: %[[DIM:.*]] = tensor.dim %[[INPUT]], %[[C0:.*]] : [[INPUT_TYPE]]
  %dim = tensor.dim %input, %c0 : tensor<?x!Zp>
  // CHECK: return %[[DIM]] : [[T]]
  return %dim : index
}

// CHECK-LABEL: @test_tensor_empty
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_empty() -> tensor<4x!Zp> {
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : [[T]]
  %empty = tensor.empty() : tensor<4x!Zp>
  // CHECK: return %[[EMPTY]] : [[T]]
  return %empty : tensor<4x!Zp>
}

// CHECK-LABEL: @test_tensor_extract
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_tensor_extract(%input : tensor<4x!Zp>) -> !Zp {
  %c0 = arith.constant 0 : index
  // CHECK: %[[EXTRACT:.*]] = tensor.extract %[[INPUT]][%[[C0:.*]]] : [[INPUT_TYPE]]
  %extract = tensor.extract %input[%c0] : tensor<4x!Zp>
  // CHECK: return %[[EXTRACT]] : [[T]]
  return %extract : !Zp
}

// CHECK-LABEL: @test_tensor_extract_slice
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_tensor_extract_slice(%input : tensor<4x!Zp>) -> tensor<?x!Zp> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[INPUT]][%[[C0:.*]]] [%[[C2:.*]]] [%[[C1:.*]]] : [[INPUT_TYPE]] to [[T]]
  %slice = tensor.extract_slice %input[%c0] [%c2] [%c1] : tensor<4x!Zp> to tensor<?x!Zp>
  // CHECK: return %[[SLICE]] : [[T]]
  return %slice : tensor<?x!Zp>
}

// CHECK-LABEL: @test_tensor_from_elements
// CHECK-SAME: (%[[ELEM1:.*]]: [[ELEM_TYPE:.*]], %[[ELEM2:.*]]: [[ELEM_TYPE:.*]], %[[ELEM3:.*]]: [[ELEM_TYPE:.*]]) -> [[T:.*]] {
func.func @test_tensor_from_elements(%elem1: !Zp, %elem2: !Zp, %elem3: !Zp) -> tensor<3x!Zp> {
  // CHECK: %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[ELEM1:.*]], %[[ELEM2:.*]], %[[ELEM3:.*]] : [[T]]
  %from_elements = tensor.from_elements %elem1, %elem2, %elem3 : tensor<3x!Zp>
  // CHECK: return %[[FROM_ELEMENTS]] : [[T]]
  return %from_elements : tensor<3x!Zp>
}

// CHECK-LABEL: @test_tensor_insert
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]], %[[ELEM:.*]]: [[ELEM_TYPE:.*]]) -> [[T:.*]] {
func.func @test_tensor_insert(%input : tensor<4x!Zp>, %elem : !Zp) -> tensor<4x!Zp> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[INSERT:.*]] = tensor.insert %[[ELEM]] into %[[INPUT]][%[[C0:.*]]] : [[INPUT_TYPE]]
  %insert = tensor.insert %elem into %input[%c0] : tensor<4x!Zp>
  // CHECK: return %[[INSERT]] : [[T]]
  return %insert : tensor<4x!Zp>
}

// CHECK-LABEL: @test_tensor_insert_slice
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]], %[[UPDATE:.*]]: [[UPDATE_TYPE:.*]]) -> [[T:.*]] {
func.func @test_tensor_insert_slice(%input : tensor<4x!Zp>, %update : tensor<2x!Zp>) -> tensor<4x!Zp> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[INSERT_SLICE:.*]] = tensor.insert_slice %[[UPDATE]] into %[[INPUT]][%[[C0:.*]]] [2] [%[[C1:.*]]] : [[UPDATE_TYPE]] into [[INPUT_TYPE]]
  %insert_slice = tensor.insert_slice %update into %input[%c0] [2] [%c1] : tensor<2x!Zp> into tensor<4x!Zp>
  // CHECK: return %[[INSERT_SLICE]] : [[T]]
  return %insert_slice : tensor<4x!Zp>
}

// CHECK-LABEL: @test_tensor_pad
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_tensor_pad(%input : tensor<4x!Zp>) -> tensor<6x!Zp> {
  %c0 = mod_arith.constant 0 : !Zp
  // CHECK: %[[PAD:.*]] = tensor.pad %[[INPUT]] low[1] high[1] {
  %pad = tensor.pad %input low[1] high[1] {
    ^bb0(%arg: index):
      tensor.yield %c0 : !Zp
  } : tensor<4x!Zp> to tensor<6x!Zp>
  // CHECK: return %[[PAD]] : [[T]]
  return %pad : tensor<6x!Zp>
}

// CHECK-LABEL: @test_tensor_reshape
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]], %[[SHAPE:.*]]: [[SHAPE_TYPE:.*]]) -> [[T:.*]] {
func.func @test_tensor_reshape(%input : tensor<4x!Zp>, %shape: tensor<2xi32>) -> tensor<2x2x!Zp> {
  // CHECK: %[[RESHAPE:.*]] = tensor.reshape %[[INPUT]](%[[SHAPE]]) : ([[INPUT_TYPE]], tensor<2xi32>) -> [[T]]
  %reshape = tensor.reshape %input(%shape) : (tensor<4x!Zp>, tensor<2xi32>) -> tensor<2x2x!Zp>
  // CHECK: return %[[RESHAPE]] : [[T]]
  return %reshape : tensor<2x2x!Zp>
}
