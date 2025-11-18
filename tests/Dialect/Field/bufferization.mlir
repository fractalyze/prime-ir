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

// CHECK-LABEL: @test_bufferization_alloc_tensor
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_bufferization_alloc_tensor() -> tensor<2x!PF> {
  // CHECK: %[[TENSOR:.*]] = bufferization.alloc_tensor() : [[T]]
  %tensor = bufferization.alloc_tensor() : tensor<2x!PF>
  // CHECK: return %[[TENSOR]] : [[T]]
  return %tensor : tensor<2x!PF>
}

// CHECK-LABEL: @test_bufferization_materialize_in_destination
// CHECK-SAME: (%[[TENSOR:.*]]: [[TENSOR_TYPE:.*]], %[[MEMREF:.*]]: [[MEMREF_TYPE:.*]]) {
func.func @test_bufferization_materialize_in_destination(%tensor : tensor<2x!PF>, %memref : memref<2x!PF>) {
  // CHECK: bufferization.materialize_in_destination %[[TENSOR]] in restrict writable %[[MEMREF]] : ([[TENSOR_TYPE]], [[MEMREF_TYPE]]) -> ()
  bufferization.materialize_in_destination %tensor in restrict writable %memref : (tensor<2x!PF>, memref<2x!PF>) -> ()
  return
}

// CHECK-LABEL: @test_bufferization_to_buffer
// CHECK-SAME: (%[[TENSOR:.*]]: [[TENSOR_TYPE:.*]]) -> [[T:.*]] {
func.func @test_bufferization_to_buffer(%tensor : tensor<2x!PF>) -> memref<2x!PF> {
  // CHECK: %[[MEMREF:.*]] = bufferization.to_buffer %[[TENSOR]] : [[TENSOR_TYPE]] to [[T]]
  %memref = bufferization.to_buffer %tensor : tensor<2x!PF> to memref<2x!PF>
  // CHECK: return %[[MEMREF]] : [[T]]
  return %memref : memref<2x!PF>
}

// CHECK-LABEL: @test_bufferization_to_tensor
// CHECK-SAME: (%[[MEMREF:.*]]: [[MEMREF_TYPE:.*]]) -> [[T:.*]] {
func.func @test_bufferization_to_tensor(%memref : memref<2x!PF>) -> tensor<2x!PF> {
  // CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[MEMREF]] : [[MEMREF_TYPE]] to [[T]]
  %tensor = bufferization.to_tensor %memref : memref<2x!PF> to tensor<2x!PF>
  // CHECK: return %[[TENSOR]] : [[T]]
  return %tensor : tensor<2x!PF>
}
