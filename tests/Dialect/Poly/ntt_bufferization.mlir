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

// RUN: zkir-opt %s -poly-to-field -field-to-mod-arith -mod-arith-to-arith -tensor-ext-to-tensor \
// RUN:    -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -canonicalize | FileCheck %s

!PF = !field.pf<7681:i32>
#root_of_unity = #field.root_of_unity<3383:i32, 4:i32> : !PF

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
