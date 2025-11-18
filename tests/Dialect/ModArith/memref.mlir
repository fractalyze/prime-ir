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

// RUN: zkir-opt -mod-arith-to-arith %s | FileCheck %s -enable-var-scope

!Zp = !mod_arith.int<65537 : i32>

// CHECK-LABEL: @test_memref_alloc
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_memref_alloc() -> memref<4x!Zp> {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : [[T]]
  %alloc = memref.alloc() : memref<4x!Zp>
  // CHECK: return %[[ALLOC]] : [[T]]
  return %alloc : memref<4x!Zp>
}

// CHECK-LABEL: @test_memref_alloca
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_memref_alloca() -> memref<4x!Zp> {
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() : [[T]]
  %alloca = memref.alloca() : memref<4x!Zp>
  // CHECK: return %[[ALLOCA]] : [[T]]
  return %alloca : memref<4x!Zp>
}

// CHECK-LABEL: @test_memref_cast
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_memref_cast(%input : memref<4x!Zp>) -> memref<?x!Zp> {
  // CHECK: %[[CAST:.*]] = memref.cast %[[INPUT]] : [[INPUT_TYPE]] to [[T]]
  %cast = memref.cast %input : memref<4x!Zp> to memref<?x!Zp>
  // CHECK: return %[[CAST]] : [[T]]
  return %cast : memref<?x!Zp>
}

// CHECK-LABEL: @test_memref_unranked_cast
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_memref_unranked_cast(%input : memref<4x!Zp>) -> memref<*x!Zp> {
  // CHECK: %[[CAST:.*]] = memref.cast %[[INPUT]] : [[INPUT_TYPE]] to [[T]]
  %cast = memref.cast %input : memref<4x!Zp> to memref<*x!Zp>
  // CHECK: return %[[CAST]] : [[T]]
  return %cast : memref<*x!Zp>
}

// CHECK-LABEL: @test_memref_copy
// CHECK-SAME: (%[[SRC:.*]]: [[SRC_TYPE:.*]], %[[DST:.*]]: [[DST_TYPE:.*]]) {
func.func @test_memref_copy(%src : memref<4x!Zp>, %dst : memref<4x!Zp>) {
  // CHECK: memref.copy %[[SRC]], %[[DST]] : [[SRC_TYPE]] to [[DST_TYPE]]
  memref.copy %src, %dst : memref<4x!Zp> to memref<4x!Zp>
  return
}

// CHECK-LABEL: @test_memref_dim
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_memref_dim(%input : memref<?x!Zp>) -> index {
  %c0 = arith.constant 0 : index
  // CHECK: %[[DIM:.*]] = memref.dim %[[INPUT]], %[[C0:.*]] : [[INPUT_TYPE]]
  %dim = memref.dim %input, %c0 : memref<?x!Zp>
  // CHECK: return %[[DIM]] : [[T]]
  return %dim : index
}

// CHECK-LABEL: @test_memref_load
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_memref_load(%input : memref<4x!Zp>) -> !Zp {
  %c0 = arith.constant 0 : index
  // CHECK: %[[LOAD:.*]] = memref.load %[[INPUT]][%[[C0:.*]]] : [[INPUT_TYPE]]
  %load = memref.load %input[%c0] : memref<4x!Zp>
  // CHECK: return %[[LOAD]] : [[T]]
  return %load : !Zp
}

// CHECK-LABEL: @test_memref_store
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]], %[[T:.*]]: [[T_TYPE:.*]]) {
func.func @test_memref_store(%input : memref<4x!Zp>, %value : !Zp) {
  %c0 = arith.constant 0 : index
  // CHECK: memref.store %[[T]], %[[INPUT]][%[[C0:.*]]] : [[INPUT_TYPE]]
  memref.store %value, %input[%c0] : memref<4x!Zp>
  return
}

/// CHECK-LABEL: @test_memref_subview
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_memref_subview(%input : memref<8x!Zp>) -> memref<4x!Zp, strided<[1], offset: 2>> {
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %[[INPUT]][2] [4] [1] : [[INPUT_TYPE]] to [[T]]
  %subview = memref.subview %input[2] [4] [1] : memref<8x!Zp> to memref<4x!Zp, strided<[1], offset: 2>>
  // CHECK: return %[[SUBVIEW]] : [[T]]
  return %subview : memref<4x!Zp, strided<[1], offset: 2>>
}

// CHECK-LABEL: @test_memref_view
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]], %[[OFFSET:.*]]: [[OFFSET_TYPE:.*]]) -> [[T:.*]] {
func.func @test_memref_view(%input : memref<4x!Zp>, %offset : index) -> memref<2x!Zp, strided<[1], offset: ?>> {
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %[[INPUT]][%[[OFFSET]]] [2] [1] : [[INPUT_TYPE]] to [[T]]
  %subview = memref.subview %input[%offset] [2] [1] : memref<4x!Zp> to memref<2x!Zp, strided<[1], offset: ?>>
  // CHECK: return %[[SUBVIEW]] : [[T]]
  return %subview : memref<2x!Zp, strided<[1], offset: ?>>
}
