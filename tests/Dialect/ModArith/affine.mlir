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

// CHECK-LABEL: @test_affine_for
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]], %[[OUTPUT:.*]]: [[OUTPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_affine_for(%input : memref<4x!Zp>, %output : memref<4x!Zp>) -> memref<4x!Zp> {
  // CHECK: affine.for %[[I:.*]] = 0 to 4 {
  // CHECK: }
  affine.for %i = 0 to 4 {
    %load = affine.load %input[%i] : memref<4x!Zp>
    %double = mod_arith.double %load : !Zp
    affine.store %double, %output[%i] : memref<4x!Zp>
  }
  // CHECK: return %[[OUTPUT]] : [[T]]
  return %output : memref<4x!Zp>
}

// CHECK-LABEL: @test_affine_parallel
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]], %[[OUTPUT:.*]]: [[OUTPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_affine_parallel(%input : memref<4x!Zp>, %output : memref<4x!Zp>) -> memref<4x!Zp> {
  // CHECK: affine.parallel (%[[I:.*]]) = (0) to (4) {
  // CHECK: }
  affine.parallel (%i) = (0) to (4) {
    %load = affine.load %input[%i] : memref<4x!Zp>
    %double = mod_arith.add %load, %load : !Zp
    affine.store %double, %output[%i] : memref<4x!Zp>
  }
  // CHECK: return %[[OUTPUT]] : [[T]]
  return %output : memref<4x!Zp>
}
