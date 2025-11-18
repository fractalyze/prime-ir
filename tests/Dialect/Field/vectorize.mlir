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

// RUN: zkir-opt -affine-super-vectorize=virtual-vector-size=16 %s | FileCheck %s -enable-var-scope

!PF = !field.pf<65537 : i32>

#beta = #field.pf.elem<65536:i32> : !PF
!QF = !field.f2<!PF, #beta>

// CHECK-LABEL: @test_vectorize
func.func @test_vectorize(%buffer : memref<1024x!PF>) {
  // CHECK: affine.for %arg1 = 0 to 1024 step 16 {
  // CHECK: vector.transfer_read
  affine.for %i = 0 to 1024 {
    %load = affine.load %buffer[%i] : memref<1024x!PF>
    %square = field.square %load : !PF
    affine.store %square, %buffer[%i] : memref<1024x!PF>
  }
  return
}

// CHECK-LABEL: @test_vectorize_quadratic
func.func @test_vectorize_quadratic(%buffer : memref<1024x!QF>) {
  // CHECK: affine.for %arg1 = 0 to 1024 step 16 {
  // CHECK: vector.transfer_read
  affine.for %i = 0 to 1024 {
    %load = affine.load %buffer[%i] : memref<1024x!QF>
    %square = field.square %load : !QF
    affine.store %square, %buffer[%i] : memref<1024x!QF>
  }
  return
}
