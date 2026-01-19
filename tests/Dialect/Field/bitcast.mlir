// Copyright 2026 The PrimeIR Authors.
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

// RUN: prime-ir-opt -split-input-file %s | FileCheck %s -enable-var-scope

!PF = !field.pf<7:i32>
!EF3 = !field.ef<3x!PF, 2:i32>
!Tensor3PF = tensor<3x!PF>

// CHECK-LABEL: @test_bitcast_ef_to_tensor
func.func @test_bitcast_ef_to_tensor(%ef: !EF3) -> !Tensor3PF {
  // CHECK: field.bitcast %{{.*}} : !field.ef<3x!pf7_i32, 2 : i32> -> tensor<3x!pf7_i32>
  %tensor = field.bitcast %ef : !EF3 -> !Tensor3PF
  return %tensor : !Tensor3PF
}

// CHECK-LABEL: @test_bitcast_tensor_to_ef
func.func @test_bitcast_tensor_to_ef(%tensor: !Tensor3PF) -> !EF3 {
  // CHECK: field.bitcast %{{.*}} : tensor<3x!pf7_i32> -> !field.ef<3x!pf7_i32, 2 : i32>
  %ef = field.bitcast %tensor : !Tensor3PF -> !EF3
  return %ef : !EF3
}

// CHECK-LABEL: @test_bitcast_tensor_ef_to_tensor_pf
func.func @test_bitcast_tensor_ef_to_tensor_pf(%ef: tensor<2x!EF3>) -> tensor<2x3x!PF> {
  // CHECK: field.bitcast %{{.*}} : tensor<2x!field.ef<3x!pf7_i32, 2 : i32>> -> tensor<2x3x!pf7_i32>
  %pf = field.bitcast %ef : tensor<2x!EF3> -> tensor<2x3x!PF>
  return %pf : tensor<2x3x!PF>
}

// CHECK-LABEL: @test_bitcast_tensor_pf_to_tensor_ef
func.func @test_bitcast_tensor_pf_to_tensor_ef(%pf: tensor<2x3x!PF>) -> tensor<2x!EF3> {
  // CHECK: field.bitcast %{{.*}} : tensor<2x3x!pf7_i32> -> tensor<2x!field.ef<3x!pf7_i32, 2 : i32>>
  %ef = field.bitcast %pf : tensor<2x3x!PF> -> tensor<2x!EF3>
  return %ef : tensor<2x!EF3>
}

// CHECK-LABEL: @test_bitcast_int_to_pf
func.func @test_bitcast_int_to_pf(%val: i32) -> !PF {
  // CHECK: field.bitcast %{{.*}} : i32 -> !pf7_i32
  %pf = field.bitcast %val : i32 -> !PF
  return %pf : !PF
}

// CHECK-LABEL: @test_bitcast_pf_to_int
func.func @test_bitcast_pf_to_int(%pf: !PF) -> i32 {
  // CHECK: field.bitcast %{{.*}} : !pf7_i32 -> i32
  %val = field.bitcast %pf : !PF -> i32
  return %val : i32
}

// CHECK-LABEL: @test_bitcast_tensor_int_to_pf
func.func @test_bitcast_tensor_int_to_pf(%val: tensor<4xi32>) -> tensor<4x!PF> {
  // CHECK: field.bitcast %{{.*}} : tensor<4xi32> -> tensor<4x!pf7_i32>
  %pf_tensor = field.bitcast %val : tensor<4xi32> -> tensor<4x!PF>
  return %pf_tensor : tensor<4x!PF>
}

// CHECK-LABEL: @test_bitcast_tensor_pf_to_int
func.func @test_bitcast_tensor_pf_to_int(%pf_tensor: tensor<4x!PF>) -> tensor<4xi32> {
  // CHECK: field.bitcast %{{.*}} : tensor<4x!pf7_i32> -> tensor<4xi32>
  %val = field.bitcast %pf_tensor : tensor<4x!PF> -> tensor<4xi32>
  return %val : tensor<4xi32>
}
