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

// RUN: prime-ir-opt %s | FileCheck %s -enable-var-scope

!PF = !field.pf<7:i32>
!EF2 = !field.ef<2x!PF, 6:i32>
!EF3 = !field.ef<3x!PF, 2:i32>

//===----------------------------------------------------------------------===//
// Basic tensor bitcast: extension field tensor to prime field tensor
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_ef_to_pf_tensor_1elem
// CHECK-SAME: (%[[ARG:.*]]: tensor<1x!field.ef<3x!pf7_i32, 2 : i32>>) -> tensor<3x!pf7_i32>
func.func @test_ef_to_pf_tensor_1elem(%arg0: tensor<1x!EF3>) -> tensor<3x!PF> {
  // CHECK: field.bitcast %[[ARG]] : tensor<1x!field.ef<3x!pf7_i32, 2 : i32>> -> tensor<3x!pf7_i32>
  %0 = field.bitcast %arg0 : tensor<1x!EF3> -> tensor<3x!PF>
  return %0 : tensor<3x!PF>
}

// CHECK-LABEL: @test_ef_to_pf_tensor_multi
// CHECK-SAME: (%[[ARG:.*]]: tensor<4x!field.ef<2x!pf7_i32, 6 : i32>>) -> tensor<8x!pf7_i32>
func.func @test_ef_to_pf_tensor_multi(%arg0: tensor<4x!EF2>) -> tensor<8x!PF> {
  // CHECK: field.bitcast %[[ARG]] : tensor<4x!field.ef<2x!pf7_i32, 6 : i32>> -> tensor<8x!pf7_i32>
  %0 = field.bitcast %arg0 : tensor<4x!EF2> -> tensor<8x!PF>
  return %0 : tensor<8x!PF>
}

//===----------------------------------------------------------------------===//
// Basic tensor bitcast: prime field tensor to extension field tensor
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_pf_to_ef_tensor_1elem
// CHECK-SAME: (%[[ARG:.*]]: tensor<3x!pf7_i32>) -> tensor<1x!field.ef<3x!pf7_i32, 2 : i32>>
func.func @test_pf_to_ef_tensor_1elem(%arg0: tensor<3x!PF>) -> tensor<1x!EF3> {
  // CHECK: field.bitcast %[[ARG]] : tensor<3x!pf7_i32> -> tensor<1x!field.ef<3x!pf7_i32, 2 : i32>>
  %0 = field.bitcast %arg0 : tensor<3x!PF> -> tensor<1x!EF3>
  return %0 : tensor<1x!EF3>
}

// CHECK-LABEL: @test_pf_to_ef_tensor_multi
// CHECK-SAME: (%[[ARG:.*]]: tensor<8x!pf7_i32>) -> tensor<4x!field.ef<2x!pf7_i32, 6 : i32>>
func.func @test_pf_to_ef_tensor_multi(%arg0: tensor<8x!PF>) -> tensor<4x!EF2> {
  // CHECK: field.bitcast %[[ARG]] : tensor<8x!pf7_i32> -> tensor<4x!field.ef<2x!pf7_i32, 6 : i32>>
  %0 = field.bitcast %arg0 : tensor<8x!PF> -> tensor<4x!EF2>
  return %0 : tensor<4x!EF2>
}

//===----------------------------------------------------------------------===//
// Round-trip tensor bitcast
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_roundtrip_ef_pf_ef
// CHECK-SAME: (%[[ARG:.*]]: tensor<2x!field.ef<3x!pf7_i32, 2 : i32>>) -> tensor<2x!field.ef<3x!pf7_i32, 2 : i32>>
func.func @test_roundtrip_ef_pf_ef(%arg0: tensor<2x!EF3>) -> tensor<2x!EF3> {
  // CHECK: %[[TMP:.*]] = field.bitcast %[[ARG]] : tensor<2x!field.ef<3x!pf7_i32, 2 : i32>> -> tensor<6x!pf7_i32>
  %0 = field.bitcast %arg0 : tensor<2x!EF3> -> tensor<6x!PF>
  // CHECK: %[[RES:.*]] = field.bitcast %[[TMP]] : tensor<6x!pf7_i32> -> tensor<2x!field.ef<3x!pf7_i32, 2 : i32>>
  %1 = field.bitcast %0 : tensor<6x!PF> -> tensor<2x!EF3>
  // CHECK: return %[[RES]]
  return %1 : tensor<2x!EF3>
}

//===----------------------------------------------------------------------===//
// 2D tensor bitcast
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_2d_ef_to_pf_tensor
// CHECK-SAME: (%[[ARG:.*]]: tensor<2x3x!field.ef<2x!pf7_i32, 6 : i32>>) -> tensor<2x6x!pf7_i32>
func.func @test_2d_ef_to_pf_tensor(%arg0: tensor<2x3x!EF2>) -> tensor<2x6x!PF> {
  // CHECK: field.bitcast %[[ARG]] : tensor<2x3x!field.ef<2x!pf7_i32, 6 : i32>> -> tensor<2x6x!pf7_i32>
  %0 = field.bitcast %arg0 : tensor<2x3x!EF2> -> tensor<2x6x!PF>
  return %0 : tensor<2x6x!PF>
}

// CHECK-LABEL: @test_flatten_ef_to_pf_tensor
// CHECK-SAME: (%[[ARG:.*]]: tensor<2x3x!field.ef<2x!pf7_i32, 6 : i32>>) -> tensor<12x!pf7_i32>
func.func @test_flatten_ef_to_pf_tensor(%arg0: tensor<2x3x!EF2>) -> tensor<12x!PF> {
  // CHECK: field.bitcast %[[ARG]] : tensor<2x3x!field.ef<2x!pf7_i32, 6 : i32>> -> tensor<12x!pf7_i32>
  %0 = field.bitcast %arg0 : tensor<2x3x!EF2> -> tensor<12x!PF>
  return %0 : tensor<12x!PF>
}
