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

// RUN: prime-ir-opt -field-to-mod-arith %s | FileCheck %s -enable-var-scope

!PF = !field.pf<7:i32>
!EF3 = !field.ef<3x!PF, 2:i32>

//===----------------------------------------------------------------------===//
// Tensor bitcast lowering: extension field tensor to prime field tensor
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_ef_to_pf_tensor_1elem
func.func @test_ef_to_pf_tensor_1elem(%arg0: tensor<1x!EF3>) -> tensor<3x!PF> {
  // Zero-copy: keeps field.bitcast op which will be bufferized to memref bitcast.
  // The output type is converted from PF to mod_arith.int, but the op remains.
  // CHECK: field.bitcast
  // CHECK-NOT: tensor.extract
  // CHECK-NOT: tensor.from_elements
  // CHECK-NOT: builtin.unrealized_conversion_cast
  %0 = field.bitcast %arg0 : tensor<1x!EF3> -> tensor<3x!PF>
  return %0 : tensor<3x!PF>
}

//===----------------------------------------------------------------------===//
// Tensor bitcast lowering: prime field tensor to extension field tensor
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_pf_to_ef_tensor_1elem
func.func @test_pf_to_ef_tensor_1elem(%arg0: tensor<3x!PF>) -> tensor<1x!EF3> {
  // Zero-copy: keeps field.bitcast op which will be bufferized to memref bitcast.
  // The input type is converted from PF to mod_arith.int, but the op remains.
  // CHECK: field.bitcast
  // CHECK-NOT: tensor.extract
  // CHECK-NOT: tensor.from_elements
  // CHECK-NOT: builtin.unrealized_conversion_cast
  %0 = field.bitcast %arg0 : tensor<3x!PF> -> tensor<1x!EF3>
  return %0 : tensor<1x!EF3>
}
