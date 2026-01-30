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

// RUN: prime-ir-opt %s --field-to-llvm \
// RUN:   | mlir-runner -e test_tensor_bitcast -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s --check-prefix=CHECK_BITCAST < %t

!PF = !field.pf<7:i32>
!EF2 = !field.ef<2x!PF, 6:i32>

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @test_tensor_bitcast() {
  // Test 1: EF2 -> PF -> i32 bitcast chain
  %ef2_tensor = field.constant dense<[[1, 2], [3, 4]]> : tensor<2x!EF2>
  %pf4_tensor = field.bitcast %ef2_tensor : tensor<2x!EF2> -> tensor<4x!PF>
  %i32_tensor = field.bitcast %pf4_tensor : tensor<4x!PF> -> tensor<4xi32>
  %memref1 = bufferization.to_buffer %i32_tensor : tensor<4xi32> to memref<4xi32>
  %unranked1 = memref.cast %memref1 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%unranked1) : (memref<*xi32>) -> ()
  // CHECK_BITCAST: [1, 2, 3, 4]

  // Test 2: EF2 -> i32 direct bitcast
  %i32_result2 = field.bitcast %ef2_tensor : tensor<2x!EF2> -> tensor<4xi32>
  %memref2 = bufferization.to_buffer %i32_result2 : tensor<4xi32> to memref<4xi32>
  %unranked2 = memref.cast %memref2 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%unranked2) : (memref<*xi32>) -> ()
  // CHECK_BITCAST: [1, 2, 3, 4]

  return
}
