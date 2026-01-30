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
!EF6 = !field.ef<3x!EF2, 2:i32>  // Fp6 = (Fp2)^3

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

  // Test 3: Tower extension EF6 -> EF2 -> PF -> i32 bitcast chain
  %ef6_tensor = field.constant dense<[[1, 2, 3, 4, 5, 6]]> : tensor<1x!EF6>
  %ef2_from_ef6 = field.bitcast %ef6_tensor : tensor<1x!EF6> -> tensor<3x!EF2>
  %pf_from_ef2 = field.bitcast %ef2_from_ef6 : tensor<3x!EF2> -> tensor<6x!PF>
  %i32_from_pf = field.bitcast %pf_from_ef2 : tensor<6x!PF> -> tensor<6xi32>
  %memref3 = bufferization.to_buffer %i32_from_pf : tensor<6xi32> to memref<6xi32>
  %unranked3 = memref.cast %memref3 : memref<6xi32> to memref<*xi32>
  func.call @printMemrefI32(%unranked3) : (memref<*xi32>) -> ()
  // CHECK_BITCAST: [1, 2, 3, 4, 5, 6]

  // Test 4: Tower extension EF6 -> i32 direct bitcast
  %i32_direct = field.bitcast %ef6_tensor : tensor<1x!EF6> -> tensor<6xi32>
  %memref4 = bufferization.to_buffer %i32_direct : tensor<6xi32> to memref<6xi32>
  %unranked4 = memref.cast %memref4 : memref<6xi32> to memref<*xi32>
  func.call @printMemrefI32(%unranked4) : (memref<*xi32>) -> ()
  // CHECK_BITCAST: [1, 2, 3, 4, 5, 6]

  return
}
