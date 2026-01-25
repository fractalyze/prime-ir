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
!EF3 = !field.ef<3x!PF, 2:i32>

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @test_tensor_bitcast() {
  // ============================================================
  // Test 1: Prime field tensor to extension field tensor (degree 2)
  // then extract and verify coefficients
  // ============================================================
  // Create prime field elements [1, 2, 3, 4] as i32 tensor
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32

  // Create tensor<4xi32> and bitcast to prime field, then to extension field
  %i32_tensor = tensor.from_elements %c1, %c2, %c3, %c4 : tensor<4xi32>
  %pf_tensor = field.bitcast %i32_tensor : tensor<4xi32> -> tensor<4x!PF>
  %ef2_tensor = field.bitcast %pf_tensor : tensor<4x!PF> -> tensor<2x!EF2>

  // Extract extension field elements and verify their coefficients
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index

  %ef2_0 = tensor.extract %ef2_tensor[%c0_idx] : tensor<2x!EF2>
  %ef2_1 = tensor.extract %ef2_tensor[%c1_idx] : tensor<2x!EF2>

  // Get coefficients: first element should be (1, 2), second should be (3, 4)
  %coeff0_0, %coeff0_1 = field.ext_to_coeffs %ef2_0 : (!EF2) -> (!PF, !PF)
  %coeff1_0, %coeff1_1 = field.ext_to_coeffs %ef2_1 : (!EF2) -> (!PF, !PF)

  %coeff0_0_i32 = field.bitcast %coeff0_0 : !PF -> i32
  %coeff0_1_i32 = field.bitcast %coeff0_1 : !PF -> i32
  %coeff1_0_i32 = field.bitcast %coeff1_0 : !PF -> i32
  %coeff1_1_i32 = field.bitcast %coeff1_1 : !PF -> i32

  %result1 = tensor.from_elements %coeff0_0_i32, %coeff0_1_i32, %coeff1_0_i32, %coeff1_1_i32 : tensor<4xi32>
  %memref1 = bufferization.to_buffer %result1 : tensor<4xi32> to memref<4xi32>
  %unranked1 = memref.cast %memref1 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%unranked1) : (memref<*xi32>) -> ()

  // ============================================================
  // Test 2: Extension field tensor back to prime field tensor
  // ============================================================
  // Bitcast back to prime field tensor
  %pf_tensor_back = field.bitcast %ef2_tensor : tensor<2x!EF2> -> tensor<4x!PF>

  %c2_idx = arith.constant 2 : index
  %c3_idx = arith.constant 3 : index

  %pf0 = tensor.extract %pf_tensor_back[%c0_idx] : tensor<4x!PF>
  %pf1 = tensor.extract %pf_tensor_back[%c1_idx] : tensor<4x!PF>
  %pf2 = tensor.extract %pf_tensor_back[%c2_idx] : tensor<4x!PF>
  %pf3 = tensor.extract %pf_tensor_back[%c3_idx] : tensor<4x!PF>

  %i0 = field.bitcast %pf0 : !PF -> i32
  %i1 = field.bitcast %pf1 : !PF -> i32
  %i2 = field.bitcast %pf2 : !PF -> i32
  %i3 = field.bitcast %pf3 : !PF -> i32

  %result2 = tensor.from_elements %i0, %i1, %i2, %i3 : tensor<4xi32>
  %memref2 = bufferization.to_buffer %result2 : tensor<4xi32> to memref<4xi32>
  %unranked2 = memref.cast %memref2 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%unranked2) : (memref<*xi32>) -> ()

  // ============================================================
  // Test 3: Prime field tensor to extension field (degree 3)
  // ============================================================
  %c5 = arith.constant 5 : i32
  %c6 = arith.constant 6 : i32

  %i32_tensor3 = tensor.from_elements %c1, %c2, %c3, %c4, %c5, %c6 : tensor<6xi32>
  %pf_tensor3 = field.bitcast %i32_tensor3 : tensor<6xi32> -> tensor<6x!PF>
  %ef3_tensor = field.bitcast %pf_tensor3 : tensor<6x!PF> -> tensor<2x!EF3>

  %ef3_0 = tensor.extract %ef3_tensor[%c0_idx] : tensor<2x!EF3>
  %ef3_1 = tensor.extract %ef3_tensor[%c1_idx] : tensor<2x!EF3>

  // Get coefficients: first element should be (1, 2, 3), second should be (4, 5, 6)
  %coeff3_0_0, %coeff3_0_1, %coeff3_0_2 = field.ext_to_coeffs %ef3_0 : (!EF3) -> (!PF, !PF, !PF)
  %coeff3_1_0, %coeff3_1_1, %coeff3_1_2 = field.ext_to_coeffs %ef3_1 : (!EF3) -> (!PF, !PF, !PF)

  %coeff3_0_0_i32 = field.bitcast %coeff3_0_0 : !PF -> i32
  %coeff3_0_1_i32 = field.bitcast %coeff3_0_1 : !PF -> i32
  %coeff3_0_2_i32 = field.bitcast %coeff3_0_2 : !PF -> i32
  %coeff3_1_0_i32 = field.bitcast %coeff3_1_0 : !PF -> i32
  %coeff3_1_1_i32 = field.bitcast %coeff3_1_1 : !PF -> i32
  %coeff3_1_2_i32 = field.bitcast %coeff3_1_2 : !PF -> i32

  %result3 = tensor.from_elements %coeff3_0_0_i32, %coeff3_0_1_i32, %coeff3_0_2_i32, %coeff3_1_0_i32, %coeff3_1_1_i32, %coeff3_1_2_i32 : tensor<6xi32>
  %memref3 = bufferization.to_buffer %result3 : tensor<6xi32> to memref<6xi32>
  %unranked3 = memref.cast %memref3 : memref<6xi32> to memref<*xi32>
  func.call @printMemrefI32(%unranked3) : (memref<*xi32>) -> ()

  // ============================================================
  // Test 4: Arithmetic after bitcast - verify operations work correctly
  // ============================================================
  // Create EF2 element via bitcast, perform arithmetic
  %arith_tensor = tensor.from_elements %c2, %c3 : tensor<2xi32>
  %arith_pf = field.bitcast %arith_tensor : tensor<2xi32> -> tensor<2x!PF>
  %arith_ef = field.bitcast %arith_pf : tensor<2x!PF> -> tensor<1x!EF2>

  %ef_elem = tensor.extract %arith_ef[%c0_idx] : tensor<1x!EF2>
  // Add element to itself: (2 + 3v) + (2 + 3v) = (4 + 6v)
  %ef_sum = field.add %ef_elem, %ef_elem : !EF2
  %sum_coeff0, %sum_coeff1 = field.ext_to_coeffs %ef_sum : (!EF2) -> (!PF, !PF)
  %sum_coeff0_i32 = field.bitcast %sum_coeff0 : !PF -> i32
  %sum_coeff1_i32 = field.bitcast %sum_coeff1 : !PF -> i32

  %result4 = tensor.from_elements %sum_coeff0_i32, %sum_coeff1_i32 : tensor<2xi32>
  %memref4 = bufferization.to_buffer %result4 : tensor<2xi32> to memref<2xi32>
  %unranked4 = memref.cast %memref4 : memref<2xi32> to memref<*xi32>
  func.call @printMemrefI32(%unranked4) : (memref<*xi32>) -> ()

  // ============================================================
  // Test 5: Multiplication after bitcast
  // ============================================================
  // (2 + 3v) * (2 + 3v) with xi = 6, v^2 = 6
  // = 4 + 6v + 6v + 9v^2
  // = 4 + 12v + 9*6
  // = 4 + 12v + 54
  // = 58 + 12v
  // mod 7: = 2 + 5v
  %ef_square = field.mul %ef_elem, %ef_elem : !EF2
  %sq_coeff0, %sq_coeff1 = field.ext_to_coeffs %ef_square : (!EF2) -> (!PF, !PF)
  %sq_coeff0_i32 = field.bitcast %sq_coeff0 : !PF -> i32
  %sq_coeff1_i32 = field.bitcast %sq_coeff1 : !PF -> i32

  %result5 = tensor.from_elements %sq_coeff0_i32, %sq_coeff1_i32 : tensor<2xi32>
  %memref5 = bufferization.to_buffer %result5 : tensor<2xi32> to memref<2xi32>
  %unranked5 = memref.cast %memref5 : memref<2xi32> to memref<*xi32>
  func.call @printMemrefI32(%unranked5) : (memref<*xi32>) -> ()

  return
}

// Test 1: PF [1, 2, 3, 4] -> EF2 [(1,2), (3,4)] coefficients
// CHECK_BITCAST: [1, 2, 3, 4]

// Test 2: EF2 [(1,2), (3,4)] -> PF [1, 2, 3, 4] round-trip
// CHECK_BITCAST: [1, 2, 3, 4]

// Test 3: PF [1, 2, 3, 4, 5, 6] -> EF3 [(1,2,3), (4,5,6)] coefficients
// CHECK_BITCAST: [1, 2, 3, 4, 5, 6]

// Test 4: (2 + 3v) + (2 + 3v) = (4 + 6v)
// CHECK_BITCAST: [4, 6]

// Test 5: (2 + 3v)^2 = 4 + 12v + 54 = 58 + 12v = 2 + 5v (mod 7)
// CHECK_BITCAST: [2, 5]
