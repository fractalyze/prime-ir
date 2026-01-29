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

// RUN: cat %S/../../default_print_utils.mlir %S/../../bn254_field_defs.mlir %S/../../bn254_ec_mont_defs.mlir %S/../../bn254_ec_utils.mlir %s \
// RUN:   | prime-ir-opt -elliptic-curve-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_bitcast -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext,%S/../../libruntime_functions%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_BITCAST < %t

// Test elliptic_curve.constant and elliptic_curve.bitcast operations
// Uses Montgomery form field types from bn254_ec_mont_defs.mlir

func.func @test_bitcast() {
  // Index constants for tensor element extraction
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index

  // ============================================================
  // Test 1: G1 jacobian tensor constant -> field tensor bitcast
  // tensor<2x!jacobian> with points (1,2,1) and (3,4,1)
  // ============================================================
  %g1_jacobian = elliptic_curve.constant dense<[[1, 2, 1], [3, 4, 1]]> : tensor<2x!jacobian>
  %g1_fields = elliptic_curve.bitcast %g1_jacobian : tensor<2x!jacobian> -> tensor<6x!PFm>

  // Extract all 6 field elements and print as i256
  %f0 = tensor.extract %g1_fields[%c0] : tensor<6x!PFm>
  %f1 = tensor.extract %g1_fields[%c1] : tensor<6x!PFm>
  %f2 = tensor.extract %g1_fields[%c2] : tensor<6x!PFm>
  %f3 = tensor.extract %g1_fields[%c3] : tensor<6x!PFm>
  %f4 = tensor.extract %g1_fields[%c4] : tensor<6x!PFm>
  %f5 = tensor.extract %g1_fields[%c5] : tensor<6x!PFm>

  %i0 = field.bitcast %f0 : !PFm -> i256
  %i1 = field.bitcast %f1 : !PFm -> i256
  %i2 = field.bitcast %f2 : !PFm -> i256
  %i3 = field.bitcast %f3 : !PFm -> i256
  %i4 = field.bitcast %f4 : !PFm -> i256
  %i5 = field.bitcast %f5 : !PFm -> i256

  %result1 = tensor.from_elements %i0, %i1, %i2, %i3, %i4, %i5 : tensor<6xi256>
  %memref1 = bufferization.to_buffer %result1 : tensor<6xi256> to memref<6xi256>
  %unranked1 = memref.cast %memref1 : memref<6xi256> to memref<*xi256>
  func.call @printMemrefI256(%unranked1) : (memref<*xi256>) -> ()

  // ============================================================
  // Test 2: G1 affine tensor constant -> field tensor bitcast
  // tensor<2x!affine> with points (1,2) and (3,4)
  // ============================================================
  %g1_affine = elliptic_curve.constant dense<[[1, 2], [3, 4]]> : tensor<2x!affine>
  %g1_affine_fields = elliptic_curve.bitcast %g1_affine : tensor<2x!affine> -> tensor<4x!PFm>

  %af0 = tensor.extract %g1_affine_fields[%c0] : tensor<4x!PFm>
  %af1 = tensor.extract %g1_affine_fields[%c1] : tensor<4x!PFm>
  %af2 = tensor.extract %g1_affine_fields[%c2] : tensor<4x!PFm>
  %af3 = tensor.extract %g1_affine_fields[%c3] : tensor<4x!PFm>

  %ai0 = field.bitcast %af0 : !PFm -> i256
  %ai1 = field.bitcast %af1 : !PFm -> i256

  %ai2 = field.bitcast %af2 : !PFm -> i256
  %ai3 = field.bitcast %af3 : !PFm -> i256

  %result2 = tensor.from_elements %ai0, %ai1, %ai2, %ai3 : tensor<4xi256>
  %memref2 = bufferization.to_buffer %result2 : tensor<4xi256> to memref<4xi256>
  %unranked2 = memref.cast %memref2 : memref<4xi256> to memref<*xi256>
  func.call @printMemrefI256(%unranked2) : (memref<*xi256>) -> ()

  // ============================================================
  // Test 3: G2 jacobian tensor constant -> field tensor bitcast
  // tensor<2x!g2jacobian> with points ((1+2i, 3+4i, 1+0i), (5+6i, 7+8i, 1+0i))
  // G2 jacobian: 3 coords * 2 degree = 6 prime field elements per point
  // Total: 2 points * 6 = 12 prime field elements
  // ============================================================
  %g2_jacobian = elliptic_curve.constant dense<[[[1, 2], [3, 4], [1, 0]], [[5, 6], [7, 8], [1, 0]]]> : tensor<2x!g2jacobian>
  %g2_fields = elliptic_curve.bitcast %g2_jacobian : tensor<2x!g2jacobian> -> tensor<12x!PFm>

  %g2f0 = tensor.extract %g2_fields[%c0] : tensor<12x!PFm>
  %g2f1 = tensor.extract %g2_fields[%c1] : tensor<12x!PFm>
  %g2f2 = tensor.extract %g2_fields[%c2] : tensor<12x!PFm>
  %g2f3 = tensor.extract %g2_fields[%c3] : tensor<12x!PFm>
  %g2f4 = tensor.extract %g2_fields[%c4] : tensor<12x!PFm>
  %g2f5 = tensor.extract %g2_fields[%c5] : tensor<12x!PFm>
  %g2f6 = tensor.extract %g2_fields[%c6] : tensor<12x!PFm>
  %g2f7 = tensor.extract %g2_fields[%c7] : tensor<12x!PFm>
  %g2f8 = tensor.extract %g2_fields[%c8] : tensor<12x!PFm>
  %g2f9 = tensor.extract %g2_fields[%c9] : tensor<12x!PFm>
  %g2f10 = tensor.extract %g2_fields[%c10] : tensor<12x!PFm>
  %g2f11 = tensor.extract %g2_fields[%c11] : tensor<12x!PFm>

  %g2i0 = field.bitcast %g2f0 : !PFm -> i256
  %g2i1 = field.bitcast %g2f1 : !PFm -> i256
  %g2i2 = field.bitcast %g2f2 : !PFm -> i256
  %g2i3 = field.bitcast %g2f3 : !PFm -> i256
  %g2i4 = field.bitcast %g2f4 : !PFm -> i256
  %g2i5 = field.bitcast %g2f5 : !PFm -> i256
  %g2i6 = field.bitcast %g2f6 : !PFm -> i256
  %g2i7 = field.bitcast %g2f7 : !PFm -> i256
  %g2i8 = field.bitcast %g2f8 : !PFm -> i256
  %g2i9 = field.bitcast %g2f9 : !PFm -> i256
  %g2i10 = field.bitcast %g2f10 : !PFm -> i256
  %g2i11 = field.bitcast %g2f11 : !PFm -> i256

  %result3 = tensor.from_elements %g2i0, %g2i1, %g2i2, %g2i3, %g2i4, %g2i5, %g2i6, %g2i7, %g2i8, %g2i9, %g2i10, %g2i11 : tensor<12xi256>
  %memref3 = bufferization.to_buffer %result3 : tensor<12xi256> to memref<12xi256>
  %unranked3 = memref.cast %memref3 : memref<12xi256> to memref<*xi256>
  func.call @printMemrefI256(%unranked3) : (memref<*xi256>) -> ()

  return
}

// Test 1: G1 jacobian [[1,2,1], [3,4,1]] -> field tensor [1,2,1,3,4,1]
// CHECK_BITCAST: [1, 2, 1, 3, 4, 1]

// Test 2: G1 affine [[1,2], [3,4]] -> field tensor [1,2,3,4]
// CHECK_BITCAST: [1, 2, 3, 4]

// Test 3: G2 jacobian [[[1,2],[3,4],[1,0]], [[5,6],[7,8],[1,0]]] -> field tensor
// Each G2 point has 3 coords, each coord has 2 prime field elements
// Point 1: (1,2), (3,4), (1,0) -> [1,2,3,4,1,0]
// Point 2: (5,6), (7,8), (1,0) -> [5,6,7,8,1,0]
// CHECK_BITCAST: [1, 2, 3, 4, 1, 0, 5, 6, 7, 8, 1, 0]
