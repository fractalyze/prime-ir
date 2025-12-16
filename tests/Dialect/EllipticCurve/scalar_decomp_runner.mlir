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

// RUN: cat %S/../../default_print_utils.mlir %S/../../bn254_field_defs.mlir %S/../../bn254_ec_mont_defs.mlir %S/../../bn254_ec_utils.mlir %s \
// RUN:   | zkir-opt -elliptic-curve-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_scalar_decomp -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext,%S/../../libruntime_functions%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_SCALAR_DECOMP < %t

// RUN: cat %S/../../default_print_utils.mlir %S/../../bn254_field_defs.mlir %S/../../bn254_ec_mont_defs.mlir %S/../../bn254_ec_utils.mlir %s \
// RUN:   | zkir-opt -elliptic-curve-to-field -field-to-gpu=parallelize-affine \
// RUN:   | mlir-runner -e test_scalar_decomp -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext,%S/../../libruntime_functions%shlibext,%mlir_lib_dir/libmlir_cuda_runtime%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_SCALAR_DECOMP < %t

func.func @test_scalar_decomp() {
  %c3 = field.constant 3 : !SF
  %c5 = field.constant 5 : !SF
  %c7 = field.constant 7 : !SF

  %mont3 = field.to_mont %c3 : !SFm
  %mont5 = field.to_mont %c5 : !SFm
  %mont7 = field.to_mont %c7 : !SFm

  %scalars_mont = tensor.from_elements %mont3, %mont5, %mont7 : tensor<3x!SFm>

  %scalars = field.from_mont %scalars_mont : tensor<3x!SF>

  %bucket_indices, %point_indices = elliptic_curve.scalar_decomp %scalars {bitsPerWindow = 2 : i16, scalarMaxBits = 4 : i16} : (tensor<3x!SF>) -> (tensor<6xindex>, tensor<6xindex>)

  %bucket_indices_i256 = arith.index_cast %bucket_indices : tensor<6xindex> to tensor<6xi256>
  %point_indices_i256 = arith.index_cast %point_indices : tensor<6xindex> to tensor<6xi256>

  %bucket_mem = bufferization.to_buffer %bucket_indices_i256 : tensor<6xi256> to memref<6xi256>
  %bucket_mem_cast = memref.cast %bucket_mem : memref<6xi256> to memref<*xi256>
  func.call @printMemrefI256(%bucket_mem_cast) : (memref<*xi256>) -> ()

  %point_mem = bufferization.to_buffer %point_indices_i256 : tensor<6xi256> to memref<6xi256>
  %point_mem_cast = memref.cast %point_mem : memref<6xi256> to memref<*xi256>
  func.call @printMemrefI256(%point_mem_cast) : (memref<*xi256>) -> ()

  return
}

// CHECK_TEST_SCALAR_DECOMP: [3, 0, 1, 5, 3, 5]
// CHECK_TEST_SCALAR_DECOMP: [0, 0, 1, 1, 2, 2]
