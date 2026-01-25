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

// RUN: prime-ir-opt %s -tensor-ext-to-tensor -one-shot-bufferize -convert-bufferization-to-memref \
// RUN:   -convert-scf-to-cf -finalize-memref-to-llvm -convert-arith-to-llvm -convert-index-to-llvm \
// RUN:   -convert-cf-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN:   | mlir-runner -e test_bit_reverse_1d -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_1D < %t

// RUN: prime-ir-opt %s -tensor-ext-to-tensor -one-shot-bufferize -convert-bufferization-to-memref \
// RUN:   -convert-scf-to-cf -finalize-memref-to-llvm -convert-arith-to-llvm -convert-index-to-llvm \
// RUN:   -convert-cf-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN:   | mlir-runner -e test_bit_reverse_2d_dim0 -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_2D_DIM0 < %t

// RUN: prime-ir-opt %s -tensor-ext-to-tensor -one-shot-bufferize -convert-bufferization-to-memref \
// RUN:   -convert-scf-to-cf -finalize-memref-to-llvm -convert-arith-to-llvm -convert-index-to-llvm \
// RUN:   -convert-cf-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts \
// RUN:   | mlir-runner -e test_bit_reverse_2d_dim1 -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_2D_DIM1 < %t

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

// Test 1D bit-reverse (dimension 0)
// Input:  [0, 1, 2, 3, 4, 5, 6, 7]
// Bit-reversed indices for 8 elements (3 bits):
//   0 (000) -> 0 (000)
//   1 (001) -> 4 (100)
//   2 (010) -> 2 (010)
//   3 (011) -> 6 (110)
//   4 (100) -> 1 (001)
//   5 (101) -> 5 (101)
//   6 (110) -> 3 (011)
//   7 (111) -> 7 (111)
// Output: [0, 4, 2, 6, 1, 5, 3, 7]
func.func @test_bit_reverse_1d() {
  %input = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>
  %dest = bufferization.alloc_tensor() : tensor<8xi32>
  %result = tensor_ext.bit_reverse %input into %dest {dimension = 0 : i64} : tensor<8xi32>

  %buf = bufferization.to_buffer %result : tensor<8xi32> to memref<8xi32>
  %unranked = memref.cast %buf : memref<8xi32> to memref<*xi32>
  func.call @printMemrefI32(%unranked) : (memref<*xi32>) -> ()
  return
}
// CHECK_1D: [0, 4, 2, 6, 1, 5, 3, 7]

// Test 2D bit-reverse along dimension 0 (rows)
// Input (4x4):
//   row 0: [0,  1,  2,  3]
//   row 1: [4,  5,  6,  7]
//   row 2: [8,  9, 10, 11]
//   row 3: [12, 13, 14, 15]
// Bit-reversed row indices for 4 rows (2 bits):
//   0 (00) -> 0 (00)
//   1 (01) -> 2 (10)
//   2 (10) -> 1 (01)
//   3 (11) -> 3 (11)
// Output:
//   row 0: [0,  1,  2,  3]   (from row 0)
//   row 1: [8,  9, 10, 11]   (from row 2)
//   row 2: [4,  5,  6,  7]   (from row 1)
//   row 3: [12, 13, 14, 15]  (from row 3)
func.func @test_bit_reverse_2d_dim0() {
  %input = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>
  %dest = bufferization.alloc_tensor() : tensor<4x4xi32>
  %result = tensor_ext.bit_reverse %input into %dest {dimension = 0 : i64} : tensor<4x4xi32>

  %buf = bufferization.to_buffer %result : tensor<4x4xi32> to memref<4x4xi32>
  %unranked = memref.cast %buf : memref<4x4xi32> to memref<*xi32>
  func.call @printMemrefI32(%unranked) : (memref<*xi32>) -> ()
  return
}
// CHECK_2D_DIM0: [0, 1, 2, 3]
// CHECK_2D_DIM0: [8, 9, 10, 11]
// CHECK_2D_DIM0: [4, 5, 6, 7]
// CHECK_2D_DIM0: [12, 13, 14, 15]

// Test 2D bit-reverse along dimension 1 (columns)
// Input (4x4):
//   row 0: [0,  1,  2,  3]
//   row 1: [4,  5,  6,  7]
//   row 2: [8,  9, 10, 11]
//   row 3: [12, 13, 14, 15]
// Bit-reversed column indices for 4 columns (2 bits):
//   0 (00) -> 0 (00)
//   1 (01) -> 2 (10)
//   2 (10) -> 1 (01)
//   3 (11) -> 3 (11)
// Output:
//   row 0: [0,  2,  1,  3]
//   row 1: [4,  6,  5,  7]
//   row 2: [8, 10,  9, 11]
//   row 3: [12, 14, 13, 15]
func.func @test_bit_reverse_2d_dim1() {
  %input = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi32>
  %dest = bufferization.alloc_tensor() : tensor<4x4xi32>
  %result = tensor_ext.bit_reverse %input into %dest {dimension = 1 : i64} : tensor<4x4xi32>

  %buf = bufferization.to_buffer %result : tensor<4x4xi32> to memref<4x4xi32>
  %unranked = memref.cast %buf : memref<4x4xi32> to memref<*xi32>
  func.call @printMemrefI32(%unranked) : (memref<*xi32>) -> ()
  return
}
// CHECK_2D_DIM1: [0, 2, 1, 3]
// CHECK_2D_DIM1: [4, 6, 5, 7]
// CHECK_2D_DIM1: [8, 10, 9, 11]
// CHECK_2D_DIM1: [12, 14, 13, 15]
