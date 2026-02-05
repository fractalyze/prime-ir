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

// End-to-end test for binary field operations using ARM PMULL
// Tests bf<6> (64-bit) and bf<7> (128-bit) multiplication using carryless multiply
// Requires: ARM CPU with PMULL support (ARMv8 Crypto extensions)

// REQUIRES: pmull

// RUN: prime-ir-opt %s \
// RUN:   --field-to-llvm="specialize-pmull=true" \
// RUN:   | mlir-runner -e main -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

!BF64 = !field.bf<6>  // GF(2^64) tower field
!BF128 = !field.bf<7>  // GF(2^128) tower field

func.func private @printMemrefI64(memref<*xi64>) attributes { llvm.emit_c_interface }

// Test: BF64 multiplication using PMULL
// In GF(2^64) tower: 2 * 3 = 1 (same as in GF(2^8) since they embed in lower bits)
func.func @test_bf64_mul() {
  %a = field.constant 2 : !BF64
  %b = field.constant 3 : !BF64
  %c = field.mul %a, %b : !BF64

  // Use field.bitcast to convert to integer for printing
  %result_i64 = field.bitcast %c : !BF64 -> i64
  %tensor = tensor.from_elements %result_i64 : tensor<1xi64>
  %buffer = bufferization.to_buffer %tensor : tensor<1xi64> to memref<1xi64>
  %cast = memref.cast %buffer : memref<1xi64> to memref<*xi64>
  func.call @printMemrefI64(%cast) : (memref<*xi64>) -> ()
  return
}
// CHECK: [1]

// Test: BF64 squaring
// 3^2 = 2 in tower field (since 3 = w+1, (w+1)^2 = w^2 + 1 = w = 2)
func.func @test_bf64_square() {
  %a = field.constant 3 : !BF64
  %c = field.square %a : !BF64

  %result_i64 = field.bitcast %c : !BF64 -> i64
  %tensor = tensor.from_elements %result_i64 : tensor<1xi64>
  %buffer = bufferization.to_buffer %tensor : tensor<1xi64> to memref<1xi64>
  %cast = memref.cast %buffer : memref<1xi64> to memref<*xi64>
  func.call @printMemrefI64(%cast) : (memref<*xi64>) -> ()
  return
}
// CHECK: [2]

// Test: BF64 multiplication with identity
// 1 * x = x for all x
func.func @test_bf64_mul_identity() {
  %a = field.constant 1 : !BF64
  %b = field.constant 42 : !BF64
  %c = field.mul %a, %b : !BF64

  %result_i64 = field.bitcast %c : !BF64 -> i64
  %tensor = tensor.from_elements %result_i64 : tensor<1xi64>
  %buffer = bufferization.to_buffer %tensor : tensor<1xi64> to memref<1xi64>
  %cast = memref.cast %buffer : memref<1xi64> to memref<*xi64>
  func.call @printMemrefI64(%cast) : (memref<*xi64>) -> ()
  return
}
// CHECK: [42]

// Test: BF64 multiplication with zero
// 0 * x = 0 for all x
func.func @test_bf64_mul_zero() {
  %a = field.constant 0 : !BF64
  %b = field.constant 12345 : !BF64
  %c = field.mul %a, %b : !BF64

  %result_i64 = field.bitcast %c : !BF64 -> i64
  %tensor = tensor.from_elements %result_i64 : tensor<1xi64>
  %buffer = bufferization.to_buffer %tensor : tensor<1xi64> to memref<1xi64>
  %cast = memref.cast %buffer : memref<1xi64> to memref<*xi64>
  func.call @printMemrefI64(%cast) : (memref<*xi64>) -> ()
  return
}
// CHECK: [0]

// Test: BF128 multiplication using PMULL
// In GF(2^128) tower: 2 * 3 = 1 (same as lower levels since they embed in lower bits)
func.func @test_bf128_mul() {
  %a = field.constant 2 : !BF128
  %b = field.constant 3 : !BF128
  %c = field.mul %a, %b : !BF128

  // Bitcast to i128 and truncate to i64 for printing (result fits in 64 bits)
  %result_i128 = field.bitcast %c : !BF128 -> i128
  %result_i64 = arith.trunci %result_i128 : i128 to i64
  %tensor = tensor.from_elements %result_i64 : tensor<1xi64>
  %buffer = bufferization.to_buffer %tensor : tensor<1xi64> to memref<1xi64>
  %cast = memref.cast %buffer : memref<1xi64> to memref<*xi64>
  func.call @printMemrefI64(%cast) : (memref<*xi64>) -> ()
  return
}
// CHECK: [1]

// Test: BF128 multiplication with identity
// 1 * x = x for all x
func.func @test_bf128_mul_identity() {
  %a = field.constant 1 : !BF128
  %b = field.constant 42 : !BF128
  %c = field.mul %a, %b : !BF128

  %result_i128 = field.bitcast %c : !BF128 -> i128
  %result_i64 = arith.trunci %result_i128 : i128 to i64
  %tensor = tensor.from_elements %result_i64 : tensor<1xi64>
  %buffer = bufferization.to_buffer %tensor : tensor<1xi64> to memref<1xi64>
  %cast = memref.cast %buffer : memref<1xi64> to memref<*xi64>
  func.call @printMemrefI64(%cast) : (memref<*xi64>) -> ()
  return
}
// CHECK: [42]

// Test: BF128 multiplication with zero
// 0 * x = 0 for all x
func.func @test_bf128_mul_zero() {
  %a = field.constant 0 : !BF128
  %b = field.constant 12345 : !BF128
  %c = field.mul %a, %b : !BF128

  %result_i128 = field.bitcast %c : !BF128 -> i128
  %result_i64 = arith.trunci %result_i128 : i128 to i64
  %tensor = tensor.from_elements %result_i64 : tensor<1xi64>
  %buffer = bufferization.to_buffer %tensor : tensor<1xi64> to memref<1xi64>
  %cast = memref.cast %buffer : memref<1xi64> to memref<*xi64>
  func.call @printMemrefI64(%cast) : (memref<*xi64>) -> ()
  return
}
// CHECK: [0]

func.func @main() {
  func.call @test_bf64_mul() : () -> ()
  func.call @test_bf64_square() : () -> ()
  func.call @test_bf64_mul_identity() : () -> ()
  func.call @test_bf64_mul_zero() : () -> ()
  func.call @test_bf128_mul() : () -> ()
  func.call @test_bf128_mul_identity() : () -> ()
  func.call @test_bf128_mul_zero() : () -> ()
  return
}
