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

// End-to-end test for the flat-basis GHASH field (`!field.bf<7, ghash>`), GF(2^128)
// as GF(2)[x]/(x^128 + x^7 + x^2 + x + 1). Exercises the portable
// (shift-XOR, no CLMUL) lowering in BinaryFieldToArith.

// RUN: prime-ir-opt %s --field-to-llvm \
// RUN:   | mlir-runner -e main -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

!G = !field.bf<7, ghash>   // GF(2^128), GHASH polynomial basis

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

// 2 * 3 = 6: in the GHASH basis 2 = x and 3 = x + 1, so 2*3 = x^2 + x = 6.
// (In the bf<7> tower this would be 1 — the bases are not bit-compatible.)
func.func @test_ghash_mul() {
  %a = field.constant 2 : !G
  %b = field.constant 3 : !G
  %c = field.mul %a, %b : !G

  %i = field.bitcast %c : !G -> i128
  %r = arith.trunci %i : i128 to i32
  %t = tensor.from_elements %r : tensor<1xi32>
  %buf = bufferization.to_buffer %t : tensor<1xi32> to memref<1xi32>
  %cast = memref.cast %buf : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
  return
}
// CHECK: [6]

// 2 * 2 = 4 (x * x = x^2).
func.func @test_ghash_square() {
  %a = field.constant 2 : !G
  %c = field.square %a : !G

  %i = field.bitcast %c : !G -> i128
  %r = arith.trunci %i : i128 to i32
  %t = tensor.from_elements %r : tensor<1xi32>
  %buf = bufferization.to_buffer %t : tensor<1xi32> to memref<1xi32>
  %cast = memref.cast %buf : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
  return
}
// CHECK: [4]

// 1 * x = x (multiplicative identity).
func.func @test_ghash_one() {
  %a = field.constant 1 : !G
  %b = field.constant 12345 : !G
  %c = field.mul %a, %b : !G

  %i = field.bitcast %c : !G -> i128
  %r = arith.trunci %i : i128 to i32
  %t = tensor.from_elements %r : tensor<1xi32>
  %buf = bufferization.to_buffer %t : tensor<1xi32> to memref<1xi32>
  %cast = memref.cast %buf : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
  return
}
// CHECK: [12345]

// Reduction: x^64 * x^64 = x^128 = x^7 + x^2 + x + 1 = 0x87 = 135.
// x^64 = 2^64 doesn't fit field.constant's int64 literal, so build it as an
// i128 and bitcast into the field.
func.func @test_ghash_reduce() {
  %c64 = arith.constant 18446744073709551616 : i128   // 2^64 = x^64
  %x64 = field.bitcast %c64 : i128 -> !G
  %p = field.mul %x64, %x64 : !G

  %i = field.bitcast %p : !G -> i128
  %r = arith.trunci %i : i128 to i32
  %t = tensor.from_elements %r : tensor<1xi32>
  %buf = bufferization.to_buffer %t : tensor<1xi32> to memref<1xi32>
  %cast = memref.cast %buf : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
  return
}
// CHECK: [135]

func.func @main() {
  func.call @test_ghash_mul() : () -> ()
  func.call @test_ghash_square() : () -> ()
  func.call @test_ghash_one() : () -> ()
  func.call @test_ghash_reduce() : () -> ()
  return
}
