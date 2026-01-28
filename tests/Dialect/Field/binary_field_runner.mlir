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

// End-to-end test for binary field operations
// Tests BF8 (GF(2⁸)) arithmetic using the field-to-llvm pipeline

// RUN: prime-ir-opt %s --field-to-llvm \
// RUN:   | mlir-runner -e main -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

!BF8 = !field.bf<3>  // GF(2^8) tower field

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

// Test: BF8 addition (should be XOR in characteristic 2)
// 5 + 3 = 5 XOR 3 = 6
func.func @test_bf8_add() {
  %a = field.constant 5 : !BF8
  %b = field.constant 3 : !BF8
  %c = field.add %a, %b : !BF8

  %result_i8 = field.bitcast %c : !BF8 -> i8
  %result = arith.extui %result_i8 : i8 to i32
  %tensor = tensor.from_elements %result : tensor<1xi32>
  %buffer = bufferization.to_buffer %tensor : tensor<1xi32> to memref<1xi32>
  %cast = memref.cast %buffer : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
  return
}
// CHECK: [6]

// Test: BF8 multiplication using tower field arithmetic
// In tower field GF(2^8) with tower construction:
//   Level 1: X² + X + 1 (α = 1), field GF(4)
//   Level 2: X² + X + 2 (α = 2), field GF(16)
//   Level 3: X² + X + 8 (α = 8), field GF(256)
// In GF(4): 2 represents ω, 3 represents ω+1
//   ω * (ω+1) = ω² + ω = (ω+1) + ω = 1
// Since 2 and 3 embed in lower half of GF(256): 2 * 3 = 1
func.func @test_bf8_mul() {
  %a = field.constant 2 : !BF8
  %b = field.constant 3 : !BF8
  %c = field.mul %a, %b : !BF8

  %result_i8 = field.bitcast %c : !BF8 -> i8
  %result = arith.extui %result_i8 : i8 to i32
  %tensor = tensor.from_elements %result : tensor<1xi32>
  %buffer = bufferization.to_buffer %tensor : tensor<1xi32> to memref<1xi32>
  %cast = memref.cast %buffer : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
  return
}
// CHECK: [1]

// Test: BF8 squaring
// In GF(4): 3 = ω+1, and (ω+1)² = ω² + 1 = (ω+1) + 1 = ω = 2
// Since 3 embeds in lower half of GF(256): 3² = 2
func.func @test_bf8_square() {
  %a = field.constant 3 : !BF8
  %c = field.square %a : !BF8

  %result_i8 = field.bitcast %c : !BF8 -> i8
  %result = arith.extui %result_i8 : i8 to i32
  %tensor = tensor.from_elements %result : tensor<1xi32>
  %buffer = bufferization.to_buffer %tensor : tensor<1xi32> to memref<1xi32>
  %cast = memref.cast %buffer : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
  return
}
// CHECK: [2]

// Test: BF8 double (should be 0 in characteristic 2)
// 2 * x = x + x = 0 for all x in GF(2ⁿ)
func.func @test_bf8_double() {
  %a = field.constant 42 : !BF8
  %c = field.double %a : !BF8

  %result_i8 = field.bitcast %c : !BF8 -> i8
  %result = arith.extui %result_i8 : i8 to i32
  %tensor = tensor.from_elements %result : tensor<1xi32>
  %buffer = bufferization.to_buffer %tensor : tensor<1xi32> to memref<1xi32>
  %cast = memref.cast %buffer : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
  return
}
// CHECK: [0]

func.func @main() {
  func.call @test_bf8_add() : () -> ()
  func.call @test_bf8_mul() : () -> ()
  func.call @test_bf8_square() : () -> ()
  func.call @test_bf8_double() : () -> ()
  return
}
