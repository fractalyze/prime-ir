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

// End-to-end numeric test for the narrow flat bases `bf<4|5, flat>` —
// GF(2)[y]/(f) with the TowerFlatBasis.h moduli — through the portable
// lowering. Expected values computed by tools/derive_tower_flat_basis.py's
// mul_flat/pow_flat (which the same file proves against the tower reference
// and the clmad fold schedule).

// RUN: prime-ir-opt %s --field-to-llvm \
// RUN:   | mlir-runner -e main -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

!F16 = !field.bf<4, flat>  // GF(2^16), y^16 + y^5 + y^3 + y + 1
!F32 = !field.bf<5, flat>  // GF(2^32), y^32 + y^7 + y^3 + y^2 + 1

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func private @print_i32(%v: i32) {
  %tensor = tensor.from_elements %v : tensor<1xi32>
  %buffer = bufferization.to_buffer %tensor : tensor<1xi32> to memref<1xi32>
  %cast = memref.cast %buffer : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
  return
}

// 0x1234 * 0xabcd = 0x1d05 (= 7429) in GF(2^16) flat.
func.func @test_f16_mul() {
  %a = field.constant 0x1234 : !F16
  %b = field.constant 0xabcd : !F16
  %c = field.mul %a, %b : !F16
  %c16 = field.bitcast %c : !F16 -> i16
  %c32 = arith.extui %c16 : i16 to i32
  func.call @print_i32(%c32) : (i32) -> ()
  return
}
// CHECK: {{^}}[7429]

// 0x12345678 * 0x9abcdef0 = 0x717b52d0 (= 1903907536) in GF(2^32) flat.
func.func @test_f32_mul() {
  %a = field.constant 0x12345678 : !F32
  %b = field.constant 0x9abcdef0 : !F32
  %c = field.mul %a, %b : !F32
  %c32 = field.bitcast %c : !F32 -> i32
  func.call @print_i32(%c32) : (i32) -> ()
  return
}
// CHECK: {{^}}[1903907536]

// square(0x12345678) = 0x9e22a490; printed as the i32 bit pattern
// (-1641896816 in two's complement is 2653070480 unsigned; extui-free print
// keeps the signed rendering, so compare the signed value).
func.func @test_f32_square() {
  %a = field.constant 0x12345678 : !F32
  %c = field.square %a : !F32
  %c32 = field.bitcast %c : !F32 -> i32
  func.call @print_i32(%c32) : (i32) -> ()
  return
}
// CHECK: {{^}}[-1641896816]

// a * a^-1 = 1 exercises inverse, square, and mul together.
func.func @test_f32_inverse_roundtrip() {
  %a = field.constant 0x12345678 : !F32
  %inv = field.inverse %a : !F32
  %one = field.mul %a, %inv : !F32
  %c32 = field.bitcast %one : !F32 -> i32
  func.call @print_i32(%c32) : (i32) -> ()
  return
}
// CHECK: {{^}}[1]

func.func @main() {
  func.call @test_f16_mul() : () -> ()
  func.call @test_f32_mul() : () -> ()
  func.call @test_f32_square() : () -> ()
  func.call @test_f32_inverse_roundtrip() : () -> ()
  return
}
