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

// End-to-end test for the flat-basis AES field (`!field.bf<3, aes>`), GF(2^8)
// as GF(2)[x]/(x^8 + x^4 + x^3 + x + 1). Exercises the portable
// (shift-XOR, no CLMUL) lowering in BinaryFieldToArith.

// RUN: prime-ir-opt %s --field-to-llvm \
// RUN:   | mlir-runner -e main -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

!A = !field.bf<3, aes>   // GF(2^8), AES polynomial basis

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @print_aes(%v: !A) {
  %i = field.bitcast %v : !A -> i8
  %r = arith.extui %i : i8 to i32
  %t = tensor.from_elements %r : tensor<1xi32>
  %buf = bufferization.to_buffer %t : tensor<1xi32> to memref<1xi32>
  %cast = memref.cast %buf : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
  return
}

// 2 * 2 = 4: in the AES basis 2 = x, so 2*2 = x^2 = 4 (no reduction).
// (In the bf<3> tower this would be 3 — the bases are not bit-compatible.)
func.func @test_aes_mul() {
  %a = field.constant 2 : !A
  %b = field.constant 2 : !A
  %c = field.mul %a, %b : !A
  func.call @print_aes(%c) : (!A) -> ()
  return
}
// CHECK: [4]

// 1 * 42 = 42 (multiplicative identity).
func.func @test_aes_one() {
  %a = field.constant 1 : !A
  %b = field.constant 42 : !A
  %c = field.mul %a, %b : !A
  func.call @print_aes(%c) : (!A) -> ()
  return
}
// CHECK: [42]

// Reduction: x^7 * x = x^8 = x^4 + x^3 + x + 1 = 0x1B = 27.
func.func @test_aes_reduce() {
  %a = field.constant 128 : !A
  %b = field.constant 2 : !A
  %c = field.mul %a, %b : !A
  func.call @print_aes(%c) : (!A) -> ()
  return
}
// CHECK: [27]

// AES standard vector: {53} * {CA} = {01} (FIPS-197 §4.2 example inverses).
func.func @test_aes_fips_vector() {
  %a = field.constant 83 : !A     // 0x53
  %b = field.constant 202 : !A    // 0xCA
  %c = field.mul %a, %b : !A
  func.call @print_aes(%c) : (!A) -> ()
  return
}
// CHECK: [1]

// The linearized squaring must hit the same reduction as the multiply:
// (x^7)^2 = x^14 = 0x9A = 154 (2^2 = 4 never reduces).
func.func @test_aes_square_reduce() {
  %a = field.constant 128 : !A
  %c = field.square %a : !A
  func.call @print_aes(%c) : (!A) -> ()
  return
}
// CHECK: [154]

// inverse({53}) = {CA} — the FIPS-197 pair pins the Fermat chain to the basis.
func.func @test_aes_inverse_fips() {
  %a = field.constant 83 : !A     // 0x53
  %inv = field.inverse %a : !A
  func.call @print_aes(%inv) : (!A) -> ()
  return
}
// CHECK: [202]

// a * a⁻¹ = 1. The multiply is independently covered above, so a wrong
// inverse cannot cancel into a spurious 1.
func.func @test_aes_inverse_roundtrip() {
  %a = field.constant 42 : !A
  %inv = field.inverse %a : !A
  %p = field.mul %a, %inv : !A
  func.call @print_aes(%p) : (!A) -> ()
  return
}
// CHECK: [1]

// inverse(0) = 0: Fermat's 0^(2^8 − 2) keeps the tower lowering's 0 ↦ 0
// convention.
func.func @test_aes_inverse_zero() {
  %a = field.constant 0 : !A
  %inv = field.inverse %a : !A
  func.call @print_aes(%inv) : (!A) -> ()
  return
}
// CHECK: [0]

func.func @main() {
  func.call @test_aes_mul() : () -> ()
  func.call @test_aes_one() : () -> ()
  func.call @test_aes_reduce() : () -> ()
  func.call @test_aes_fips_vector() : () -> ()
  func.call @test_aes_square_reduce() : () -> ()
  func.call @test_aes_inverse_fips() : () -> ()
  func.call @test_aes_inverse_roundtrip() : () -> ()
  func.call @test_aes_inverse_zero() : () -> ()
  return
}
