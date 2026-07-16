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

!BF8 = !field.bf<3>     // GF(2^8) tower field
!BF64 = !field.bf<6>    // GF(2^64) tower field
!BF128 = !field.bf<7>   // GF(2^128) tower field

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
// CHECK: {{^}}[6]

// Test: BF8 multiplication using tower field arithmetic.
// Fan-Paar/Binius canonical tower GF(2^8): each level GF(2^(2ᵏ)) =
// subfield[X]/(X² + βₖ₋₁·X + 1), βₖ₋₁ the subfield generator.
// 2 and 3 live in the GF(4) subfield (β₀ = 1 ⇒ X² + X + 1, unchanged across
// towers): 2 = ω, 3 = ω+1, ω·(ω+1) = ω²+ω = (ω+1)+ω = 1. So 2 * 3 = 1.
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
// CHECK: {{^}}[1]

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
// CHECK: {{^}}[2]

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
// CHECK: {{^}}[0]

// Test: BF8 self-multiply. 3 = ω+1 in the GF(4) subfield, so 3*3 = (ω+1)² =
// ω = 2 (= square(3)). Subfield-only, so tower-independent.
func.func @test_bf8_mul_self() {
  %a = field.constant 3 : !BF8
  %c = field.mul %a, %a : !BF8

  %result_i8 = field.bitcast %c : !BF8 -> i8
  %result = arith.extui %result_i8 : i8 to i32
  %tensor = tensor.from_elements %result : tensor<1xi32>
  %buffer = bufferization.to_buffer %tensor : tensor<1xi32> to memref<1xi32>
  %cast = memref.cast %buffer : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
  return
}
// CHECK: {{^}}[2]

// Test: BF8 multiply where operands span tower level ≥ 2, so the result
// distinguishes the Fan-Paar tower from the previous x²+x+α tower. 5 spans
// GF(16); 5 * 3 = 15 (0x0f) in the canonical tower (the old tower gave 6).
// This is the case that a bare zk_dtypes bump silently broke — every other
// case here is subfield-only and agrees across towers.
func.func @test_bf8_mul_cross() {
  %a = field.constant 5 : !BF8
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
// CHECK: {{^}}[15]

// Test: BF8 square spanning level ≥ 2. 5² = 8 in the canonical tower.
func.func @test_bf8_square_cross() {
  %a = field.constant 5 : !BF8
  %c = field.square %a : !BF8

  %result_i8 = field.bitcast %c : !BF8 -> i8
  %result = arith.extui %result_i8 : i8 to i32
  %tensor = tensor.from_elements %result : tensor<1xi32>
  %buffer = bufferization.to_buffer %tensor : tensor<1xi32> to memref<1xi32>
  %cast = memref.cast %buffer : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
  return
}
// CHECK: {{^}}[8]

// Test: BF128 (GF(2^128)) multiplication on the portable tower path.
// 2 and 3 embed in the GF(4) subfield, so 2 * 3 = 1 at every tower level.
func.func @test_bf128_mul() {
  %a = field.constant 2 : !BF128
  %b = field.constant 3 : !BF128
  %c = field.mul %a, %b : !BF128

  %result_i128 = field.bitcast %c : !BF128 -> i128
  %result = arith.trunci %result_i128 : i128 to i32
  %tensor = tensor.from_elements %result : tensor<1xi32>
  %buffer = bufferization.to_buffer %tensor : tensor<1xi32> to memref<1xi32>
  %cast = memref.cast %buffer : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
  return
}
// CHECK: {{^}}[1]

// Test: BF128 self-multiply — 3 * 3 = 2 (zero-divisor guard at the top level).
func.func @test_bf128_mul_self() {
  %a = field.constant 3 : !BF128
  %c = field.mul %a, %a : !BF128

  %result_i128 = field.bitcast %c : !BF128 -> i128
  %result = arith.trunci %result_i128 : i128 to i32
  %tensor = tensor.from_elements %result : tensor<1xi32>
  %buffer = bufferization.to_buffer %tensor : tensor<1xi32> to memref<1xi32>
  %cast = memref.cast %buffer : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
  return
}
// CHECK: {{^}}[2]

// Test: BF128 multiply spanning level ≥ 2 (5 and 3 embed in GF(16) ⊂ GF(2^128),
// so 5 * 3 = 15 at the top level too — guards the full recursive tower).
func.func @test_bf128_mul_cross() {
  %a = field.constant 5 : !BF128
  %b = field.constant 3 : !BF128
  %c = field.mul %a, %b : !BF128

  %result_i128 = field.bitcast %c : !BF128 -> i128
  %result = arith.trunci %result_i128 : i128 to i32
  %tensor = tensor.from_elements %result : tensor<1xi32>
  %buffer = bufferization.to_buffer %tensor : tensor<1xi32> to memref<1xi32>
  %cast = memref.cast %buffer : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
  return
}
// CHECK: {{^}}[15]

// Test: BF64 inverse via tower descent. 2 = ω lives in the GF(4) subfield
// (invariant across tower levels), and ω·(ω+1) = 1, so 2⁻¹ = 3 at every
// level — an exact known value that exercises the full descent to the
// level-3 lookup base.
func.func @test_bf64_inverse_known() {
  %a = field.constant 2 : !BF64
  %c = field.inverse %a : !BF64

  %result_i64 = field.bitcast %c : !BF64 -> i64
  %result = arith.trunci %result_i64 : i64 to i32
  %tensor = tensor.from_elements %result : tensor<1xi32>
  %buffer = bufferization.to_buffer %tensor : tensor<1xi32> to memref<1xi32>
  %cast = memref.cast %buffer : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
  return
}
// CHECK: {{^}}[3]

// Test: BF64 full-width inverse round-trip — x·x⁻¹ = 1 for an x that fills
// all 64 bits, so every descent level does real work. Prints the low i32 of
// the product (1) then the folded-down high bits (0).
func.func @test_bf64_inverse_roundtrip() {
  %x = field.constant 0xDEADBEEFCAFEBABE : !BF64
  %xinv = field.inverse %x : !BF64
  %p = field.mul %x, %xinv : !BF64

  %p_i64 = field.bitcast %p : !BF64 -> i64
  %lo = arith.trunci %p_i64 : i64 to i32
  %c32 = arith.constant 32 : i64
  %hi64 = arith.shrui %p_i64, %c32 : i64
  %hi = arith.trunci %hi64 : i64 to i32
  %tensor = tensor.from_elements %lo, %hi : tensor<2xi32>
  %buffer = bufferization.to_buffer %tensor : tensor<2xi32> to memref<2xi32>
  %cast = memref.cast %buffer : memref<2xi32> to memref<*xi32>
  func.call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
  return
}
// CHECK: {{^}}[1, 0]

// Test: BF128 inverse known value (same GF(4) embedding as BF64 above).
func.func @test_bf128_inverse_known() {
  %a = field.constant 2 : !BF128
  %c = field.inverse %a : !BF128

  %result_i128 = field.bitcast %c : !BF128 -> i128
  %result = arith.trunci %result_i128 : i128 to i32
  %tensor = tensor.from_elements %result : tensor<1xi32>
  %buffer = bufferization.to_buffer %tensor : tensor<1xi32> to memref<1xi32>
  %cast = memref.cast %buffer : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
  return
}
// CHECK: {{^}}[3]

// Test: BF128 full-width inverse round-trip — prints all four 32-bit limbs
// of x·x⁻¹ independently (an XOR-fold of the upper limbs could false-pass,
// e.g. [1, e, e, 0]).
func.func @test_bf128_inverse_roundtrip() {
  %x = field.constant 0x0123456789ABCDEFFEDCBA9876543210 : !BF128
  %xinv = field.inverse %x : !BF128
  %p = field.mul %x, %xinv : !BF128

  %p_i128 = field.bitcast %p : !BF128 -> i128
  %lo = arith.trunci %p_i128 : i128 to i32
  %c32 = arith.constant 32 : i128
  %c64 = arith.constant 64 : i128
  %c96 = arith.constant 96 : i128
  %s1 = arith.shrui %p_i128, %c32 : i128
  %s2 = arith.shrui %p_i128, %c64 : i128
  %s3 = arith.shrui %p_i128, %c96 : i128
  %h1 = arith.trunci %s1 : i128 to i32
  %h2 = arith.trunci %s2 : i128 to i32
  %h3 = arith.trunci %s3 : i128 to i32
  %tensor = tensor.from_elements %lo, %h1, %h2, %h3 : tensor<4xi32>
  %buffer = bufferization.to_buffer %tensor : tensor<4xi32> to memref<4xi32>
  %cast = memref.cast %buffer : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
  return
}
// CHECK: {{^}}[1, 0, 0, 0]

// Test: BF64 inverse of zero is zero (invert_or_zero, matching binius and
// the level-3 lookup table's 0 → 0 entry).
func.func @test_bf64_inverse_zero() {
  %a = field.constant 0 : !BF64
  %c = field.inverse %a : !BF64

  %result_i64 = field.bitcast %c : !BF64 -> i64
  %result = arith.trunci %result_i64 : i64 to i32
  %tensor = tensor.from_elements %result : tensor<1xi32>
  %buffer = bufferization.to_buffer %tensor : tensor<1xi32> to memref<1xi32>
  %cast = memref.cast %buffer : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
  return
}
// CHECK: {{^}}[0]

// Test: BF128 inverse of zero is zero — one more descent level of the
// norm-zero propagation than the BF64 case.
func.func @test_bf128_inverse_zero() {
  %a = field.constant 0 : !BF128
  %c = field.inverse %a : !BF128

  %result_i128 = field.bitcast %c : !BF128 -> i128
  %result = arith.trunci %result_i128 : i128 to i32
  %tensor = tensor.from_elements %result : tensor<1xi32>
  %buffer = bufferization.to_buffer %tensor : tensor<1xi32> to memref<1xi32>
  %cast = memref.cast %buffer : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
  return
}
// CHECK: {{^}}[0]

func.func @main() {
  func.call @test_bf8_add() : () -> ()
  func.call @test_bf8_mul() : () -> ()
  func.call @test_bf8_square() : () -> ()
  func.call @test_bf8_double() : () -> ()
  func.call @test_bf8_mul_self() : () -> ()
  func.call @test_bf8_mul_cross() : () -> ()
  func.call @test_bf8_square_cross() : () -> ()
  func.call @test_bf128_mul() : () -> ()
  func.call @test_bf128_mul_self() : () -> ()
  func.call @test_bf128_mul_cross() : () -> ()
  func.call @test_bf64_inverse_known() : () -> ()
  func.call @test_bf64_inverse_roundtrip() : () -> ()
  func.call @test_bf128_inverse_known() : () -> ()
  func.call @test_bf128_inverse_roundtrip() : () -> ()
  func.call @test_bf64_inverse_zero() : () -> ()
  func.call @test_bf128_inverse_zero() : () -> ()
  return
}
