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

// Test ARM PMULL specialization for binary field operations.
//
// PMULL is a carryless (polynomial) multiply, which realizes the flat GHASH
// polynomial basis (reduction x¹²⁸ + x⁷ + x² + x + 1), not the recursive tower
// basis. So the 128-bit fast path applies only to `bf<7, ghash>`; tower
// bf<6>/bf<7> stay portable (`field.mul`/`field.square`). Packed 16×bf<3>
// vectors keep their own PMULL byte-wise fast path.

// Test PMULL enabled (default).
// RUN: prime-ir-opt --specialize-binary-field-to-arm %s | FileCheck %s --check-prefix=CHECK-PMULL

// Test PMULL disabled.
// RUN: prime-ir-opt --specialize-binary-field-to-arm="use-pmull=false" %s | FileCheck %s --check-prefix=CHECK-DISABLED

!BF8 = !field.bf<3>          // GF(2^8)
!BF64 = !field.bf<6>         // GF(2^64)
!BF128 = !field.bf<7>        // GF(2^128), tower basis
!GHASH = !field.bf<7, ghash> // GF(2^128), flat GHASH polynomial basis

//===----------------------------------------------------------------------===//
// Tower BF64/BF128: portable (carryless multiply computes the wrong basis)
//===----------------------------------------------------------------------===//

// CHECK-PMULL-LABEL: @test_bf64_mul
// CHECK-PMULL: field.mul
// CHECK-PMULL-NOT: pmull
// CHECK-DISABLED-LABEL: @test_bf64_mul
// CHECK-DISABLED: field.mul
func.func @test_bf64_mul(%a: !BF64, %b: !BF64) -> !BF64 {
  %c = field.mul %a, %b : !BF64
  return %c : !BF64
}

// CHECK-PMULL-LABEL: @test_bf64_square
// CHECK-PMULL: field.square
// CHECK-PMULL-NOT: pmull
// CHECK-DISABLED-LABEL: @test_bf64_square
// CHECK-DISABLED: field.square
func.func @test_bf64_square(%a: !BF64) -> !BF64 {
  %c = field.square %a : !BF64
  return %c : !BF64
}

// CHECK-PMULL-LABEL: @test_bf128_mul
// CHECK-PMULL: field.mul
// CHECK-PMULL-NOT: pmull
// CHECK-DISABLED-LABEL: @test_bf128_mul
// CHECK-DISABLED: field.mul
func.func @test_bf128_mul(%a: !BF128, %b: !BF128) -> !BF128 {
  %c = field.mul %a, %b : !BF128
  return %c : !BF128
}

//===----------------------------------------------------------------------===//
// GHASH BF128: carryless PMULL fast path (3 PMULL Karatsuba)
//===----------------------------------------------------------------------===//

// CHECK-PMULL-LABEL: @test_ghash_mul
// CHECK-PMULL: builtin.unrealized_conversion_cast
// CHECK-PMULL: llvm.inline_asm{{.*}}pmull{{[[:space:]]}}
// CHECK-PMULL: llvm.inline_asm{{.*}}pmull2
// CHECK-PMULL: llvm.inline_asm{{.*}}pmull{{[[:space:]]}}
// CHECK-PMULL: builtin.unrealized_conversion_cast
// CHECK-DISABLED-LABEL: @test_ghash_mul
// CHECK-DISABLED: field.mul
// CHECK-DISABLED-NOT: pmull
func.func @test_ghash_mul(%a: !GHASH, %b: !GHASH) -> !GHASH {
  %c = field.mul %a, %b : !GHASH
  return %c : !GHASH
}

// CHECK-PMULL-LABEL: @test_ghash_square
// CHECK-PMULL: builtin.unrealized_conversion_cast
// CHECK-PMULL: llvm.inline_asm{{.*}}pmull{{[[:space:]]}}
// CHECK-PMULL: llvm.inline_asm{{.*}}pmull2
// CHECK-PMULL: llvm.inline_asm{{.*}}pmull{{[[:space:]]}}
// CHECK-PMULL: builtin.unrealized_conversion_cast
// CHECK-DISABLED-LABEL: @test_ghash_square
// CHECK-DISABLED: field.square
// CHECK-DISABLED-NOT: pmull
func.func @test_ghash_square(%a: !GHASH) -> !GHASH {
  %c = field.square %a : !GHASH
  return %c : !GHASH
}

//===----------------------------------------------------------------------===//
// Packed BF8 Tests (16 x 8-bit binary field = 128-bit vector)
//===----------------------------------------------------------------------===//

// Packed 16 x BF8 vector multiplication should use PMULL
// CHECK-PMULL-LABEL: @test_packed_bf8_mul_128
// CHECK-PMULL: builtin.unrealized_conversion_cast
// CHECK-PMULL: llvm.inline_asm{{.*}}pmull
// CHECK-PMULL: llvm.inline_asm{{.*}}pmull2
// CHECK-PMULL: builtin.unrealized_conversion_cast
// CHECK-DISABLED-LABEL: @test_packed_bf8_mul_128
// CHECK-DISABLED: field.mul
// CHECK-DISABLED-NOT: pmull
func.func @test_packed_bf8_mul_128(%a: vector<16x!BF8>, %b: vector<16x!BF8>) -> vector<16x!BF8> {
  %c = field.mul %a, %b : vector<16x!BF8>
  return %c : vector<16x!BF8>
}

// Vector square should use PMULL
// CHECK-PMULL-LABEL: @test_packed_bf8_square_128
// CHECK-PMULL: builtin.unrealized_conversion_cast
// CHECK-PMULL: llvm.inline_asm{{.*}}pmull{{[[:space:]]}}
// CHECK-PMULL: llvm.inline_asm{{.*}}pmull2
// CHECK-PMULL: builtin.unrealized_conversion_cast
// CHECK-DISABLED-LABEL: @test_packed_bf8_square_128
// CHECK-DISABLED: field.square
func.func @test_packed_bf8_square_128(%a: vector<16x!BF8>) -> vector<16x!BF8> {
  %c = field.square %a : vector<16x!BF8>
  return %c : vector<16x!BF8>
}

//===----------------------------------------------------------------------===//
// Negative Tests (should NOT be specialized)
//===----------------------------------------------------------------------===//

// Scalar BF8 multiplication should NOT be specialized
// CHECK-PMULL-LABEL: @test_scalar_bf8_mul
// CHECK-PMULL: field.mul
// CHECK-PMULL-NOT: llvm.inline_asm
// CHECK-DISABLED-LABEL: @test_scalar_bf8_mul
// CHECK-DISABLED: field.mul
func.func @test_scalar_bf8_mul(%a: !BF8, %b: !BF8) -> !BF8 {
  %c = field.mul %a, %b : !BF8
  return %c : !BF8
}

// Small packed (not 16) should NOT be specialized
// CHECK-PMULL-LABEL: @test_small_packed_bf8_mul
// CHECK-PMULL: field.mul
// CHECK-PMULL-NOT: llvm.inline_asm
// CHECK-DISABLED-LABEL: @test_small_packed_bf8_mul
// CHECK-DISABLED: field.mul
func.func @test_small_packed_bf8_mul(%a: vector<8x!BF8>, %b: vector<8x!BF8>) -> vector<8x!BF8> {
  %c = field.mul %a, %b : vector<8x!BF8>
  return %c : vector<8x!BF8>
}

// Tensor types should NOT be specialized (need bufferization first)
// CHECK-PMULL-LABEL: @test_tensor_bf8_mul
// CHECK-PMULL: field.mul
// CHECK-PMULL-NOT: llvm.inline_asm
// CHECK-DISABLED-LABEL: @test_tensor_bf8_mul
// CHECK-DISABLED: field.mul
func.func @test_tensor_bf8_mul(%a: tensor<16x!BF8>, %b: tensor<16x!BF8>) -> tensor<16x!BF8> {
  %c = field.mul %a, %b : tensor<16x!BF8>
  return %c : tensor<16x!BF8>
}

// Vector of BF64 should NOT be specialized (only scalar supported)
// CHECK-PMULL-LABEL: @test_bf64_vec_mul
// CHECK-PMULL: field.mul
// CHECK-PMULL-NOT: llvm.inline_asm
// CHECK-DISABLED-LABEL: @test_bf64_vec_mul
// CHECK-DISABLED: field.mul
func.func @test_bf64_vec_mul(%a: vector<2x!BF64>, %b: vector<2x!BF64>) -> vector<2x!BF64> {
  %c = field.mul %a, %b : vector<2x!BF64>
  return %c : vector<2x!BF64>
}
