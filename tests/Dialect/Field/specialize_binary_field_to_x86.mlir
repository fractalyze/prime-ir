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

// Test GFNI specialization for packed 8-bit binary field multiplication
// Test PCLMULQDQ specialization for 64/128-bit binary field multiplication
//
// Note: By default, both use-gfni and use-pclmulqdq are true.
// GFNI operates on vector types, not tensors. Tensors should be
// converted to vectors via bufferization before this pass.

// Test GFNI-only (disable PCLMULQDQ)
// RUN: prime-ir-opt --specialize-binary-field-to-x86="use-pclmulqdq=false" %s | FileCheck %s --check-prefix=CHECK-GFNI

// Test PCLMULQDQ-only (disable GFNI)
// RUN: prime-ir-opt --specialize-binary-field-to-x86="use-gfni=false" %s | FileCheck %s --check-prefix=CHECK-PCLMULQDQ

// Test both enabled (default)
// RUN: prime-ir-opt --specialize-binary-field-to-x86 %s | FileCheck %s --check-prefix=CHECK-ALL

!BF8 = !field.bf<3>   // GF(2⁸)
!BF64 = !field.bf<6>  // GF(2⁶⁴)
!BF128 = !field.bf<7> // GF(2¹²⁸)

//===----------------------------------------------------------------------===//
// GFNI Tests (packed BF8 vectors)
//===----------------------------------------------------------------------===//

// Packed 16 x BF8 vector multiplication should use GFNI (128-bit)
// CHECK-GFNI-LABEL: @test_packed_bf8_mul_128
// CHECK-GFNI: llvm.inline_asm{{.*}}vgf2p8affineqb
// CHECK-GFNI: llvm.inline_asm{{.*}}vgf2p8mulb
// CHECK-GFNI: llvm.inline_asm{{.*}}vgf2p8affineqb
// CHECK-PCLMULQDQ-LABEL: @test_packed_bf8_mul_128
// CHECK-PCLMULQDQ: field.mul
// CHECK-ALL-LABEL: @test_packed_bf8_mul_128
// CHECK-ALL: llvm.inline_asm{{.*}}vgf2p8affineqb
func.func @test_packed_bf8_mul_128(%a: vector<16x!BF8>, %b: vector<16x!BF8>) -> vector<16x!BF8> {
  %c = field.mul %a, %b : vector<16x!BF8>
  return %c : vector<16x!BF8>
}

// Packed 32 x BF8 vector multiplication should use GFNI (256-bit AVX2)
// CHECK-GFNI-LABEL: @test_packed_bf8_mul_256
// CHECK-GFNI: llvm.inline_asm{{.*}}vgf2p8affineqb
// CHECK-GFNI: llvm.inline_asm{{.*}}vgf2p8mulb
// CHECK-GFNI: llvm.inline_asm{{.*}}vgf2p8affineqb
// CHECK-PCLMULQDQ-LABEL: @test_packed_bf8_mul_256
// CHECK-PCLMULQDQ: field.mul
// CHECK-ALL-LABEL: @test_packed_bf8_mul_256
// CHECK-ALL: llvm.inline_asm{{.*}}vgf2p8affineqb
func.func @test_packed_bf8_mul_256(%a: vector<32x!BF8>, %b: vector<32x!BF8>) -> vector<32x!BF8> {
  %c = field.mul %a, %b : vector<32x!BF8>
  return %c : vector<32x!BF8>
}

// Packed 64 x BF8 vector multiplication should use GFNI (512-bit AVX-512)
// CHECK-GFNI-LABEL: @test_packed_bf8_mul_512
// CHECK-GFNI: llvm.inline_asm{{.*}}vgf2p8affineqb
// CHECK-GFNI: llvm.inline_asm{{.*}}vgf2p8mulb
// CHECK-GFNI: llvm.inline_asm{{.*}}vgf2p8affineqb
// CHECK-PCLMULQDQ-LABEL: @test_packed_bf8_mul_512
// CHECK-PCLMULQDQ: field.mul
// CHECK-ALL-LABEL: @test_packed_bf8_mul_512
// CHECK-ALL: llvm.inline_asm{{.*}}vgf2p8affineqb
func.func @test_packed_bf8_mul_512(%a: vector<64x!BF8>, %b: vector<64x!BF8>) -> vector<64x!BF8> {
  %c = field.mul %a, %b : vector<64x!BF8>
  return %c : vector<64x!BF8>
}

// Scalar BF8 multiplication should NOT be specialized by GFNI
// CHECK-GFNI-LABEL: @test_scalar_bf8_mul
// CHECK-GFNI: field.mul
// CHECK-GFNI-NOT: llvm.inline_asm
// CHECK-PCLMULQDQ-LABEL: @test_scalar_bf8_mul
// CHECK-PCLMULQDQ: field.mul
// CHECK-ALL-LABEL: @test_scalar_bf8_mul
// CHECK-ALL: field.mul
func.func @test_scalar_bf8_mul(%a: !BF8, %b: !BF8) -> !BF8 {
  %c = field.mul %a, %b : !BF8
  return %c : !BF8
}

// Small packed (not 16/32/64) should NOT be specialized
// CHECK-GFNI-LABEL: @test_small_packed_bf8_mul
// CHECK-GFNI: field.mul
// CHECK-GFNI-NOT: llvm.inline_asm
// CHECK-PCLMULQDQ-LABEL: @test_small_packed_bf8_mul
// CHECK-PCLMULQDQ: field.mul
// CHECK-ALL-LABEL: @test_small_packed_bf8_mul
// CHECK-ALL: field.mul
func.func @test_small_packed_bf8_mul(%a: vector<8x!BF8>, %b: vector<8x!BF8>) -> vector<8x!BF8> {
  %c = field.mul %a, %b : vector<8x!BF8>
  return %c : vector<8x!BF8>
}

// Tensor types should NOT be specialized (need bufferization first)
// CHECK-GFNI-LABEL: @test_tensor_bf8_mul
// CHECK-GFNI: field.mul
// CHECK-GFNI-NOT: llvm.inline_asm
// CHECK-PCLMULQDQ-LABEL: @test_tensor_bf8_mul
// CHECK-PCLMULQDQ: field.mul
// CHECK-ALL-LABEL: @test_tensor_bf8_mul
// CHECK-ALL: field.mul
func.func @test_tensor_bf8_mul(%a: tensor<16x!BF8>, %b: tensor<16x!BF8>) -> tensor<16x!BF8> {
  %c = field.mul %a, %b : tensor<16x!BF8>
  return %c : tensor<16x!BF8>
}

// Vector square should use GFNI
// CHECK-GFNI-LABEL: @test_packed_bf8_square_128
// CHECK-GFNI: llvm.inline_asm{{.*}}vgf2p8affineqb
// CHECK-GFNI: llvm.inline_asm{{.*}}vgf2p8mulb
// CHECK-GFNI: llvm.inline_asm{{.*}}vgf2p8affineqb
// CHECK-PCLMULQDQ-LABEL: @test_packed_bf8_square_128
// CHECK-PCLMULQDQ: field.square
// CHECK-ALL-LABEL: @test_packed_bf8_square_128
// CHECK-ALL: llvm.inline_asm{{.*}}vgf2p8affineqb
func.func @test_packed_bf8_square_128(%a: vector<16x!BF8>) -> vector<16x!BF8> {
  %c = field.square %a : vector<16x!BF8>
  return %c : vector<16x!BF8>
}

//===----------------------------------------------------------------------===//
// PCLMULQDQ Tests (BF64/BF128 scalar)
//===----------------------------------------------------------------------===//

// BF64 scalar multiplication should use PCLMULQDQ
// CHECK-GFNI-LABEL: @test_bf64_mul
// CHECK-GFNI: field.mul
// CHECK-GFNI-NOT: llvm.inline_asm
// CHECK-PCLMULQDQ-LABEL: @test_bf64_mul
// CHECK-PCLMULQDQ: builtin.unrealized_conversion_cast
// CHECK-PCLMULQDQ: llvm.inline_asm{{.*}}vpclmulqdq
// CHECK-PCLMULQDQ: builtin.unrealized_conversion_cast
// CHECK-ALL-LABEL: @test_bf64_mul
// CHECK-ALL: llvm.inline_asm{{.*}}vpclmulqdq
func.func @test_bf64_mul(%a: !BF64, %b: !BF64) -> !BF64 {
  %c = field.mul %a, %b : !BF64
  return %c : !BF64
}

// BF128 scalar multiplication should use PCLMULQDQ with Karatsuba
// CHECK-GFNI-LABEL: @test_bf128_mul
// CHECK-GFNI: field.mul
// CHECK-GFNI-NOT: llvm.inline_asm
// CHECK-PCLMULQDQ-LABEL: @test_bf128_mul
// CHECK-PCLMULQDQ: builtin.unrealized_conversion_cast
// CHECK-PCLMULQDQ: llvm.inline_asm{{.*}}vpclmulqdq
// CHECK-PCLMULQDQ: builtin.unrealized_conversion_cast
// CHECK-ALL-LABEL: @test_bf128_mul
// CHECK-ALL: llvm.inline_asm{{.*}}vpclmulqdq
func.func @test_bf128_mul(%a: !BF128, %b: !BF128) -> !BF128 {
  %c = field.mul %a, %b : !BF128
  return %c : !BF128
}

// Vector of BF64 should NOT be specialized by this pass (use ARM pass for vectors)
// CHECK-GFNI-LABEL: @test_bf64_vec_mul
// CHECK-GFNI: field.mul
// CHECK-PCLMULQDQ-LABEL: @test_bf64_vec_mul
// CHECK-PCLMULQDQ: field.mul
// CHECK-ALL-LABEL: @test_bf64_vec_mul
// CHECK-ALL: field.mul
func.func @test_bf64_vec_mul(%a: vector<2x!BF64>, %b: vector<2x!BF64>) -> vector<2x!BF64> {
  %c = field.mul %a, %b : vector<2x!BF64>
  return %c : vector<2x!BF64>
}

// Tensor of BF64 should NOT be specialized (need bufferization first)
// CHECK-GFNI-LABEL: @test_tensor_bf64_mul
// CHECK-GFNI: field.mul
// CHECK-PCLMULQDQ-LABEL: @test_tensor_bf64_mul
// CHECK-PCLMULQDQ: field.mul
// CHECK-ALL-LABEL: @test_tensor_bf64_mul
// CHECK-ALL: field.mul
func.func @test_tensor_bf64_mul(%a: tensor<2x!BF64>, %b: tensor<2x!BF64>) -> tensor<2x!BF64> {
  %c = field.mul %a, %b : tensor<2x!BF64>
  return %c : tensor<2x!BF64>
}
