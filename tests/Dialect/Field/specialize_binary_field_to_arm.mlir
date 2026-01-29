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

// Test ARM PMULL specialization for binary field operations
//
// Note: By default, use-pmull is true.
// PMULL operates on vector types for bf<8>, and scalars for bf<64>/bf<128>.

// Test Tower reduction (default)
// RUN: prime-ir-opt --specialize-binary-field-to-arm %s | FileCheck %s --check-prefix=CHECK-TOWER

// Test Polyval reduction for bf<128>
// RUN: prime-ir-opt --specialize-binary-field-to-arm="use-polyval=true" %s | FileCheck %s --check-prefix=CHECK-POLYVAL

// Test PMULL disabled
// RUN: prime-ir-opt --specialize-binary-field-to-arm="use-pmull=false" %s | FileCheck %s --check-prefix=CHECK-DISABLED

!BF8 = !field.bf<3>   // GF(2^8)
!BF64 = !field.bf<6>  // GF(2^64)
!BF128 = !field.bf<7> // GF(2^128)

//===----------------------------------------------------------------------===//
// BF64 Tests (64-bit binary field)
//===----------------------------------------------------------------------===//

// BF64 scalar multiplication should use PMULL
// CHECK-TOWER-LABEL: @test_bf64_mul
// CHECK-TOWER: builtin.unrealized_conversion_cast
// CHECK-TOWER: llvm.inline_asm{{.*}}pmull
// CHECK-TOWER: builtin.unrealized_conversion_cast
// CHECK-POLYVAL-LABEL: @test_bf64_mul
// CHECK-POLYVAL: llvm.inline_asm{{.*}}pmull
// CHECK-DISABLED-LABEL: @test_bf64_mul
// CHECK-DISABLED: field.mul
// CHECK-DISABLED-NOT: llvm.inline_asm
func.func @test_bf64_mul(%a: !BF64, %b: !BF64) -> !BF64 {
  %c = field.mul %a, %b : !BF64
  return %c : !BF64
}

//===----------------------------------------------------------------------===//
// BF128 Tests (128-bit binary field)
//===----------------------------------------------------------------------===//

// BF128 scalar multiplication: Tower uses Karatsuba + shift/XOR reduction
// Polyval uses Montgomery reduction with 2 PMULL + ext
// CHECK-TOWER-LABEL: @test_bf128_mul
// CHECK-TOWER: builtin.unrealized_conversion_cast
// CHECK-TOWER: llvm.bitcast
// CHECK-TOWER: llvm.inline_asm{{.*}}pmull
// CHECK-TOWER: llvm.inline_asm{{.*}}pmull2
// CHECK-TOWER: arith.shrui
// CHECK-TOWER: arith.shli
// CHECK-TOWER: arith.xori
// CHECK-TOWER: builtin.unrealized_conversion_cast
// CHECK-POLYVAL-LABEL: @test_bf128_mul
// CHECK-POLYVAL: builtin.unrealized_conversion_cast
// CHECK-POLYVAL: llvm.bitcast
// Karatsuba step: ext for swapping, then pmull/pmull2 for products
// CHECK-POLYVAL: llvm.inline_asm{{.*}}ext{{.*}}#8
// CHECK-POLYVAL: llvm.inline_asm{{.*}}ext{{.*}}#8
// CHECK-POLYVAL: llvm.inline_asm{{.*}}pmull
// CHECK-POLYVAL: llvm.inline_asm{{.*}}pmull2
// CHECK-POLYVAL: llvm.inline_asm{{.*}}pmull
// Montgomery reduction: pmull, ext, pmull2
// CHECK-POLYVAL: llvm.inline_asm{{.*}}pmull
// CHECK-POLYVAL: llvm.inline_asm{{.*}}ext{{.*}}#8
// CHECK-POLYVAL: llvm.inline_asm{{.*}}pmull2
// CHECK-POLYVAL: builtin.unrealized_conversion_cast
// CHECK-DISABLED-LABEL: @test_bf128_mul
// CHECK-DISABLED: field.mul
// CHECK-DISABLED-NOT: llvm.inline_asm
func.func @test_bf128_mul(%a: !BF128, %b: !BF128) -> !BF128 {
  %c = field.mul %a, %b : !BF128
  return %c : !BF128
}

//===----------------------------------------------------------------------===//
// Packed BF8 Tests (16 x 8-bit binary field = 128-bit vector)
//===----------------------------------------------------------------------===//

// Packed 16 x BF8 vector multiplication should use PMULL
// CHECK-TOWER-LABEL: @test_packed_bf8_mul_128
// CHECK-TOWER: builtin.unrealized_conversion_cast
// CHECK-TOWER: llvm.inline_asm{{.*}}pmull
// CHECK-TOWER: llvm.inline_asm{{.*}}pmull2
// CHECK-TOWER: builtin.unrealized_conversion_cast
// CHECK-POLYVAL-LABEL: @test_packed_bf8_mul_128
// CHECK-POLYVAL: llvm.inline_asm{{.*}}pmull
// CHECK-DISABLED-LABEL: @test_packed_bf8_mul_128
// CHECK-DISABLED: field.mul
// CHECK-DISABLED-NOT: llvm.inline_asm
func.func @test_packed_bf8_mul_128(%a: vector<16x!BF8>, %b: vector<16x!BF8>) -> vector<16x!BF8> {
  %c = field.mul %a, %b : vector<16x!BF8>
  return %c : vector<16x!BF8>
}

// Vector square should use PMULL
// CHECK-TOWER-LABEL: @test_packed_bf8_square_128
// CHECK-TOWER: builtin.unrealized_conversion_cast
// CHECK-TOWER: llvm.inline_asm{{.*}}pmull
// CHECK-TOWER: builtin.unrealized_conversion_cast
// CHECK-POLYVAL-LABEL: @test_packed_bf8_square_128
// CHECK-POLYVAL: llvm.inline_asm{{.*}}pmull
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
// CHECK-TOWER-LABEL: @test_scalar_bf8_mul
// CHECK-TOWER: field.mul
// CHECK-TOWER-NOT: llvm.inline_asm
// CHECK-POLYVAL-LABEL: @test_scalar_bf8_mul
// CHECK-POLYVAL: field.mul
// CHECK-DISABLED-LABEL: @test_scalar_bf8_mul
// CHECK-DISABLED: field.mul
func.func @test_scalar_bf8_mul(%a: !BF8, %b: !BF8) -> !BF8 {
  %c = field.mul %a, %b : !BF8
  return %c : !BF8
}

// Small packed (not 16) should NOT be specialized
// CHECK-TOWER-LABEL: @test_small_packed_bf8_mul
// CHECK-TOWER: field.mul
// CHECK-TOWER-NOT: llvm.inline_asm
// CHECK-POLYVAL-LABEL: @test_small_packed_bf8_mul
// CHECK-POLYVAL: field.mul
// CHECK-DISABLED-LABEL: @test_small_packed_bf8_mul
// CHECK-DISABLED: field.mul
func.func @test_small_packed_bf8_mul(%a: vector<8x!BF8>, %b: vector<8x!BF8>) -> vector<8x!BF8> {
  %c = field.mul %a, %b : vector<8x!BF8>
  return %c : vector<8x!BF8>
}

// Tensor types should NOT be specialized (need bufferization first)
// CHECK-TOWER-LABEL: @test_tensor_bf8_mul
// CHECK-TOWER: field.mul
// CHECK-TOWER-NOT: llvm.inline_asm
// CHECK-POLYVAL-LABEL: @test_tensor_bf8_mul
// CHECK-POLYVAL: field.mul
// CHECK-DISABLED-LABEL: @test_tensor_bf8_mul
// CHECK-DISABLED: field.mul
func.func @test_tensor_bf8_mul(%a: tensor<16x!BF8>, %b: tensor<16x!BF8>) -> tensor<16x!BF8> {
  %c = field.mul %a, %b : tensor<16x!BF8>
  return %c : tensor<16x!BF8>
}

// Vector of BF64 should NOT be specialized (only scalar supported)
// CHECK-TOWER-LABEL: @test_bf64_vec_mul
// CHECK-TOWER: field.mul
// CHECK-TOWER-NOT: llvm.inline_asm
// CHECK-POLYVAL-LABEL: @test_bf64_vec_mul
// CHECK-POLYVAL: field.mul
// CHECK-DISABLED-LABEL: @test_bf64_vec_mul
// CHECK-DISABLED: field.mul
func.func @test_bf64_vec_mul(%a: vector<2x!BF64>, %b: vector<2x!BF64>) -> vector<2x!BF64> {
  %c = field.mul %a, %b : vector<2x!BF64>
  return %c : vector<2x!BF64>
}
