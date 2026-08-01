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

// Test clmad specialization for binary field multiplication.
//
// clmad computes a carryless product: `bf<7, ghash>` and `bf<3, aes>` are
// already flat so they multiply directly, tower bf<4>/bf<5> go through the
// flat-basis conversion (TowerFlatBasis.h), and tower bf<6>/bf<7> stay
// portable (`field.mul`).

// RUN: prime-ir-opt --specialize-binary-field-to-nvptx %s | FileCheck %s --check-prefix=CHECK-CLMAD

// use-clmad=false disables specialization.
// RUN: prime-ir-opt --specialize-binary-field-to-nvptx="use-clmad=false" %s | FileCheck %s --check-prefix=CHECK-OFF

// The emitted tower<->iN casts must be reconciled by the downstream
// binary-field-to-arith lowering — no field ops or casts may survive the pair.
// RUN: prime-ir-opt --specialize-binary-field-to-nvptx --binary-field-to-arith %s | FileCheck %s --check-prefix=CHECK-PIPELINE
// CHECK-PIPELINE-NOT: field.mul
// CHECK-PIPELINE-NOT: unrealized_conversion_cast

!BF16 = !field.bf<4>         // GF(2¹⁶), tower basis
!BF32 = !field.bf<5>         // GF(2³²), tower basis
!BF64 = !field.bf<6>         // GF(2⁶⁴), tower basis
!BF128 = !field.bf<7>        // GF(2¹²⁸), tower basis
!GHASH = !field.bf<7, ghash> // GF(2¹²⁸), flat GHASH polynomial basis
!AES = !field.bf<3, aes>     // GF(2⁸), flat AES polynomial basis
!F32 = !field.bf<5, flat>    // GF(2³²), flat polynomial basis

// GHASH-basis scalar multiplication should use clmad: eight clmad.{lo,hi}.u64
// build the 128×128 carryless product, then the shared GHASH reduction.
// CHECK-CLMAD-LABEL: @test_ghash_mul
// CHECK-CLMAD: builtin.unrealized_conversion_cast
// CHECK-CLMAD-COUNT-8: llvm.inline_asm{{.*}}clmad{{.*}}u64
// CHECK-CLMAD: builtin.unrealized_conversion_cast
// CHECK-OFF-LABEL: @test_ghash_mul
// CHECK-OFF: field.mul
// CHECK-OFF-NOT: clmad
func.func @test_ghash_mul(%a: !GHASH, %b: !GHASH) -> !GHASH {
  %c = field.mul %a, %b : !GHASH
  return %c : !GHASH
}

// BF32 (tower) multiplies via the flat basis: three clmad.lo.u64 (one product
// + two reduction folds) between the conversion ladders.
// CHECK-CLMAD-LABEL: @test_bf32_mul
// CHECK-CLMAD: builtin.unrealized_conversion_cast
// CHECK-CLMAD-COUNT-3: llvm.inline_asm{{.*}}clmad.lo{{.*}}u64
// CHECK-CLMAD: builtin.unrealized_conversion_cast
// CHECK-OFF-LABEL: @test_bf32_mul
// CHECK-OFF: field.mul
// CHECK-OFF-NOT: clmad
func.func @test_bf32_mul(%a: !BF32, %b: !BF32) -> !BF32 {
  %c = field.mul %a, %b : !BF32
  return %c : !BF32
}

// BF16 (tower) takes the same flat-basis path.
// CHECK-CLMAD-LABEL: @test_bf16_mul
// CHECK-CLMAD: builtin.unrealized_conversion_cast
// CHECK-CLMAD-COUNT-3: llvm.inline_asm{{.*}}clmad.lo{{.*}}u64
// CHECK-CLMAD: builtin.unrealized_conversion_cast
// CHECK-OFF-LABEL: @test_bf16_mul
// CHECK-OFF: field.mul
// CHECK-OFF-NOT: clmad
func.func @test_bf16_mul(%a: !BF16, %b: !BF16) -> !BF16 {
  %c = field.mul %a, %b : !BF16
  return %c : !BF16
}

// AES (flat) multiplies directly like GHASH, but the 8×8 product is 15-bit so
// it needs no Karatsuba limbs: one clmad.lo product + two reduction folds =
// three clmad.lo.u64, and no toFlat/fromFlat conversion.
// CHECK-CLMAD-LABEL: @test_aes_mul
// CHECK-CLMAD: builtin.unrealized_conversion_cast
// CHECK-CLMAD-COUNT-3: llvm.inline_asm{{.*}}clmad.lo{{.*}}u64
// CHECK-CLMAD: builtin.unrealized_conversion_cast
// CHECK-OFF-LABEL: @test_aes_mul
// CHECK-OFF: field.mul
// CHECK-OFF-NOT: clmad
func.func @test_aes_mul(%a: !AES, %b: !AES) -> !AES {
  %c = field.mul %a, %b : !AES
  return %c : !AES
}

// BF128 (tower) scalar multiplication stays portable — no carryless fast path.
// CHECK-CLMAD-LABEL: @test_bf128_mul
// CHECK-CLMAD: field.mul
// CHECK-CLMAD-NOT: clmad
// CHECK-OFF-LABEL: @test_bf128_mul
// CHECK-OFF: field.mul
func.func @test_bf128_mul(%a: !BF128, %b: !BF128) -> !BF128 {
  %c = field.mul %a, %b : !BF128
  return %c : !BF128
}

// BF64 (tower) scalar multiplication stays portable — no carryless fast path.
// CHECK-CLMAD-LABEL: @test_bf64_mul
// CHECK-CLMAD: field.mul
// CHECK-CLMAD-NOT: clmad
// CHECK-OFF-LABEL: @test_bf64_mul
// CHECK-OFF: field.mul
func.func @test_bf64_mul(%a: !BF64, %b: !BF64) -> !BF64 {
  %c = field.mul %a, %b : !BF64
  return %c : !BF64
}

// Native flat bf<5, flat> multiply is the clmad product with NO conversion
// ladders — no arith.select appears between the casts (the tower path above
// emits 2n of them per conversion).
// CHECK-CLMAD-LABEL: @test_f32_mul
// CHECK-CLMAD-NOT: arith.select
// CHECK-CLMAD-COUNT-3: llvm.inline_asm{{.*}}clmad.lo{{.*}}u64
// CHECK-CLMAD-NOT: arith.select
// CHECK-OFF-LABEL: @test_f32_mul
// CHECK-OFF: field.mul
// CHECK-OFF-NOT: clmad
func.func @test_f32_mul(%a: !F32, %b: !F32) -> !F32 {
  %c = field.mul %a, %b : !F32
  return %c : !F32
}

// Square is not specialized (only MulOp patterns are registered); it stays on
// the portable recursive tower square.
// CHECK-CLMAD-LABEL: @test_bf32_square
// CHECK-CLMAD: field.square
// CHECK-CLMAD-NOT: clmad
func.func @test_bf32_square(%a: !BF32) -> !BF32 {
  %c = field.square %a : !BF32
  return %c : !BF32
}
