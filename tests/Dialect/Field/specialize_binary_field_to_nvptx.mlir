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
// clmad computes a carryless product, realizing multiplication in a flat
// polynomial basis. `bf<7, ghash>` multiplies directly (reduction
// x¹²⁸ + x⁷ + x² + x + 1). Tower bf<4>/bf<5> specialize through the
// GF(2)-linear tower->flat basis conversion (TowerFlatBasis.h): ladder in,
// one clmad.lo product + two reduction folds, ladder out. Tower bf<6>/bf<7>
// stay portable (`field.mul`) until a clmad.hi limb + degree-64 modulus land.

// RUN: prime-ir-opt --specialize-binary-field-to-nvptx %s | FileCheck %s --check-prefix=CHECK-CLMAD

// use-clmad=false disables specialization.
// RUN: prime-ir-opt --specialize-binary-field-to-nvptx="use-clmad=false" %s | FileCheck %s --check-prefix=CHECK-OFF

!BF16 = !field.bf<4>         // GF(2¹⁶), tower basis
!BF32 = !field.bf<5>         // GF(2³²), tower basis
!BF64 = !field.bf<6>         // GF(2⁶⁴), tower basis
!BF128 = !field.bf<7>        // GF(2¹²⁸), tower basis
!GHASH = !field.bf<7, ghash> // GF(2¹²⁸), flat GHASH polynomial basis

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

// BF16 (tower) takes the same flat-basis path with the degree-16 modulus.
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
