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

// Test clmad specialization for GHASH-basis binary field multiplication.
//
// clmad computes a carryless product, realizing multiplication in the flat
// GHASH polynomial basis (reduction x¹²⁸ + x⁷ + x² + x + 1). It does NOT match
// the recursive tower basis, so the fast path applies only to `bf<7, ghash>`;
// tower bf<6>/bf<7> stay portable (`field.mul`).

// RUN: prime-ir-opt --specialize-binary-field-to-nvptx %s | FileCheck %s --check-prefix=CHECK-CLMAD

// use-clmad=false disables specialization.
// RUN: prime-ir-opt --specialize-binary-field-to-nvptx="use-clmad=false" %s | FileCheck %s --check-prefix=CHECK-OFF

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
