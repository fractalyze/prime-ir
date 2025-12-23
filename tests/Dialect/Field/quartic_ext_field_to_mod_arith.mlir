// Copyright 2025 The ZKIR Authors.
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
// See the License for the specific Language governing permissions and
// limitations under the License.
// ==============================================================================

// RUN: zkir-opt -field-to-mod-arith %s | FileCheck %s

// Modulus must be > 120 because Toom-Cook uses Vandermonde matrix coefficients.
// TODO(junbeomlee): Select multiplication algorithm based on modulus value.
!PF = !field.pf<127:i32>
!PFm = !field.pf<127:i32, true>
!QF = !field.f4<!PF, 2:i32>
!QFm = !field.f4<!PFm, 2:i32>

// CHECK-LABEL: @test_lower_add
func.func @test_lower_add(%arg0: !QF, %arg1: !QF) -> !QF {
    // CHECK-COUNT-2: field.ext_to_coeffs
    // CHECK-COUNT-4: mod_arith.add
    // CHECK: field.ext_from_coeffs
    %0 = field.add %arg0, %arg1 : !QF
    return %0 : !QF
}

// CHECK-LABEL: @test_lower_sub
func.func @test_lower_sub(%arg0: !QF, %arg1: !QF) -> !QF {
    // CHECK-COUNT-2: field.ext_to_coeffs
    // CHECK-COUNT-4: mod_arith.sub
    // CHECK: field.ext_from_coeffs
    %0 = field.sub %arg0, %arg1 : !QF
    return %0 : !QF
}

// CHECK-LABEL: @test_lower_mul
func.func @test_lower_mul(%arg0: !QF, %arg1: !QF) -> !QF {
    // Toom-Cook multiplication algorithm with 7 evaluation points (0, 1, -1, 2, -2, 3, ∞)
    // v0 = x₀*y₀, v1 = (x₀+x₁+x₂+x₃)(y₀+y₁+y₂+y₃), v2 = (x₀-x₁+x₂-x₃)(y₀-y₁+y₂-y₃),
    // v3 = (x₀+2x₁+4x₂+8x₃)(y₀+2y₁+4y₂+8y₃), v4 = (x₀-2x₁+4x₂-8x₃)(y₀-2y₁+4y₂-8y₃),
    // v5 = (x₀+3x₁+9x₂+27x₃)(y₀+3y₁+9y₂+27y₃), v6 = x₃*y₃
    // CHECK-COUNT-2: field.ext_to_coeffs
    // CHECK-COUNT-7: mod_arith.mul
    // CHECK: field.ext_from_coeffs
    %0 = field.mul %arg0, %arg1 : !QF
    return %0 : !QF
}

// CHECK-LABEL: @test_lower_square
func.func @test_lower_square(%arg0: !QF) -> !QF {
    // Toom-Cook squaring algorithm with 7 evaluation points (0, 1, -1, 2, -2, 3, ∞)
    // v0 = x₀², v1 = (x₀+x₁+x₂+x₃)², v2 = (x₀-x₁+x₂-x₃)²,
    // v3 = (x₀+2x₁+4x₂+8x₃)², v4 = (x₀-2x₁+4x₂-8x₃)²,
    // v5 = (x₀+3x₁+9x₂+27x₃)², v6 = x₃²
    // CHECK: field.ext_to_coeffs
    // CHECK-COUNT-7: mod_arith.square
    // CHECK: field.ext_from_coeffs
    %0 = field.square %arg0 : !QF
    return %0 : !QF
}

// CHECK-LABEL: @test_lower_inverse
func.func @test_lower_inverse(%arg0: !QF) -> !QF {
    // TODO(junbeomlee): After canonicalization, the norm computation should be
    // optimized from Toom-Cook (7 muls) to direct formula (5 muls):
    //   Norm = x₀ * t₀ + ξ * (x₁ * t₃ + x₂ * t₂ + x₃ * t₁)
    // where t = φ¹(x) * φ²(x) * φ³(x).
    // This reduces total from 34 muls to 32 muls.
    //
    // Frobenius-based inverse: x⁻¹ = Frobenius_product(x) * Norm(x)⁻¹
    // Frobenius_product = φ¹(x) * φ²(x) * φ³(x)
    //
    // Frobenius φ¹(x): coeffs scaled by ξ^(i * (p - 1) / 4)
    // CHECK: field.ext_to_coeffs
    // CHECK-COUNT-3: mod_arith.mul
    // CHECK: field.ext_from_coeffs
    //
    // Frobenius φ²(x): coeffs scaled by ξ^(i * (p² - 1) / 4)
    // CHECK: field.ext_to_coeffs
    // CHECK-COUNT-3: mod_arith.mul
    // CHECK: field.ext_from_coeffs
    //
    // Frobenius φ³(x): coeffs scaled by ξ^(i * (p³ - 1) / 4)
    // CHECK: field.ext_to_coeffs
    // CHECK-COUNT-3: mod_arith.mul
    // CHECK: field.ext_from_coeffs
    //
    // Toom-Cook mul: φ²(x) * φ³(x)
    // CHECK-COUNT-2: field.ext_to_coeffs
    // CHECK-COUNT-7: mod_arith.mul
    // CHECK: field.ext_from_coeffs
    //
    // Toom-Cook mul: result * φ¹(x)
    // CHECK-COUNT-2: field.ext_to_coeffs
    // CHECK-COUNT-7: mod_arith.mul
    // CHECK: field.ext_from_coeffs
    //
    // Toom-Cook mul: x * frob_product (for norm)
    // CHECK-COUNT-2: field.ext_to_coeffs
    // CHECK-COUNT-7: mod_arith.mul
    // CHECK: field.ext_from_coeffs
    //
    // Extract norm (constant term) and inverse
    // CHECK: field.ext_to_coeffs
    // CHECK: mod_arith.inverse
    //
    // Final scaling: frob_product * norm⁻¹
    // CHECK: field.ext_to_coeffs
    // CHECK-COUNT-4: mod_arith.mul
    // CHECK: field.ext_from_coeffs
    %inv = field.inverse %arg0 : !QF
    return %inv : !QF
}

// CHECK-LABEL: @test_lower_f4_create
func.func @test_lower_f4_create(%arg0: !PF, %arg1: !PF, %arg2: !PF, %arg3: !PF) -> !QF {
    // CHECK: field.ext_from_coeffs
    %0 = field.f4.create %arg0, %arg1, %arg2, %arg3 : !QF
    return %0 : !QF
}

// CHECK-LABEL: @test_lower_cmp_eq
func.func @test_lower_cmp_eq(%arg0: !QF, %arg1: !QF) -> i1 {
    // CHECK-COUNT-2: field.ext_to_coeffs
    // CHECK-COUNT-4: mod_arith.cmp eq
    // CHECK-COUNT-3: arith.andi
    %0 = field.cmp eq, %arg0, %arg1 : !QF
    return %0 : i1
}

// CHECK-LABEL: @test_lower_to_mont
func.func @test_lower_to_mont(%arg0: !QF) -> !QFm {
    // CHECK: field.ext_to_coeffs
    // CHECK-COUNT-4: mod_arith.to_mont
    // CHECK: field.ext_from_coeffs
    %0 = field.to_mont %arg0 : !QFm
    return %0 : !QFm
}

// CHECK-LABEL: @test_lower_from_mont
func.func @test_lower_from_mont(%arg0: !QFm) -> !QF {
    // CHECK: field.ext_to_coeffs
    // CHECK-COUNT-4: mod_arith.from_mont
    // CHECK: field.ext_from_coeffs
    %0 = field.from_mont %arg0 : !QF
    return %0 : !QF
}
