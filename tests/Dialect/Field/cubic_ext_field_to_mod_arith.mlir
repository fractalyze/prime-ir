// Copyright 2025 The PrimeIR Authors.
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

// RUN: prime-ir-opt -field-to-mod-arith %s | FileCheck %s

!PF = !field.pf<7:i32>
!PFm = !field.pf<7:i32, true>
!CF = !field.f3<!PF, 2:i32>
!CFm = !field.f3<!PFm, 2:i32>

// CHECK-LABEL: @test_lower_add
func.func @test_lower_add(%arg0: !CF, %arg1: !CF) -> !CF {
    // CHECK-COUNT-2: field.ext_to_coeffs
    // CHECK-COUNT-3: mod_arith.add
    // CHECK: field.ext_from_coeffs
    %0 = field.add %arg0, %arg1 : !CF
    return %0 : !CF
}

// CHECK-LABEL: @test_lower_sub
func.func @test_lower_sub(%arg0: !CF, %arg1: !CF) -> !CF {
    // CHECK-COUNT-2: field.ext_to_coeffs
    // CHECK-COUNT-3: mod_arith.sub
    // CHECK: field.ext_from_coeffs
    %0 = field.sub %arg0, %arg1 : !CF
    return %0 : !CF
}

// CHECK-LABEL: @test_lower_mul
func.func @test_lower_mul(%arg0: !CF, %arg1: !CF) -> !CF {
    // CHECK: mod_arith.constant 2 : !z7_i32
    // CHECK-COUNT-2: field.ext_to_coeffs
    // CHECK-COUNT-8: mod_arith.mul
    // CHECK: field.ext_from_coeffs
    %0 = field.mul %arg0, %arg1 : !CF
    return %0 : !CF
}

// CHECK-LABEL: @test_lower_square
func.func @test_lower_square(%arg0: !CF) -> !CF {
    // Karatsuba square: first compute coefficient squares, then cross products
    // CHECK: mod_arith.constant 2 : !z7_i32
    // CHECK: field.ext_to_coeffs
    // CHECK-COUNT-3: mod_arith.square
    // CHECK-COUNT-3: mod_arith.mul
    // CHECK: field.ext_from_coeffs
    %0 = field.square %arg0 : !CF
    return %0 : !CF
}

// CHECK-LABEL: @test_lower_inverse
func.func @test_lower_inverse(%arg0: !CF) -> !CF {
    // CHECK: mod_arith.constant 2 : !z7_i32
    // CHECK: field.ext_to_coeffs
    //   t₀ = x₀² - ξ * x₁ * x₂
    // CHECK: mod_arith.square
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.sub
    //   t₁ = ξ * x₂² - x₀ * x₁
    // CHECK: mod_arith.square
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.sub
    //   t₂ = x₁² - x₀ * x₂
    // CHECK: mod_arith.square
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.sub
    //   t₃ = x₀ * t₀ + ξ * (x₂ * t₁ + x₁ * t₂)
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.add
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.add
    //   (y₀, y₁, y₂) = (t₀, t₁, t₂) * t₃⁻¹
    // CHECK: mod_arith.inverse
    // CHECK-COUNT-3: mod_arith.mul
    // CHECK: field.ext_from_coeffs
    %inv = field.inverse %arg0 : !CF
    return %inv : !CF
}

// CHECK-LABEL: @test_lower_cmp_eq
func.func @test_lower_cmp_eq(%arg0: !CF, %arg1: !CF) -> i1 {
    // CHECK-COUNT-2: field.ext_to_coeffs
    // CHECK-COUNT-3: mod_arith.cmp eq
    // CHECK-COUNT-2: arith.andi
    %0 = field.cmp eq, %arg0, %arg1 : !CF
    return %0 : i1
}

// CHECK-LABEL: @test_lower_to_mont
func.func @test_lower_to_mont(%arg0: !CF) -> !CFm {
    // CHECK: field.ext_to_coeffs
    // CHECK-COUNT-3: mod_arith.to_mont
    // CHECK: field.ext_from_coeffs
    %0 = field.to_mont %arg0 : !CFm
    return %0 : !CFm
}

// CHECK-LABEL: @test_lower_from_mont
func.func @test_lower_from_mont(%arg0: !CFm) -> !CF {
    // CHECK: field.ext_to_coeffs
    // CHECK-COUNT-3: mod_arith.from_mont
    // CHECK: field.ext_from_coeffs
    %0 = field.from_mont %arg0 : !CF
    return %0 : !CF
}
