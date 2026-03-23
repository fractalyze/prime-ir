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

// 32-bit prime field (single limb) → uses Karatsuba mul + kCustom square.
!PF = !field.pf<127:i32>
!PFm = !field.pf<127:i32, true>
!QF = !field.ef<4x!PF, 2:i32>
!QFm = !field.ef<4x!PFm, 2:i32>

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
    // Karatsuba multiplication (single-limb field):
    // 4 diagonal (xᵢyᵢ) + 6 cross ((xᵢ+xⱼ)(yᵢ+yⱼ) - xᵢyᵢ - xⱼyⱼ) + 3 reduction (ξ * cᵢ₊₄)
    // CHECK-COUNT-2: field.ext_to_coeffs
    // CHECK-COUNT-13: mod_arith.mul
    // CHECK: field.ext_from_coeffs
    %0 = field.mul %arg0, %arg1 : !QF
    return %0 : !QF
}

// CHECK-LABEL: @test_lower_square
func.func @test_lower_square(%arg0: !QF) -> !QF {
    // Custom squaring (single-limb field):
    // 5 squares (x₀², x₁², x₂², x₃², (x₀+x₂)²) + 4 cross muls + 3 non-residue muls
    // CHECK: field.ext_to_coeffs
    // CHECK-COUNT-5: mod_arith.square
    // CHECK-COUNT-7: mod_arith.mul
    // CHECK: field.ext_from_coeffs
    %0 = field.square %arg0 : !QF
    return %0 : !QF
}

// CHECK-LABEL: @test_lower_inverse
func.func @test_lower_inverse(%arg0: !QF) -> !QF {
    // Fp4 inverse via quadratic tower (u⁴ = ξ) with v = u²:
    // Write x = A + B·u, where A, B are in Fp² = Fp[v]/(v²-ξ), with A=x₀+x₂v, B=x₁+x₃v.
    // Then x⁻¹ = (A − B·u) · (A² − v·B²)⁻¹. If D = A² − v·B² = D₀ + D₁·v,
    // we invert D in Fp₂ by D⁻¹ = (D₀ − D₁·v)/(D₀² − ξ·D₁²), and expand back to
    // {1,u,u²,u³}.

    // CHECK: mod_arith.constant 2 : !z127_i32
    // CHECK: field.ext_to_coeffs

    //   x₀², x₁², x₂², x₃²
    // CHECK: mod_arith.square
    // CHECK: mod_arith.square
    // CHECK: mod_arith.square
    // CHECK: mod_arith.square
    //
    //   D₀ = (x₀² + 2·x₂²) − 4·x₁·x₃
    //   D₁ = (2·x₀·x₂) − (x₁² + 2·x₃²)
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.add
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.double
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.add
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.double
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.sub
    // CHECK: mod_arith.sub
    //
    //   denom = D₀² − 2·D₁²
    // CHECK: mod_arith.square
    // CHECK: mod_arith.square
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.sub
    //
    //   a = D₀ * denom⁻¹
    //   b = -D₁ * denom⁻¹
    // CHECK: mod_arith.inverse
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.negate
    // CHECK: mod_arith.mul
    //
    //   y₀ = x₀ * a + 2 * x₂ * b
    //   y₁ = -(x₁ * a + 2 * x₃ * b)
    //   y₂ = x₂ * a + x₀ * b
    //   y₃ = -(x₃ * a + x₁ * b)
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.add
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.add
    // CHECK: mod_arith.negate
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.add
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.add
    // CHECK: mod_arith.negate
    // CHECK: field.ext_from_coeffs
    %inv = field.inverse %arg0 : !QF
    return %inv : !QF
}

// Tensor extension field inverse uses Montgomery's batch inversion trick:
// O(3(n-1)) field multiplications + 1 scalar inversion.
// CHECK-LABEL: @test_tensor_batch_inverse
func.func @test_tensor_batch_inverse(%arg0: tensor<2x!QF>) -> tensor<2x!QF> {
    // Forward pass: extract a[0], build prefix products in scf.for.
    // CHECK:      tensor.extract %arg0
    // CHECK:      tensor.empty
    // CHECK:      tensor.insert
    // CHECK:      scf.for
    // CHECK:        tensor.extract
    // CHECK:        field.ext_to_coeffs
    // CHECK:        field.ext_from_coeffs
    // CHECK:        tensor.insert
    // Single scalar inverse on the total product (lowered to mod_arith).
    // CHECK:      mod_arith.inverse
    // CHECK:      field.ext_from_coeffs
    // Backward pass: recover individual inverses.
    // CHECK:      tensor.empty
    // CHECK:      scf.for
    // CHECK:        tensor.extract
    // CHECK:        field.ext_to_coeffs
    // CHECK:        field.ext_from_coeffs
    // CHECK:        tensor.insert
    // CHECK:        tensor.extract
    // CHECK:        field.ext_to_coeffs
    // CHECK:        field.ext_from_coeffs
    // Final insert for result[0].
    // CHECK:      tensor.insert
    %inv = field.inverse %arg0 : tensor<2x!QF>
    return %inv : tensor<2x!QF>
}

// Multi-dimensional tensor: collapse to 1-D, batch inverse, expand back.
// CHECK-LABEL: @test_multidim_tensor_batch_inverse
func.func @test_multidim_tensor_batch_inverse(%arg0: tensor<2x3x!QF>) -> tensor<2x3x!QF> {
    // CHECK:      tensor.collapse_shape %arg0 {{\[}}[0, 1]{{\]}}
    // CHECK:      scf.for
    // CHECK:      mod_arith.inverse
    // CHECK:      scf.for
    // CHECK:      tensor.insert
    // CHECK:      tensor.expand_shape {{.*}} {{\[}}[0, 1]{{\]}}
    %inv = field.inverse %arg0 : tensor<2x3x!QF>
    return %inv : tensor<2x3x!QF>
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
