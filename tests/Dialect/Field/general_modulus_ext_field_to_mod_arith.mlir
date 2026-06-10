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

// RUN: prime-ir-opt -field-to-mod-arith %s | FileCheck %s

// General monic modulus u^3 = 1 + u (x^3 - x - 1, irreducible over F13 —
// no roots mod 13), encoded as dense low coefficients on a prime base.
// The pil2-stark Goldilocks3 convention at test size.
!PF = !field.pf<13:i32>
!TF = !field.ef<3x!PF, dense<[1, 1, 0]> : tensor<3xi32>>

// Add/sub never touch the modulus: same shape as the binomial lowering.
// CHECK-LABEL: @test_lower_add
func.func @test_lower_add(%arg0: !TF, %arg1: !TF) -> !TF {
    // CHECK-COUNT-2: field.ext_to_coeffs
    // CHECK-COUNT-3: mod_arith.add
    // CHECK: field.ext_from_coeffs
    %0 = field.add %arg0, %arg1 : !TF
    return %0 : !TF
}

// CHECK-LABEL: @test_lower_mul
func.func @test_lower_mul(%arg0: !TF, %arg1: !TF) -> !TF {
    // The modulus low coefficients materialize as constants (1, 1, 0)...
    // CHECK-DAG: mod_arith.constant 1 : !z13_i32
    // CHECK-DAG: mod_arith.constant 0 : !z13_i32
    // CHECK-COUNT-2: field.ext_to_coeffs
    // ...and the Karatsuba product folds every overflow coefficient through
    // them (6 product muls + 2 overflow slots x 3 modulus muls).
    // CHECK-COUNT-12: mod_arith.mul
    // CHECK: field.ext_from_coeffs
    %0 = field.mul %arg0, %arg1 : !TF
    return %0 : !TF
}

// CHECK-LABEL: @test_lower_square
func.func @test_lower_square(%arg0: !TF) -> !TF {
    // The general modulus routes Square through Karatsuba's generic fold
    // (CH-SQR2 is binomial-only): 3 coefficient squares + cross/fold muls.
    // CHECK: field.ext_to_coeffs
    // CHECK-COUNT-3: mod_arith.square
    // CHECK: field.ext_from_coeffs
    %0 = field.square %arg0 : !TF
    return %0 : !TF
}

// CHECK-LABEL: @test_lower_inverse
func.func @test_lower_inverse(%arg0: !TF) -> !TF {
    // Multiplication-by-x matrix inverse: build x*u and x*u^2 columns
    // (shift + modulus fold), first-row cofactors, Laplace determinant,
    // one base inverse, then scale the cofactors.
    // CHECK: field.ext_to_coeffs
    // CHECK: mod_arith.inverse
    // CHECK-COUNT-3: mod_arith.mul
    // CHECK: field.ext_from_coeffs
    %inv = field.inverse %arg0 : !TF
    return %inv : !TF
}

// The Montgomery variant lowers through the same path; the dense modulus
// coefficients are stored (and printed) in standard form regardless.
!PFm = !field.pf<13:i32, true>
!TFm = !field.ef<3x!PFm, dense<[1, 1, 0]> : tensor<3xi32>>

// CHECK-LABEL: @test_lower_mul_mont
func.func @test_lower_mul_mont(%arg0: !TFm, %arg1: !TFm) -> !TFm {
    // CHECK-COUNT-2: field.ext_to_coeffs
    // CHECK-COUNT-12: mod_arith.mul
    // CHECK: field.ext_from_coeffs
    %0 = field.mul %arg0, %arg1 : !TFm
    return %0 : !TFm
}
