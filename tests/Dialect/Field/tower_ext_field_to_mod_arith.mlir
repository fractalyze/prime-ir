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

// Tower extension: Fp6 = (Fp2)³ where Fp2 = Fp[v]/(v² - 6), Fp6 = Fp2[w]/(w³ - 2)
!PF = !field.pf<7:i32>
!QF = !field.ef<2x!PF, 6:i32>
!TowerF6 = !field.ef<3x!QF, 2:i32>

// CHECK-LABEL: @test_lower_tower_cmp_eq
func.func @test_lower_tower_cmp_eq(%arg0: !TowerF6, %arg1: !TowerF6) -> i1 {
    // Fp6 → 3×Fp2 → 6×Fp (per operand, 2 operands = 8 ext_to_coeffs total)
    // CHECK-COUNT-8: field.ext_to_coeffs
    // CHECK-COUNT-6: mod_arith.cmp eq
    // CHECK-COUNT-5: arith.andi
    %0 = field.cmp eq, %arg0, %arg1 : !TowerF6
    return %0 : i1
}

// CHECK-LABEL: @test_lower_tower_cmp_ne
func.func @test_lower_tower_cmp_ne(%arg0: !TowerF6, %arg1: !TowerF6) -> i1 {
    // CHECK-COUNT-8: field.ext_to_coeffs
    // CHECK-COUNT-6: mod_arith.cmp ne
    // CHECK-COUNT-5: arith.ori
    %0 = field.cmp ne, %arg0, %arg1 : !TowerF6
    return %0 : i1
}
