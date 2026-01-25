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

// RUN: prime-ir-opt %s | FileCheck %s -enable-var-scope

// Test tower extension field type parsing and round-trip

!PF = !field.pf<7:i32>
!QF = !field.ef<2x!PF, 6:i32>
// Tower: Fp6 = (Fp2)^3 over Fp, nonResidue = 2 in Fp2
!TowerF6 = !field.ef<3x!QF, dense<[2, 0]> : tensor<2xi32>>
// Tower: Fp12 = (Fp6)^2, nonResidue = 2 in Fp6 (flattened: [2, 0, 0, 0, 0, 0])
!TowerF12 = !field.ef<2x!TowerF6, dense<[2, 0, 0, 0, 0, 0]> : tensor<6xi32>>

//===----------------------------------------------------------------------===//
// Test tower type parsing round-trip
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_tower_type_parsing
// CHECK-SAME: (%{{.*}}: [[TF6:.*]]) -> [[TF6]]
func.func @test_tower_type_parsing(%arg0: !TowerF6) -> !TowerF6 {
    return %arg0 : !TowerF6
}

// CHECK-LABEL: @test_deep_tower_type_parsing
// CHECK-SAME: (%{{.*}}: [[TF12:.*]]) -> [[TF12]]
func.func @test_deep_tower_type_parsing(%arg0: !TowerF12) -> !TowerF12 {
    return %arg0 : !TowerF12
}

//===----------------------------------------------------------------------===//
// Test ext_to_coeffs and ext_from_coeffs with tower types
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_tower_ext_to_coeffs
func.func @test_tower_ext_to_coeffs(%arg0: !TowerF6) -> (!QF, !QF, !QF) {
    // Tower extension to coeffs: Fp6 -> (Fp2, Fp2, Fp2)
    // CHECK: field.ext_to_coeffs
    %0:3 = field.ext_to_coeffs %arg0 : (!TowerF6) -> (!QF, !QF, !QF)
    return %0#0, %0#1, %0#2 : !QF, !QF, !QF
}

// CHECK-LABEL: @test_tower_ext_from_coeffs
func.func @test_tower_ext_from_coeffs(%arg0: !QF, %arg1: !QF, %arg2: !QF) -> !TowerF6 {
    // Coeffs to tower extension: (Fp2, Fp2, Fp2) -> Fp6
    // CHECK: field.ext_from_coeffs
    %0 = field.ext_from_coeffs %arg0, %arg1, %arg2 : (!QF, !QF, !QF) -> !TowerF6
    return %0 : !TowerF6
}

//===----------------------------------------------------------------------===//
// Test arithmetic operations on tower extensions
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_tower_add
func.func @test_tower_add(%arg0: !TowerF6, %arg1: !TowerF6) -> !TowerF6 {
    // CHECK: field.add
    %0 = field.add %arg0, %arg1 : !TowerF6
    return %0 : !TowerF6
}

// CHECK-LABEL: @test_tower_sub
func.func @test_tower_sub(%arg0: !TowerF6, %arg1: !TowerF6) -> !TowerF6 {
    // CHECK: field.sub
    %0 = field.sub %arg0, %arg1 : !TowerF6
    return %0 : !TowerF6
}

// CHECK-LABEL: @test_tower_negate
func.func @test_tower_negate(%arg0: !TowerF6) -> !TowerF6 {
    // CHECK: field.negate
    %0 = field.negate %arg0 : !TowerF6
    return %0 : !TowerF6
}

// CHECK-LABEL: @test_tower_double
func.func @test_tower_double(%arg0: !TowerF6) -> !TowerF6 {
    // CHECK: field.double
    %0 = field.double %arg0 : !TowerF6
    return %0 : !TowerF6
}

// CHECK-LABEL: @test_tower_mul
func.func @test_tower_mul(%arg0: !TowerF6, %arg1: !TowerF6) -> !TowerF6 {
    // CHECK: field.mul
    %0 = field.mul %arg0, %arg1 : !TowerF6
    return %0 : !TowerF6
}

// CHECK-LABEL: @test_tower_square
func.func @test_tower_square(%arg0: !TowerF6) -> !TowerF6 {
    // CHECK: field.square
    %0 = field.square %arg0 : !TowerF6
    return %0 : !TowerF6
}

//===----------------------------------------------------------------------===//
// Test tensor of tower extension field
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_tower_tensor
// CHECK-SAME: (%{{.*}}: tensor<4x[[TF6T:.*]]>) -> tensor<4x[[TF6T]]>
func.func @test_tower_tensor(%arg0: tensor<4x!TowerF6>) -> tensor<4x!TowerF6> {
    return %arg0 : tensor<4x!TowerF6>
}
