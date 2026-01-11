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
!EF2 = !field.ef<2x!PF, 6:i32>
!EF3 = !field.ef<3x!PF, 2:i32>
!EF4 = !field.ef<4x!PF, 2:i32>

// CHECK-LABEL: @test_lower_create_ef2
func.func @test_lower_create_ef2(%arg0: !PF, %arg1: !PF) -> !EF2 {
    // CHECK: field.ext_from_coeffs
    %0 = field.create [%arg0, %arg1] : !EF2
    return %0 : !EF2
}

// CHECK-LABEL: @test_lower_create_ef3
func.func @test_lower_create_ef3(%arg0: !PF, %arg1: !PF, %arg2: !PF) -> !EF3 {
    // CHECK: field.ext_from_coeffs
    %0 = field.create [%arg0, %arg1, %arg2] : !EF3
    return %0 : !EF3
}

// CHECK-LABEL: @test_lower_create_ef4
func.func @test_lower_create_ef4(%arg0: !PF, %arg1: !PF, %arg2: !PF, %arg3: !PF) -> !EF4 {
    // CHECK: field.ext_from_coeffs
    %0 = field.create [%arg0, %arg1, %arg2, %arg3] : !EF4
    return %0 : !EF4
}
