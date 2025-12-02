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

#xi = #field.pf.elem<2:i32> : !field.pf<7:i32>
!PF = !field.pf<7:i32>
!PFm = !field.pf<7:i32, true>
#xi_mont = #field.pf.elem<1:i32> : !PFm
!CF = !field.f3<!PF, #xi>
!CFm = !field.f3<!PFm, #xi_mont>

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
    // CH-SQR2 algorithm uses 3 squares
    // CHECK: mod_arith.constant 2 : !z7_i32
    // CHECK: field.ext_to_coeffs
    // CHECK: mod_arith.square
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.square
    // CHECK: mod_arith.mul
    // CHECK: mod_arith.square
    // CHECK-COUNT-2: mod_arith.mul
    // CHECK: field.ext_from_coeffs
    %0 = field.square %arg0 : !CF
    return %0 : !CF
}

// CHECK-LABEL: @test_lower_inverse
func.func @test_lower_inverse(%arg0: !CF) -> !CF {
    // CHECK: field.ext_to_coeffs
    // CHECK: mod_arith.square
    // CHECK-COUNT-2: mod_arith.mul
    // CHECK: mod_arith.square
    // CHECK-COUNT-2: mod_arith.mul
    // CHECK: mod_arith.square
    // CHECK-COUNT-5: mod_arith.mul
    // CHECK: mod_arith.inverse
    // CHECK-COUNT-3: mod_arith.mul
    // CHECK: field.ext_from_coeffs
    %inv = field.inverse %arg0 : !CF
    return %inv : !CF
}

// CHECK-LABEL: @test_lower_f3_constant
func.func @test_lower_f3_constant(%arg0: !PF, %arg1: !PF, %arg2: !PF) -> !CF {
    // CHECK: field.ext_from_coeffs
    %0 = field.f3.constant %arg0, %arg1, %arg2 : !CF
    return %0 : !CF
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
