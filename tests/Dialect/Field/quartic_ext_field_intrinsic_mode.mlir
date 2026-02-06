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
// See the License for the specific Language governing permissions and
// limitations under the License.
// ==============================================================================

// Test that quartic extension field operations use intrinsic functions when
// lowering-mode=intrinsic is specified.

// RUN: prime-ir-opt -field-to-mod-arith="lowering-mode=intrinsic" %s | FileCheck %s
// RUN: prime-ir-opt -field-to-mod-arith="lowering-mode=auto" %s | FileCheck %s

// Use the same type aliases as existing quartic tests (modulus > 120 required for Toom-Cook)
!PF = !field.pf<127:i32>
!QF = !field.ef<4x!PF, 2:i32>

// CHECK-LABEL: @test_intrinsic_mul
func.func @test_intrinsic_mul(%arg0: !QF, %arg1: !QF) -> !QF {
    // Intrinsic mode generates a direct call with high-level types
    // CHECK: call @__prime_ir_ext4_mul_4_127(%arg0, %arg1)
    %0 = field.mul %arg0, %arg1 : !QF
    return %0 : !QF
}

// CHECK-LABEL: @test_intrinsic_square
func.func @test_intrinsic_square(%arg0: !QF) -> !QF {
    // CHECK: call @__prime_ir_ext4_square_4_127(%arg0)
    %0 = field.square %arg0 : !QF
    return %0 : !QF
}

// CHECK-LABEL: @test_intrinsic_inverse
func.func @test_intrinsic_inverse(%arg0: !QF) -> !QF {
    // CHECK: call @__prime_ir_ext4_inverse_4_127(%arg0)
    %inv = field.inverse %arg0 : !QF
    return %inv : !QF
}

// Check that intrinsic function definitions are generated at module level.
// The intrinsic function bodies contain high-level ops that are lowered
// by the same pass with inline mode (detected via __prime_ir_ prefix).
// CHECK-DAG: func.func private @__prime_ir_ext4_mul_4_127
// CHECK-DAG: func.func private @__prime_ir_ext4_square_4_127
// CHECK-DAG: func.func private @__prime_ir_ext4_inverse_4_127
