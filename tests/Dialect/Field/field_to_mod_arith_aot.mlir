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

// Verify that FieldToModArith in AOT runtime mode emits func.call to
// pre-compiled AOT functions for expensive extension field operations
// (mul, square, inverse) while cheap ops (add, sub, negate) remain inline.

// RUN: cat %S/../../bn254_defs.mlir %s \
// RUN:   | prime-ir-opt -field-to-mod-arith='lowering-mode=aot_runtime' \
// RUN:   | FileCheck %s

// Verify private function declarations are emitted for expensive ops.
// CHECK-DAG: func.func private @ef_mul_bn254_bfx2(
// CHECK-DAG: func.func private @ef_mul_bn254_bfx2_mont(
// CHECK-DAG: func.func private @ef_square_bn254_bfx2(
// CHECK-DAG: func.func private @ef_inverse_bn254_bfx2(

// Verify NO declarations for cheap ops.
// CHECK-NOT: func.func private @ef_add_
// CHECK-NOT: func.func private @ef_sub_
// CHECK-NOT: func.func private @ef_negate_

// CHECK-LABEL: @test_aot_mul
// CHECK: call @ef_mul_bn254_bfx2(
func.func @test_aot_mul(%a: !QF, %b: !QF) -> !QF {
    %r = field.mul %a, %b : !QF
    return %r : !QF
}

// CHECK-LABEL: @test_aot_mul_mont
// CHECK: call @ef_mul_bn254_bfx2_mont(
func.func @test_aot_mul_mont(%a: !QFm, %b: !QFm) -> !QFm {
    %r = field.mul %a, %b : !QFm
    return %r : !QFm
}

// CHECK-LABEL: @test_aot_square
// CHECK: call @ef_square_bn254_bfx2(
func.func @test_aot_square(%a: !QF) -> !QF {
    %r = field.square %a : !QF
    return %r : !QF
}

// CHECK-LABEL: @test_aot_inverse
// CHECK: call @ef_inverse_bn254_bfx2(
func.func @test_aot_inverse(%a: !QF) -> !QF {
    %r = field.inverse %a : !QF
    return %r : !QF
}

// Cheap ops always inline (no AOT function call).
// CHECK-LABEL: @test_add_inlines
// CHECK-NOT: call @ef_
// CHECK: return
func.func @test_add_inlines(%a: !QF, %b: !QF) -> !QF {
    %r = field.add %a, %b : !QF
    return %r : !QF
}

// CHECK-LABEL: @test_sub_inlines
// CHECK-NOT: call @ef_
// CHECK: return
func.func @test_sub_inlines(%a: !QF, %b: !QF) -> !QF {
    %r = field.sub %a, %b : !QF
    return %r : !QF
}

// CHECK-LABEL: @test_negate_inlines
// CHECK-NOT: call @ef_
// CHECK: return
func.func @test_negate_inlines(%a: !QF) -> !QF {
    %r = field.negate %a : !QF
    return %r : !QF
}

// CHECK-LABEL: @test_double_inlines
// CHECK-NOT: call @ef_
// CHECK: return
func.func @test_double_inlines(%a: !QF) -> !QF {
    %r = field.double %a : !QF
    return %r : !QF
}

// Prime field ops should NOT use AOT (degree < 2).
// CHECK-LABEL: @test_prime_field_inlines
// CHECK-NOT: call @ef_
// CHECK: mod_arith.mul
func.func @test_prime_field_inlines(%a: !PF, %b: !PF) -> !PF {
    %r = field.mul %a, %b : !PF
    return %r : !PF
}
