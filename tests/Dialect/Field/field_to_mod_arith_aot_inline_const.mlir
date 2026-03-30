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

// Verify --inline-constant-ops behavior: when true (default), ops with a
// constant operand skip AOT and are inlined; when false, all qualifying ops
// use AOT regardless.

// ---- inline-constant-ops=true (default) ----
// RUN: cat %S/../../bn254_defs.mlir %s \
// RUN:   | prime-ir-opt -field-to-mod-arith='lowering-mode=aot_runtime inline-constant-ops=true' \
// RUN:   | FileCheck %s --check-prefix=INLINE

// INLINE-LABEL: @test_mul_with_constant
// INLINE-NOT: call @ef_mul
// INLINE: return
func.func @test_mul_with_constant(%a: !QF) -> !QF {
    %c = field.constant [1, 2] : !QF
    %r = field.mul %a, %c : !QF
    return %r : !QF
}

// INLINE-LABEL: @test_mul_no_constant
// INLINE: call @ef_mul_bn254_bfx2(
func.func @test_mul_no_constant(%a: !QF, %b: !QF) -> !QF {
    %r = field.mul %a, %b : !QF
    return %r : !QF
}

// ---- inline-constant-ops=false ----
// RUN: cat %S/../../bn254_defs.mlir %s \
// RUN:   | prime-ir-opt -field-to-mod-arith='lowering-mode=aot_runtime inline-constant-ops=false' \
// RUN:   | FileCheck %s --check-prefix=ALWAYS

// ALWAYS-LABEL: @test_mul_with_constant
// ALWAYS: call @ef_mul_bn254_bfx2(
func.func @test_mul_with_constant_always(%a: !QF) -> !QF {
    %c = field.constant [1, 2] : !QF
    %r = field.mul %a, %c : !QF
    return %r : !QF
}

// ALWAYS-LABEL: @test_mul_no_constant
// ALWAYS: call @ef_mul_bn254_bfx2(
func.func @test_mul_no_constant_always(%a: !QF, %b: !QF) -> !QF {
    %r = field.mul %a, %b : !QF
    return %r : !QF
}
