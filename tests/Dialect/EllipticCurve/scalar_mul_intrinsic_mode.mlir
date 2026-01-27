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

// Test that scalar_mul operations use intrinsic functions when
// lowering-mode=intrinsic is specified, particularly for G2 points
// over extension fields.

// RUN: cat %S/../../bn254_field_defs.mlir %S/../../bn254_ec_mont_defs.mlir %s \
// RUN:   | prime-ir-opt -elliptic-curve-to-field="lowering-mode=intrinsic" \
// RUN:   | FileCheck %s --check-prefix=CHECK-INTRINSIC

// RUN: cat %S/../../bn254_field_defs.mlir %S/../../bn254_ec_mont_defs.mlir %s \
// RUN:   | prime-ir-opt -elliptic-curve-to-field="lowering-mode=auto" \
// RUN:   | FileCheck %s --check-prefix=CHECK-AUTO

// RUN: cat %S/../../bn254_field_defs.mlir %S/../../bn254_ec_mont_defs.mlir %s \
// RUN:   | prime-ir-opt -elliptic-curve-to-field="lowering-mode=inline" \
// RUN:   | FileCheck %s --check-prefix=CHECK-INLINE

// =============================================================================
// Test G1 scalar_mul (prime field points)
// =============================================================================

// CHECK-INTRINSIC-LABEL: @test_g1_scalar_mul
// CHECK-AUTO-LABEL: @test_g1_scalar_mul
// CHECK-INLINE-LABEL: @test_g1_scalar_mul
func.func @test_g1_scalar_mul(%point: !affine, %scalar: !SFm) -> !jacobian {
  // Intrinsic mode: uses intrinsic for all scalar_mul (direct call with high-level types)
  // CHECK-INTRINSIC: call @__prime_ir_ec_scalar_mul_jacobian_{{.*}}_mont

  // Auto mode: G1 (prime field) uses inline
  // CHECK-AUTO-NOT: call @__prime_ir_ec_scalar_mul
  // CHECK-AUTO: scf.while

  // Inline mode: always inline
  // CHECK-INLINE-NOT: call @__prime_ir_ec_scalar_mul
  // CHECK-INLINE: scf.while

  %result = elliptic_curve.scalar_mul %scalar, %point : !SFm, !affine -> !jacobian
  return %result : !jacobian
}

// =============================================================================
// Test G2 scalar_mul (extension field points)
// =============================================================================

// CHECK-INTRINSIC-LABEL: @test_g2_scalar_mul
// CHECK-AUTO-LABEL: @test_g2_scalar_mul
// CHECK-INLINE-LABEL: @test_g2_scalar_mul
func.func @test_g2_scalar_mul(%point: !g2affine, %scalar: !SFm) -> !g2jacobian {
  // Intrinsic mode: uses intrinsic for scalar_mul (direct call with high-level types)
  // CHECK-INTRINSIC: call @__prime_ir_ec_scalar_mul_jacobian_{{.*}}_mont

  // Auto mode: G2 (extension field) uses intrinsic
  // CHECK-AUTO: call @__prime_ir_ec_scalar_mul_jacobian_{{.*}}_mont

  // Inline mode: always inline (no function call)
  // CHECK-INLINE-NOT: call @__prime_ir_ec_scalar_mul
  // CHECK-INLINE: scf.while

  %result = elliptic_curve.scalar_mul %scalar, %point : !SFm, !g2affine -> !g2jacobian
  return %result : !g2jacobian
}

// =============================================================================
// Test G2 xyzz scalar_mul
// =============================================================================

// CHECK-INTRINSIC-LABEL: @test_g2_xyzz_scalar_mul
// CHECK-AUTO-LABEL: @test_g2_xyzz_scalar_mul
func.func @test_g2_xyzz_scalar_mul(%point: !g2xyzz, %scalar: !SFm) -> !g2xyzz {
  // CHECK-INTRINSIC: call @__prime_ir_ec_scalar_mul_xyzz_{{.*}}_mont
  // CHECK-AUTO: call @__prime_ir_ec_scalar_mul_xyzz_{{.*}}_mont
  %result = elliptic_curve.scalar_mul %scalar, %point : !SFm, !g2xyzz -> !g2xyzz
  return %result : !g2xyzz
}

// =============================================================================
// Check that intrinsic function definitions are generated at module level.
// The intrinsic function bodies contain high-level ops that are lowered
// by the same pass with inline mode (detected via __prime_ir_ prefix).
// =============================================================================

// CHECK-INTRINSIC-DAG: func.func private @__prime_ir_ec_scalar_mul_jacobian_
// CHECK-INTRINSIC-DAG: func.func private @__prime_ir_ec_scalar_mul_xyzz_

// CHECK-AUTO-DAG: func.func private @__prime_ir_ec_scalar_mul_jacobian_
// CHECK-AUTO-DAG: func.func private @__prime_ir_ec_scalar_mul_xyzz_
