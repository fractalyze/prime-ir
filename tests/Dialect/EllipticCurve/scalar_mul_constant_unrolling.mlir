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

// Tests that elliptic_curve.scalar_mul with constant scalars is unrolled into
// straight-line IR (no scf.while), while dynamic scalars produce a runtime loop.

// RUN: cat %S/../../bn254_field_defs.mlir %s \
// RUN:   | prime-ir-opt -elliptic-curve-to-field \
// RUN:   | FileCheck %s -enable-var-scope

#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PFm
!affine = !elliptic_curve.affine<#curve>
!jacobian = !elliptic_curve.jacobian<#curve>

// scalar = 0 → identity point, no runtime loop
// CHECK-LABEL: @test_scalar_mul_const_zero
// CHECK-NOT: scf.while
// CHECK: return
func.func @test_scalar_mul_const_zero(%point: !affine) -> !jacobian {
  %scalar = field.constant 0 : !SFm
  %result = elliptic_curve.scalar_mul %scalar, %point : !SFm, !affine -> !jacobian
  return %result : !jacobian
}

// scalar = 1 (1₂) → single add, no doubles, no runtime loop
// CHECK-LABEL: @test_scalar_mul_const_one
// CHECK-NOT: scf.while
// CHECK: return
func.func @test_scalar_mul_const_one(%point: !affine) -> !jacobian {
  %scalar = field.constant 1 : !SFm
  %result = elliptic_curve.scalar_mul %scalar, %point : !SFm, !affine -> !jacobian
  return %result : !jacobian
}

// scalar = 5 (101₂) → unrolled double-and-add, no runtime loop
// CHECK-LABEL: @test_scalar_mul_const_five
// CHECK-NOT: scf.while
// CHECK: return
func.func @test_scalar_mul_const_five(%point: !affine) -> !jacobian {
  %scalar = field.constant 5 : !SFm
  %result = elliptic_curve.scalar_mul %scalar, %point : !SFm, !affine -> !jacobian
  return %result : !jacobian
}

// Dynamic scalar: must produce scf.while loop
// CHECK-LABEL: @test_scalar_mul_dynamic
// CHECK: scf.while
func.func @test_scalar_mul_dynamic(%point: !affine, %scalar: !SFm) -> !jacobian {
  %result = elliptic_curve.scalar_mul %scalar, %point : !SFm, !affine -> !jacobian
  return %result : !jacobian
}
