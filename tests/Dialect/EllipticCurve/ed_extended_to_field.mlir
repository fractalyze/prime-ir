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

// RUN: cat %S/../../ed25519_defs.mlir %s \
// RUN:   | prime-ir-opt -elliptic-curve-to-field \
// RUN:   | FileCheck %s -enable-var-scope

// Test: Edwards extended point addition lowers to field ops.
// CHECK-LABEL: @test_ed_extended_add
func.func @test_ed_extended_add(%p1: !ed_extended, %p2: !ed_extended) -> !ed_extended {
  // CHECK-NOT: elliptic_curve.add
  // CHECK: field.mul
  // CHECK: field.add
  %result = elliptic_curve.add %p1, %p2 : !ed_extended, !ed_extended -> !ed_extended
  return %result : !ed_extended
}

// Test: Edwards extended point doubling lowers to field ops.
// CHECK-LABEL: @test_ed_extended_double
func.func @test_ed_extended_double(%p1: !ed_extended) -> !ed_extended {
  // CHECK-NOT: elliptic_curve.double
  // CHECK: field.mul
  %result = elliptic_curve.double %p1 : !ed_extended -> !ed_extended
  return %result : !ed_extended
}

// Test: Edwards negation negates X and T (coords 0 and 3), not Y (coord 1).
// CHECK-LABEL: @test_ed_extended_negate
func.func @test_ed_extended_negate(%p1: !ed_extended) -> !ed_extended {
  // CHECK-NOT: elliptic_curve.negate
  // CHECK: elliptic_curve.to_coords
  // Two negate ops (for X and T)
  // CHECK: field.negate
  // CHECK: field.negate
  // CHECK: elliptic_curve.from_coords
  %result = elliptic_curve.negate %p1 : !ed_extended
  return %result : !ed_extended
}

// Test: Edwards subtraction lowers (via negate + add).
// CHECK-LABEL: @test_ed_extended_sub
func.func @test_ed_extended_sub(%p1: !ed_extended, %p2: !ed_extended) -> !ed_extended {
  // CHECK-NOT: elliptic_curve.sub
  %result = elliptic_curve.sub %p1, %p2 : !ed_extended, !ed_extended -> !ed_extended
  return %result : !ed_extended
}

// Test: Edwards constant point lowers.
// CHECK-LABEL: @test_ed_extended_constant
func.func @test_ed_extended_constant() {
  // CHECK-NOT: elliptic_curve.constant
  // CHECK: field.constant
  %gen = elliptic_curve.constant dense<[15112221349535400772501151409588531511454012693041857206046113283949847762202, 46316835694926478169428394003475163141307993866256225615783033603165251855960, 1, 0]> : !ed_extended
  return
}
