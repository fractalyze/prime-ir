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
// RUN:   | prime-ir-opt -canonicalize \
// RUN:   | FileCheck %s -enable-var-scope

// CHECK-LABEL: @test_ed_add_self_is_double
// CHECK-SAME: (%[[ARG0:.*]]: [[EDEXT:.*]]) -> [[EDEXT]] {
func.func @test_ed_add_self_is_double(%point: !ed_extended) -> !ed_extended {
  // CHECK: %[[DOUBLE:.*]] = elliptic_curve.double %[[ARG0]] : [[EDEXT]] -> [[EDEXT]]
  // CHECK-NOT: elliptic_curve.add
  // CHECK: return %[[DOUBLE]] : [[EDEXT]]
  %sum = elliptic_curve.add %point, %point : !ed_extended, !ed_extended -> !ed_extended
  return %sum : !ed_extended
}

// CHECK-LABEL: @test_ed_add_after_neg_lhs
// CHECK-SAME: (%[[ARG0:.*]]: [[EDEXT:.*]], %[[ARG1:.*]]: [[EDEXT]]) -> [[EDEXT]] {
func.func @test_ed_add_after_neg_lhs(%point1: !ed_extended, %point2: !ed_extended) -> !ed_extended {
  // CHECK: %[[SUB:.*]] = elliptic_curve.sub %[[ARG1]], %[[ARG0]] : [[EDEXT]], [[EDEXT]] -> [[EDEXT]]
  // CHECK-NOT: elliptic_curve.add
  // CHECK-NOT: elliptic_curve.negate
  // CHECK: return %[[SUB]] : [[EDEXT]]
  %neg = elliptic_curve.negate %point1 : !ed_extended
  %sum = elliptic_curve.add %neg, %point2 : !ed_extended, !ed_extended -> !ed_extended
  return %sum : !ed_extended
}

// CHECK-LABEL: @test_ed_double_negate
// CHECK-SAME: (%[[ARG0:.*]]: [[EDEXT:.*]]) -> [[EDEXT]] {
func.func @test_ed_double_negate(%point: !ed_extended) -> !ed_extended {
  // CHECK: return %[[ARG0]] : [[EDEXT]]
  %neg1 = elliptic_curve.negate %point : !ed_extended
  %neg2 = elliptic_curve.negate %neg1 : !ed_extended
  return %neg2 : !ed_extended
}

// CHECK-LABEL: @test_ed_sub_after_neg_rhs
// CHECK-SAME: (%[[ARG0:.*]]: [[EDEXT:.*]], %[[ARG1:.*]]: [[EDEXT]]) -> [[EDEXT]] {
func.func @test_ed_sub_after_neg_rhs(%point1: !ed_extended, %point2: !ed_extended) -> !ed_extended {
  // CHECK: %[[ADD:.*]] = elliptic_curve.add %[[ARG0]], %[[ARG1]] : [[EDEXT]], [[EDEXT]] -> [[EDEXT]]
  // CHECK-NOT: elliptic_curve.sub
  // CHECK-NOT: elliptic_curve.negate
  // CHECK: return %[[ADD]] : [[EDEXT]]
  %neg = elliptic_curve.negate %point2 : !ed_extended
  %diff = elliptic_curve.sub %point1, %neg : !ed_extended, !ed_extended -> !ed_extended
  return %diff : !ed_extended
}
