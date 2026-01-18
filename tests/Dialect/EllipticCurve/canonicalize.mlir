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

// RUN: cat %S/../../bn254_field_defs.mlir %S/../../bn254_ec_defs.mlir %s \
// RUN:   | prime-ir-opt -canonicalize \
// RUN:   | FileCheck %s -enable-var-scope

//===----------------------------------------------------------------------===//
// Group operation canonicalization patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_add_self_is_double
// CHECK-SAME: (%[[ARG0:.*]]: [[JACOBIAN:.*]]) -> [[JACOBIAN:.*]] {
func.func @test_add_self_is_double(%point: !jacobian) -> !jacobian {
  // CHECK: %[[DOUBLE:.*]] = elliptic_curve.double %[[ARG0]] : [[JACOBIAN]] -> [[JACOBIAN]]
  // CHECK-NOT: elliptic_curve.add
  // CHECK: return %[[DOUBLE]] : [[JACOBIAN]]
  %sum = elliptic_curve.add %point, %point : !jacobian, !jacobian -> !jacobian
  return %sum : !jacobian
}

// CHECK-LABEL: @test_add_after_sub
// CHECK-SAME: (%[[ARG0:.*]]: [[JACOBIAN:.*]], %[[ARG1:.*]]: [[JACOBIAN:.*]]) -> [[JACOBIAN:.*]] {
func.func @test_add_after_sub(%point1: !jacobian, %point2: !jacobian) -> !jacobian {
  // CHECK-NOT: elliptic_curve.sub
  // CHECK-NOT: elliptic_curve.add
  // CHECK: return %[[ARG0]] : [[JACOBIAN]]
  %diff = elliptic_curve.sub %point1, %point2 : !jacobian, !jacobian -> !jacobian
  %sum = elliptic_curve.add %diff, %point2 : !jacobian, !jacobian -> !jacobian
  return %sum : !jacobian
}

// CHECK-LABEL: @test_add_after_neg_lhs
// CHECK-SAME: (%[[ARG0:.*]]: [[JACOBIAN:.*]], %[[ARG1:.*]]: [[JACOBIAN:.*]]) -> [[JACOBIAN:.*]] {
func.func @test_add_after_neg_lhs(%point1: !jacobian, %point2: !jacobian) -> !jacobian {
  // CHECK: %[[SUB:.*]] = elliptic_curve.sub %[[ARG1]], %[[ARG0]] : [[JACOBIAN]], [[JACOBIAN]] -> [[JACOBIAN]]
  // CHECK-NOT: elliptic_curve.add
  // CHECK-NOT: elliptic_curve.negate
  // CHECK: return %[[SUB]] : [[JACOBIAN]]
  %neg = elliptic_curve.negate %point1 : !jacobian
  %sum = elliptic_curve.add %neg, %point2 : !jacobian, !jacobian -> !jacobian
  return %sum : !jacobian
}

// CHECK-LABEL: @test_add_after_neg_rhs
// CHECK-SAME: (%[[ARG0:.*]]: [[JACOBIAN:.*]], %[[ARG1:.*]]: [[JACOBIAN:.*]]) -> [[JACOBIAN:.*]] {
func.func @test_add_after_neg_rhs(%point1: !jacobian, %point2: !jacobian) -> !jacobian {
  // CHECK: %[[SUB:.*]] = elliptic_curve.sub %[[ARG0]], %[[ARG1]] : [[JACOBIAN]], [[JACOBIAN]] -> [[JACOBIAN]]
  // CHECK-NOT: elliptic_curve.add
  // CHECK-NOT: elliptic_curve.negate
  // CHECK: return %[[SUB]] : [[JACOBIAN]]
  %neg = elliptic_curve.negate %point2 : !jacobian
  %sum = elliptic_curve.add %point1, %neg : !jacobian, !jacobian -> !jacobian
  return %sum : !jacobian
}

// CHECK-LABEL: @test_sub_after_neg_rhs
// CHECK-SAME: (%[[ARG0:.*]]: [[JACOBIAN:.*]], %[[ARG1:.*]]: [[JACOBIAN:.*]]) -> [[JACOBIAN:.*]] {
func.func @test_sub_after_neg_rhs(%point1: !jacobian, %point2: !jacobian) -> !jacobian {
  // CHECK: %[[ADD:.*]] = elliptic_curve.add %[[ARG0]], %[[ARG1]] : [[JACOBIAN]], [[JACOBIAN]] -> [[JACOBIAN]]
  // CHECK-NOT: elliptic_curve.sub
  // CHECK-NOT: elliptic_curve.negate
  // CHECK: return %[[ADD]] : [[JACOBIAN]]
  %neg = elliptic_curve.negate %point2 : !jacobian
  %diff = elliptic_curve.sub %point1, %neg : !jacobian, !jacobian -> !jacobian
  return %diff : !jacobian
}

// CHECK-LABEL: @test_sub_both_negated
// CHECK-SAME: (%[[ARG0:.*]]: [[JACOBIAN:.*]], %[[ARG1:.*]]: [[JACOBIAN:.*]]) -> [[JACOBIAN:.*]] {
func.func @test_sub_both_negated(%point1: !jacobian, %point2: !jacobian) -> !jacobian {
  // CHECK: %[[SUB:.*]] = elliptic_curve.sub %[[ARG1]], %[[ARG0]] : [[JACOBIAN]], [[JACOBIAN]] -> [[JACOBIAN]]
  // CHECK-NOT: elliptic_curve.negate
  // CHECK: return %[[SUB]] : [[JACOBIAN]]
  %neg1 = elliptic_curve.negate %point1 : !jacobian
  %neg2 = elliptic_curve.negate %point2 : !jacobian
  %diff = elliptic_curve.sub %neg1, %neg2 : !jacobian, !jacobian -> !jacobian
  return %diff : !jacobian
}

// CHECK-LABEL: @test_add_both_negated
// CHECK-SAME: (%[[ARG0:.*]]: [[JACOBIAN:.*]], %[[ARG1:.*]]: [[JACOBIAN:.*]]) -> [[JACOBIAN:.*]] {
func.func @test_add_both_negated(%point1: !jacobian, %point2: !jacobian) -> !jacobian {
  // CHECK: %[[ADD:.*]] = elliptic_curve.add %[[ARG0]], %[[ARG1]] : [[JACOBIAN]], [[JACOBIAN]] -> [[JACOBIAN]]
  // CHECK: %[[NEG_RESULT:.*]] = elliptic_curve.negate %[[ADD]] : [[JACOBIAN]]
  // CHECK-NOT: elliptic_curve.add
  // CHECK: return %[[NEG_RESULT]] : [[JACOBIAN]]
  %neg1 = elliptic_curve.negate %point1 : !jacobian
  %neg2 = elliptic_curve.negate %point2 : !jacobian
  %sum = elliptic_curve.add %neg1, %neg2 : !jacobian, !jacobian -> !jacobian
  return %sum : !jacobian
}
