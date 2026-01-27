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

//===----------------------------------------------------------------------===//
// SubOp patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_sub_lhs_after_add
// CHECK-SAME: (%[[ARG0:.*]]: [[JACOBIAN:.*]], %[[ARG1:.*]]: [[JACOBIAN:.*]]) -> [[JACOBIAN:.*]] {
func.func @test_sub_lhs_after_add(%point1: !jacobian, %point2: !jacobian) -> !jacobian {
  // (P + Q) - P -> Q
  // CHECK-NOT: elliptic_curve.add
  // CHECK-NOT: elliptic_curve.sub
  // CHECK: return %[[ARG1]] : [[JACOBIAN]]
  %sum = elliptic_curve.add %point1, %point2 : !jacobian, !jacobian -> !jacobian
  %diff = elliptic_curve.sub %sum, %point1 : !jacobian, !jacobian -> !jacobian
  return %diff : !jacobian
}

// CHECK-LABEL: @test_sub_rhs_after_add
// CHECK-SAME: (%[[ARG0:.*]]: [[JACOBIAN:.*]], %[[ARG1:.*]]: [[JACOBIAN:.*]]) -> [[JACOBIAN:.*]] {
func.func @test_sub_rhs_after_add(%point1: !jacobian, %point2: !jacobian) -> !jacobian {
  // (P + Q) - Q -> P
  // CHECK-NOT: elliptic_curve.add
  // CHECK-NOT: elliptic_curve.sub
  // CHECK: return %[[ARG0]] : [[JACOBIAN]]
  %sum = elliptic_curve.add %point1, %point2 : !jacobian, !jacobian -> !jacobian
  %diff = elliptic_curve.sub %sum, %point2 : !jacobian, !jacobian -> !jacobian
  return %diff : !jacobian
}

// CHECK-LABEL: @test_sub_lhs_after_sub
// CHECK-SAME: (%[[ARG0:.*]]: [[JACOBIAN:.*]], %[[ARG1:.*]]: [[JACOBIAN:.*]]) -> [[JACOBIAN:.*]] {
func.func @test_sub_lhs_after_sub(%point1: !jacobian, %point2: !jacobian) -> !jacobian {
  // (P - Q) - P -> -Q
  // CHECK: %[[NEG:.*]] = elliptic_curve.negate %[[ARG1]] : [[JACOBIAN]]
  // CHECK-NOT: elliptic_curve.sub
  // CHECK: return %[[NEG]] : [[JACOBIAN]]
  %diff1 = elliptic_curve.sub %point1, %point2 : !jacobian, !jacobian -> !jacobian
  %diff2 = elliptic_curve.sub %diff1, %point1 : !jacobian, !jacobian -> !jacobian
  return %diff2 : !jacobian
}

// CHECK-LABEL: @test_sub_after_neg_lhs
// CHECK-SAME: (%[[ARG0:.*]]: [[JACOBIAN:.*]], %[[ARG1:.*]]: [[JACOBIAN:.*]]) -> [[JACOBIAN:.*]] {
func.func @test_sub_after_neg_lhs(%point1: !jacobian, %point2: !jacobian) -> !jacobian {
  // (-P) - Q -> -(P + Q)
  // CHECK: %[[ADD:.*]] = elliptic_curve.add %[[ARG0]], %[[ARG1]] : [[JACOBIAN]], [[JACOBIAN]] -> [[JACOBIAN]]
  // CHECK: %[[NEG:.*]] = elliptic_curve.negate %[[ADD]] : [[JACOBIAN]]
  // CHECK-NOT: elliptic_curve.sub
  // CHECK: return %[[NEG]] : [[JACOBIAN]]
  %neg = elliptic_curve.negate %point1 : !jacobian
  %diff = elliptic_curve.sub %neg, %point2 : !jacobian, !jacobian -> !jacobian
  return %diff : !jacobian
}

//===----------------------------------------------------------------------===//
// ScalarMul strength reduction patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_scalar_mul_by_one
// CHECK-SAME: (%[[ARG0:.*]]: [[JACOBIAN:.*]]) -> [[JACOBIAN:.*]] {
func.func @test_scalar_mul_by_one(%point: !jacobian) -> !jacobian {
  // 1 * P -> P
  // TODO(chokobole): This pattern requires isScalarEqualTo to be implemented.
  // CHECK: elliptic_curve.scalar_mul
  %one = field.constant 1 : !SF
  %result = elliptic_curve.scalar_mul %one, %point : !SF, !jacobian -> !jacobian
  return %result : !jacobian
}

// CHECK-LABEL: @test_scalar_mul_by_zero
// CHECK-SAME: (%[[ARG0:.*]]: [[JACOBIAN:.*]]) -> [[JACOBIAN:.*]] {
func.func @test_scalar_mul_by_zero(%point: !jacobian) -> !jacobian {
  // 0 * P -> zero point (created via from_coords with identity coords)
  // TODO(chokobole): This pattern requires isScalarEqualTo to be implemented.
  // CHECK: elliptic_curve.scalar_mul
  %zero = field.constant 0 : !SF
  %result = elliptic_curve.scalar_mul %zero, %point : !SF, !jacobian -> !jacobian
  return %result : !jacobian
}

// CHECK-LABEL: @test_scalar_mul_by_two
// CHECK-SAME: (%[[ARG0:.*]]: [[JACOBIAN:.*]]) -> [[JACOBIAN:.*]] {
func.func @test_scalar_mul_by_two(%point: !jacobian) -> !jacobian {
  // 2 * P -> double(P)
  // TODO(chokobole): This pattern requires isScalarEqualTo to be implemented.
  // CHECK: elliptic_curve.scalar_mul
  %two = field.constant 2 : !SF
  %result = elliptic_curve.scalar_mul %two, %point : !SF, !jacobian -> !jacobian
  return %result : !jacobian
}

// CHECK-LABEL: @test_scalar_mul_by_neg_one
// CHECK-SAME: (%[[ARG0:.*]]: [[JACOBIAN:.*]]) -> [[JACOBIAN:.*]] {
func.func @test_scalar_mul_by_neg_one(%point: !jacobian) -> !jacobian {
  // (-1) * P -> -P
  // -1 mod SF = 21888242871839275222246405745257275088548364400416034343698204186575808495616
  // TODO(chokobole): This pattern requires isScalarNegativeOf to be implemented.
  // CHECK: elliptic_curve.scalar_mul
  %neg_one = field.constant 21888242871839275222246405745257275088548364400416034343698204186575808495616 : !SF
  %result = elliptic_curve.scalar_mul %neg_one, %point : !SF, !jacobian -> !jacobian
  return %result : !jacobian
}

//===----------------------------------------------------------------------===//
// ScalarMul distributivity patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_factor_scalar_mul_add
// CHECK-SAME: (%[[S:.*]]: [[SF:.*]], %[[P:.*]]: [[JACOBIAN:.*]], %[[Q:.*]]: [[JACOBIAN:.*]]) -> [[JACOBIAN:.*]] {
func.func @test_factor_scalar_mul_add(%s: !SF, %p: !jacobian, %q: !jacobian) -> !jacobian {
  // s*P + s*Q -> s * (P + Q)
  // CHECK: %[[ADD:.*]] = elliptic_curve.add %[[P]], %[[Q]] : [[JACOBIAN]], [[JACOBIAN]] -> [[JACOBIAN]]
  // CHECK: %[[MUL:.*]] = elliptic_curve.scalar_mul %[[S]], %[[ADD]] : [[SF]], [[JACOBIAN]] -> [[JACOBIAN]]
  // CHECK: return %[[MUL]] : [[JACOBIAN]]
  %sp = elliptic_curve.scalar_mul %s, %p : !SF, !jacobian -> !jacobian
  %sq = elliptic_curve.scalar_mul %s, %q : !SF, !jacobian -> !jacobian
  %sum = elliptic_curve.add %sp, %sq : !jacobian, !jacobian -> !jacobian
  return %sum : !jacobian
}

// CHECK-LABEL: @test_factor_scalar_mul_sub
// CHECK-SAME: (%[[S:.*]]: [[SF:.*]], %[[P:.*]]: [[JACOBIAN:.*]], %[[Q:.*]]: [[JACOBIAN:.*]]) -> [[JACOBIAN:.*]] {
func.func @test_factor_scalar_mul_sub(%s: !SF, %p: !jacobian, %q: !jacobian) -> !jacobian {
  // s*P - s*Q -> s * (P - Q)
  // CHECK: %[[SUB:.*]] = elliptic_curve.sub %[[P]], %[[Q]] : [[JACOBIAN]], [[JACOBIAN]] -> [[JACOBIAN]]
  // CHECK: %[[MUL:.*]] = elliptic_curve.scalar_mul %[[S]], %[[SUB]] : [[SF]], [[JACOBIAN]] -> [[JACOBIAN]]
  // CHECK: return %[[MUL]] : [[JACOBIAN]]
  %sp = elliptic_curve.scalar_mul %s, %p : !SF, !jacobian -> !jacobian
  %sq = elliptic_curve.scalar_mul %s, %q : !SF, !jacobian -> !jacobian
  %diff = elliptic_curve.sub %sp, %sq : !jacobian, !jacobian -> !jacobian
  return %diff : !jacobian
}

//===----------------------------------------------------------------------===//
// ScalarMul associativity patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_scalar_mul_assoc_left
// CHECK-SAME: (%[[S:.*]]: [[SF:.*]], %[[T:.*]]: [[SF:.*]], %[[P:.*]]: [[JACOBIAN:.*]]) -> [[JACOBIAN:.*]] {
func.func @test_scalar_mul_assoc_left(%s: !SF, %t: !SF, %p: !jacobian) -> !jacobian {
  // s * (t * P) -> (s * t) * P
  // CHECK: %[[PROD:.*]] = field.mul %[[S]], %[[T]] : [[SF]]
  // CHECK: %[[MUL:.*]] = elliptic_curve.scalar_mul %[[PROD]], %[[P]] : [[SF]], [[JACOBIAN]] -> [[JACOBIAN]]
  // CHECK: return %[[MUL]] : [[JACOBIAN]]
  %tp = elliptic_curve.scalar_mul %t, %p : !SF, !jacobian -> !jacobian
  %result = elliptic_curve.scalar_mul %s, %tp : !SF, !jacobian -> !jacobian
  return %result : !jacobian
}

//===----------------------------------------------------------------------===//
// Constant Folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_negate_fold
// CHECK-SAME: () -> [[JACOBIAN:.*]] {
func.func @test_negate_fold() -> !jacobian {
  // negate((1, 2, 1)) = (1, P-2, 1) where P is the base field modulus
  // P = 21888242871839275222246405745257275088696311157297823662689037894645226208583
  // P - 2 = 21888242871839275222246405745257275088696311157297823662689037894645226208581
  // CHECK: %[[C:.*]] = elliptic_curve.constant 1, 21888242871839275222246405745257275088696311157297823662689037894645226208581, 1 : [[JACOBIAN]]
  // CHECK-NOT: elliptic_curve.negate
  // CHECK: return %[[C]] : [[JACOBIAN]]
  %0 = elliptic_curve.constant 1, 2, 1 : !jacobian
  %1 = elliptic_curve.negate %0 : !jacobian
  return %1 : !jacobian
}

// CHECK-LABEL: @test_double_fold
// CHECK-SAME: () -> [[JACOBIAN:.*]] {
func.func @test_double_fold() -> !jacobian {
  // double((1, 2, 1)) computes the point doubling using Jacobian formulas
  // CHECK: %[[C:.*]] = elliptic_curve.constant
  // CHECK-NOT: elliptic_curve.double
  // CHECK: return %[[C]] : [[JACOBIAN]]
  %0 = elliptic_curve.constant 1, 2, 1 : !jacobian
  %1 = elliptic_curve.double %0 : !jacobian -> !jacobian
  return %1 : !jacobian
}

// CHECK-LABEL: @test_add_fold
// CHECK-SAME: () -> [[JACOBIAN:.*]] {
func.func @test_add_fold() -> !jacobian {
  // add((1, 2, 1), (3, 4, 1)) computes point addition
  // CHECK: %[[C:.*]] = elliptic_curve.constant
  // CHECK-NOT: elliptic_curve.add
  // CHECK: return %[[C]] : [[JACOBIAN]]
  %0 = elliptic_curve.constant 1, 2, 1 : !jacobian
  %1 = elliptic_curve.constant 3, 4, 1 : !jacobian
  %2 = elliptic_curve.add %0, %1 : !jacobian, !jacobian -> !jacobian
  return %2 : !jacobian
}

// CHECK-LABEL: @test_sub_fold
// CHECK-SAME: () -> [[JACOBIAN:.*]] {
func.func @test_sub_fold() -> !jacobian {
  // sub((1, 2, 1), (3, 4, 1)) computes point subtraction
  // CHECK: %[[C:.*]] = elliptic_curve.constant
  // CHECK-NOT: elliptic_curve.sub
  // CHECK: return %[[C]] : [[JACOBIAN]]
  %0 = elliptic_curve.constant 1, 2, 1 : !jacobian
  %1 = elliptic_curve.constant 3, 4, 1 : !jacobian
  %2 = elliptic_curve.sub %0, %1 : !jacobian, !jacobian -> !jacobian
  return %2 : !jacobian
}
