// Copyright 2025 The PrimeIR Authors.
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

// RUN: prime-ir-opt -canonicalize %s | FileCheck %s -enable-var-scope

!PF = !field.pf<7:i32>
!QF = !field.f2<!PF, 6:i32>

//===----------------------------------------------------------------------===//
// NegateOp constant folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_fold_negate
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_fold_negate() -> !QF {
  // CHECK: %[[C:.*]] = field.constant dense<[6, 5]> : [[T]]
  // -[1, 2] mod 7 = [6, 5]
  %0 = field.constant [1, 2] : !QF
  %1 = field.negate %0 : !QF
  // CHECK-NOT: field.negate
  // CHECK: return %[[C]] : [[T]]
  return %1 : !QF
}

//===----------------------------------------------------------------------===//
// DoubleOp constant folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_fold_double
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_fold_double() -> !QF {
  // CHECK: %[[C:.*]] = field.constant dense<[2, 4]> : [[T]]
  // 2 * [1, 2] mod 7 = [2, 4]
  %0 = field.constant [1, 2] : !QF
  %1 = field.double %0 : !QF
  // CHECK-NOT: field.double
  // CHECK: return %[[C]] : [[T]]
  return %1 : !QF
}

//===----------------------------------------------------------------------===//
// SquareOp constant folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_fold_square
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_fold_square() -> !QF {
  // CHECK: %[[C:.*]] = field.constant dense<4> : [[T]]
  // [a, b]² = [a² + ξ * b², 2 * a * b] where ξ = 6
  // [1, 2]² = [1 + 6 * 4, 2 * 1 * 2] = [25, 4] mod 7 = [4, 4]
  %0 = field.constant [1, 2] : !QF
  %1 = field.square %0 : !QF
  // CHECK-NOT: field.square
  // CHECK: return %[[C]] : [[T]]
  return %1 : !QF
}

//===----------------------------------------------------------------------===//
// InverseOp constant folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_fold_inverse
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_fold_inverse() -> !QF {
  // CHECK: %[[C:.*]] = field.constant dense<[3, 1]> : [[T]]
  // [a, b]⁻¹ = [a / norm, -b / norm] where norm = a² - ξ * b²
  // [1, 2]⁻¹: norm = 1 - 6 * 4 = -23 ≡ 5 (mod 7), 5⁻¹ = 3
  // = [1 * 3, -2 * 3] = [3, -6] = [3, 1]
  %0 = field.constant [1, 2] : !QF
  %1 = field.inverse %0 : !QF
  // CHECK-NOT: field.inverse
  // CHECK: return %[[C]] : [[T]]
  return %1 : !QF
}

//===----------------------------------------------------------------------===//
// AddOp constant folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_fold_add
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_fold_add() -> !QF {
  // CHECK: %[[C:.*]] = field.constant dense<[4, 6]> : [[T]]
  // [1, 2] + [3, 4] = [4, 6]
  %0 = field.constant [1, 2] : !QF
  %1 = field.constant [3, 4] : !QF
  %2 = field.add %0, %1 : !QF
  // CHECK-NOT: field.add
  // CHECK: return %[[C]] : [[T]]
  return %2 : !QF
}

//===----------------------------------------------------------------------===//
// SubOp constant folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_fold_sub
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_fold_sub() -> !QF {
  // CHECK: %[[C:.*]] = field.constant dense<5> : [[T]]
  // [1, 2] - [3, 4] = [-2, -2] mod 7 = [5, 5]
  %0 = field.constant [1, 2] : !QF
  %1 = field.constant [3, 4] : !QF
  %2 = field.sub %0, %1 : !QF
  // CHECK-NOT: field.sub
  // CHECK: return %[[C]] : [[T]]
  return %2 : !QF
}

//===----------------------------------------------------------------------===//
// MulOp constant folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_fold_mul
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_fold_mul() -> !QF {
  // CHECK: %[[C:.*]] = field.constant dense<[2, 3]> : [[T]]
  // [a₀, a₁] * [b₀, b₁] = [a₀b₀ + ξa₁b₁, a₀b₁ + a₁b₀] where ξ = 6
  // [1, 2] * [3, 4] = [1*3 + 6*2*4, 1*4 + 2*3] = [51, 10] mod 7 = [2, 3]
  %0 = field.constant [1, 2] : !QF
  %1 = field.constant [3, 4] : !QF
  %2 = field.mul %0, %1 : !QF
  // CHECK-NOT: field.mul
  // CHECK: return %[[C]] : [[T]]
  return %2 : !QF
}

//===----------------------------------------------------------------------===//
// ExtToCoeffsOp and ExtFromCoeffsOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_ext_to_coeffs_of_ext_from_coeffs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_ext_to_coeffs_of_ext_from_coeffs(%arg0: !QF) -> !QF {
  %0:2 = field.ext_to_coeffs %arg0 : (!QF) -> (!PF, !PF)
  %1 = field.ext_from_coeffs %0#0, %0#1 : (!PF, !PF) -> !QF
  // CHECK-NOT: field.ext_from_coeffs
  // CHECK: return %[[ARG0]] : [[T]]
  return %1 : !QF
}

// CHECK-LABEL: @test_ext_to_coeffs_of_swapped_ext_from_coeffs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_ext_to_coeffs_of_swapped_ext_from_coeffs(%arg0: !QF) -> !QF {
  %0:2 = field.ext_to_coeffs %arg0 : (!QF) -> (!PF, !PF)
  %1 = field.ext_from_coeffs %0#1, %0#0 : (!PF, !PF) -> !QF
  // CHECK: %[[EXT_TO_COEFFS:.*]]:2 = field.ext_to_coeffs %[[ARG0]] : ([[T]]) -> ([[T2:.*]], [[T2]])
  // CHECK: %[[EXT_FROM_COEFFS:.*]] = field.ext_from_coeffs %[[EXT_TO_COEFFS]]#1, %[[EXT_TO_COEFFS]]#0 : ([[T2]], [[T2]]) -> [[T]]
  // CHECK: return %[[EXT_FROM_COEFFS]] : [[T]]
  return %1 : !QF
}

// CHECK-LABEL: @test_ext_from_coeffs_of_ext_to_coeffs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> ([[T]], [[T]]) {
func.func @test_ext_from_coeffs_of_ext_to_coeffs(%arg0: !PF, %arg1: !PF) -> (!PF, !PF) {
  %0 = field.ext_from_coeffs %arg0, %arg1 : (!PF, !PF) -> (!QF)
  %1:2 = field.ext_to_coeffs %0 : (!QF) -> (!PF, !PF)
  // CHECK-NOT: field.ext_to_coeffs
  // CHECK: return %[[ARG0]], %[[ARG1]] : [[T]], [[T]]
  return %1#0, %1#1 : !PF, !PF
}
