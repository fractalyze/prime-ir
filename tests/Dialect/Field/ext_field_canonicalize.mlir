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
!PFm = !field.pf<7:i32, true>
!QF = !field.ef<2x!PF, 6:i32>
!QFm = !field.ef<2x!PFm, 6:i32>

// Tower extension: Fp6 = (Fp2)³ where Fp6 = Fp2[w]/(w³ - 2)
!Fp6 = !field.ef<3x!QF, 2:i32>

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

// CHECK-LABEL: @test_sub_self_is_zero
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_sub_self_is_zero(%arg0: !QF) -> !QF {
  // CHECK: %[[C:.*]] = field.constant dense<0> : [[T]]
  %0 = field.sub %arg0, %arg0 : !QF
  // CHECK-NOT: field.sub
  // CHECK: return %[[C]] : [[T]]
  return %0 : !QF
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

// CHECK-LABEL: @test_ext_from_coeffs_of_ext_to_coeffs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_ext_from_coeffs_of_ext_to_coeffs(%arg0: !QF) -> !QF {
  %0:2 = field.ext_to_coeffs %arg0 : (!QF) -> (!PF, !PF)
  %1 = field.ext_from_coeffs %0#0, %0#1 : (!PF, !PF) -> !QF
  // CHECK-NOT: field.ext_from_coeffs
  // CHECK: return %[[ARG0]] : [[T]]
  return %1 : !QF
}

// CHECK-LABEL: @test_swapped_ext_from_coeffs_of_ext_to_coeffs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_swapped_ext_from_coeffs_of_ext_to_coeffs(%arg0: !QF) -> !QF {
  %0:2 = field.ext_to_coeffs %arg0 : (!QF) -> (!PF, !PF)
  %1 = field.ext_from_coeffs %0#1, %0#0 : (!PF, !PF) -> !QF
  // CHECK: %[[EXT_TO_COEFFS:.*]]:2 = field.ext_to_coeffs %[[ARG0]] : ([[T]]) -> ([[T2:.*]], [[T2]])
  // CHECK: %[[EXT_FROM_COEFFS:.*]] = field.ext_from_coeffs %[[EXT_TO_COEFFS]]#1, %[[EXT_TO_COEFFS]]#0 : ([[T2]], [[T2]]) -> [[T]]
  // CHECK: return %[[EXT_FROM_COEFFS]] : [[T]]
  return %1 : !QF
}

// CHECK-LABEL: @test_ext_to_coeffs_of_ext_from_coeffs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> ([[T]], [[T]]) {
func.func @test_ext_to_coeffs_of_ext_from_coeffs(%arg0: !PF, %arg1: !PF) -> (!PF, !PF) {
  %0 = field.ext_from_coeffs %arg0, %arg1 : (!PF, !PF) -> (!QF)
  %1:2 = field.ext_to_coeffs %0 : (!QF) -> (!PF, !PF)
  // CHECK-NOT: field.ext_to_coeffs
  // CHECK: return %[[ARG0]], %[[ARG1]] : [[T]], [[T]]
  return %1#0, %1#1 : !PF, !PF
}

//===----------------------------------------------------------------------===//
// CmpOp constant folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_cmp_fold
// CHECK-SAME: () -> i1 {
func.func @test_cmp_fold() -> i1 {
  // CHECK: %[[C:.*]] = arith.constant true
  %0 = field.constant [1, 2] : !QF
  %1 = field.constant [1, 2] : !QF
  %2 = field.cmp eq, %0, %1 : !QF
  // CHECK-NOT: field.cmp
  // CHECK: return %[[C]]
  return %2 : i1
}

//===----------------------------------------------------------------------===//
// Tensor of extension field constants (parsing round-trip)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_tensor_ext_field_constant
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_ext_field_constant() -> tensor<2x!QF> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[}}[1, 2], [3, 4]{{\]}}> : [[T]]
  %0 = field.constant dense<[[1, 2], [3, 4]]> : tensor<2x!QF>
  // CHECK: return %[[C]] : [[T]]
  return %0 : tensor<2x!QF>
}

// CHECK-LABEL: @test_tensor_ext_field_splat_constant
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_ext_field_splat_constant() -> tensor<2x!QF> {
  // Splat constant: all elements are [1, 1]
  // CHECK: %[[C:.*]] = field.constant dense<1> : [[T]]
  %0 = field.constant dense<1> : tensor<2x!QF>
  // CHECK: return %[[C]] : [[T]]
  return %0 : tensor<2x!QF>
}

//===----------------------------------------------------------------------===//
// Tensor of extension field constant folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_tensor_ext_field_fold_negate
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_ext_field_fold_negate() -> tensor<2x!QF> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[}}[6, 5], [4, 3]{{\]}}> : [[T]]
  // -[[1, 2], [3, 4]] mod 7 = [[6, 5], [4, 3]]
  %0 = field.constant dense<[[1, 2], [3, 4]]> : tensor<2x!QF>
  %1 = field.negate %0 : tensor<2x!QF>
  // CHECK-NOT: field.negate
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!QF>
}

// CHECK-LABEL: @test_tensor_ext_field_fold_double
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_ext_field_fold_double() -> tensor<2x!QF> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[}}[2, 4], [6, 1]{{\]}}> : [[T]]
  // 2 * [[1, 2], [3, 4]] mod 7 = [[2, 4], [6, 8 mod 7]] = [[2, 4], [6, 1]]
  %0 = field.constant dense<[[1, 2], [3, 4]]> : tensor<2x!QF>
  %1 = field.double %0 : tensor<2x!QF>
  // CHECK-NOT: field.double
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!QF>
}

// CHECK-LABEL: @test_tensor_ext_field_fold_square
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_ext_field_fold_square() -> tensor<2x!QF> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[}}[4, 4], [0, 3]{{\]}}> : [[T]]
  // [a, b]² = [a² + ξ * b², 2 * a * b] where ξ = 6
  // [1, 2]² = [1 + 6 * 4, 2 * 1 * 2] = [25, 4] mod 7 = [4, 4]
  // [3, 4]² = [9 + 6 * 16, 2 * 3 * 4] = [105, 24] mod 7 = [0, 3]
  %0 = field.constant dense<[[1, 2], [3, 4]]> : tensor<2x!QF>
  %1 = field.square %0 : tensor<2x!QF>
  // CHECK-NOT: field.square
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!QF>
}

// CHECK-LABEL: @test_tensor_ext_field_fold_inverse
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_ext_field_fold_inverse() -> tensor<2x!QF> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[}}[3, 1], [6, 6]{{\]}}> : [[T]]
  // [a, b]⁻¹ = [a / norm, -b / norm] where norm = a² - ξ * b²
  // [1, 2]⁻¹: norm = 1 - 6 * 4 = -23 ≡ 5 (mod 7), 5⁻¹ = 3
  //           = [1 * 3, -2 * 3] = [3, -6] = [3, 1]
  // [3, 4]⁻¹: norm = 9 - 6 * 16 = -87 ≡ 4 (mod 7), 4⁻¹ = 2
  //           = [3 * 2, -4 * 2] = [6, -8] = [6, 6]
  %0 = field.constant dense<[[1, 2], [3, 4]]> : tensor<2x!QF>
  %1 = field.inverse %0 : tensor<2x!QF>
  // CHECK-NOT: field.inverse
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!QF>
}

// CHECK-LABEL: @test_tensor_ext_field_fold_add
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_ext_field_fold_add() -> tensor<2x!QF> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[}}[4, 6], [4, 6]{{\]}}> : [[T]]
  // [[1, 2], [3, 4]] + [[3, 4], [1, 2]] = [[4, 6], [4, 6]]
  %0 = field.constant dense<[[1, 2], [3, 4]]> : tensor<2x!QF>
  %1 = field.constant dense<[[3, 4], [1, 2]]> : tensor<2x!QF>
  %2 = field.add %0, %1 : tensor<2x!QF>
  // CHECK-NOT: field.add
  // CHECK: return %[[C]] : [[T]]
  return %2 : tensor<2x!QF>
}

// CHECK-LABEL: @test_tensor_ext_field_fold_sub
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_ext_field_fold_sub() -> tensor<2x!QF> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[}}[5, 5], [2, 2]{{\]}}> : [[T]]
  // [[1, 2], [3, 4]] - [[3, 4], [1, 2]] = [[-2, -2], [2, 2]] mod 7 = [[5, 5], [2, 2]]
  %0 = field.constant dense<[[1, 2], [3, 4]]> : tensor<2x!QF>
  %1 = field.constant dense<[[3, 4], [1, 2]]> : tensor<2x!QF>
  %2 = field.sub %0, %1 : tensor<2x!QF>
  // CHECK-NOT: field.sub
  // CHECK: return %[[C]] : [[T]]
  return %2 : tensor<2x!QF>
}

// CHECK-LABEL: @test_tensor_ext_field_fold_mul
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_ext_field_fold_mul() -> tensor<2x!QF> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[}}[2, 3], [2, 3]{{\]}}> : [[T]]
  // [a₀, a₁] * [b₀, b₁] = [a₀b₀ + ξa₁b₁, a₀b₁ + a₁b₀] where ξ = 6
  // [1, 2] * [3, 4] = [1*3 + 6*2*4, 1*4 + 2*3] = [51, 10] mod 7 = [2, 3]
  // [3, 4] * [1, 2] = [3*1 + 6*4*2, 3*2 + 4*1] = [51, 10] mod 7 = [2, 3]
  %0 = field.constant dense<[[1, 2], [3, 4]]> : tensor<2x!QF>
  %1 = field.constant dense<[[3, 4], [1, 2]]> : tensor<2x!QF>
  %2 = field.mul %0, %1 : tensor<2x!QF>
  // CHECK-NOT: field.mul
  // CHECK: return %[[C]] : [[T]]
  return %2 : tensor<2x!QF>
}

//===----------------------------------------------------------------------===//
// Tensor operation constant folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_ext_tensor_extract_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_ext_tensor_extract_fold() -> !QF {
  // CHECK: %[[C:.*]] = field.constant dense<[3, 4]> : [[T]]
  // CHECK-NOT: tensor.extract
  // CHECK: return %[[C]] : [[T]]
  %c1 = arith.constant 1 : index
  %0 = field.constant dense<[[1, 2], [3, 4]]> : tensor<2x!QF>
  %1 = tensor.extract %0[%c1] : tensor<2x!QF>
  return %1 : !QF
}

// CHECK-LABEL: @test_ext_tensor_extract_splat_0d
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_ext_tensor_extract_splat_0d() -> !QF {
  // Extracting from a 0-d splat EF tensor should produce a valid EF constant.
  // CHECK: %[[C:.*]] = field.constant dense<1> : [[T]]
  // CHECK-NOT: tensor.extract
  // CHECK: return %[[C]] : [[T]]
  %0 = field.constant dense<1> : tensor<!QF>
  %1 = tensor.extract %0[] : tensor<!QF>
  return %1 : !QF
}

// CHECK-LABEL: @test_ext_tensor_extract_splat_1d
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_ext_tensor_extract_splat_1d() -> !QF {
  // Extracting from a 1-d splat EF tensor should produce a valid EF constant.
  // CHECK: %[[C:.*]] = field.constant dense<3> : [[T]]
  // CHECK-NOT: tensor.extract
  // CHECK: return %[[C]] : [[T]]
  %c0 = arith.constant 0 : index
  %0 = field.constant dense<3> : tensor<2x!QF>
  %1 = tensor.extract %0[%c0] : tensor<2x!QF>
  return %1 : !QF
}

// CHECK-LABEL: @test_tower_tensor_extract_splat
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tower_tensor_extract_splat() -> !Fp6 {
  // Extracting from a splat tower EF tensor should produce a valid EF constant.
  // CHECK: %[[C:.*]] = field.constant dense<2> : [[T]]
  // CHECK-NOT: tensor.extract
  // CHECK: return %[[C]] : [[T]]
  %c0 = arith.constant 0 : index
  %0 = field.constant dense<2> : tensor<2x!Fp6>
  %1 = tensor.extract %0[%c0] : tensor<2x!Fp6>
  return %1 : !Fp6
}


// CHECK-LABEL: @test_tower_tensor_extract_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tower_tensor_extract_fold() -> !Fp6 {
  // CHECK: %[[C:.*]] = field.constant dense<[6, 5, 4, 3, 2, 1]> : [[T]]
  // CHECK-NOT: tensor.extract
  // CHECK: return %[[C]] : [[T]]
  %c1 = arith.constant 1 : index
  %0 = field.constant dense<[[[1, 2], [3, 4], [5, 6]], [[6, 5], [4, 3], [2, 1]]]> : tensor<2x!Fp6>
  %1 = tensor.extract %0[%c1] : tensor<2x!Fp6>
  return %1 : !Fp6
}

// CHECK-LABEL: @test_ext_tensor_splat_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_ext_tensor_splat_fold() -> tensor<4x!QF> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[\[}}3, 5], [3, 5], [3, 5], [3, 5{{\]\]}}> : [[T]]
  // CHECK-NOT: tensor.splat
  // CHECK: return %[[C]] : [[T]]
  %0 = field.constant [3, 5] : !QF
  %1 = tensor.splat %0 : tensor<4x!QF>
  return %1 : tensor<4x!QF>
}

// CHECK-LABEL: @test_tower_tensor_splat_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tower_tensor_splat_fold() -> tensor<2x!Fp6> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[\[\[}}1, 2], [3, 4], [5, 6{{\]\]}}, {{\[\[}}1, 2], [3, 4], [5, 6{{\]\]\]}}> : [[T]]
  // CHECK-NOT: tensor.splat
  // CHECK: return %[[C]] : [[T]]
  %0 = field.constant [1, 2, 3, 4, 5, 6] : !Fp6
  %1 = tensor.splat %0 : tensor<2x!Fp6>
  return %1 : tensor<2x!Fp6>
}


// CHECK-LABEL: @test_ext_tensor_from_elements_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_ext_tensor_from_elements_fold() -> tensor<2x!QF> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[\[}}1, 2], [3, 4{{\]\]}}> : [[T]]
  // CHECK-NOT: tensor.from_elements
  // CHECK: return %[[C]] : [[T]]
  %0 = field.constant [1, 2] : !QF
  %1 = field.constant [3, 4] : !QF
  %2 = tensor.from_elements %0, %1 : tensor<2x!QF>
  return %2 : tensor<2x!QF>
}

// CHECK-LABEL: @test_tower_tensor_from_elements_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tower_tensor_from_elements_fold() -> tensor<2x!Fp6> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[\[\[}}1, 2], [3, 4], [5, 6{{\]\]}}, {{\[\[}}6, 5], [4, 3], [2, 1{{\]\]\]}}> : [[T]]
  // CHECK-NOT: tensor.from_elements
  // CHECK: return %[[C]] : [[T]]
  %0 = field.constant [1, 2, 3, 4, 5, 6] : !Fp6
  %1 = field.constant [6, 5, 4, 3, 2, 1] : !Fp6
  %2 = tensor.from_elements %0, %1 : tensor<2x!Fp6>
  return %2 : tensor<2x!Fp6>
}

// CHECK-LABEL: @test_ext_collapse_shape_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_ext_collapse_shape_fold() -> tensor<4x!QF> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[\[}}1, 2], [3, 4], [5, 6], [0, 1{{\]\]}}> : [[T]]
  // CHECK-NOT: tensor.collapse_shape
  // CHECK: return %[[C]] : [[T]]
  %0 = field.constant dense<[[[1, 2], [3, 4]], [[5, 6], [0, 1]]]> : tensor<2x2x!QF>
  %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<2x2x!QF> into tensor<4x!QF>
  return %1 : tensor<4x!QF>
}

// CHECK-LABEL: @test_tower_collapse_shape_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tower_collapse_shape_fold() -> tensor<4x!Fp6> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[\[\[}}1, 2], [3, 4], [5, 6{{\]\]}}, {{\[\[}}6, 5], [4, 3], [2, 1{{\]\]}}, {{\[\[}}0, 1], [2, 3], [4, 5{{\]\]}}, {{\[\[}}5, 4], [3, 2], [1, 0{{\]\]\]}}> : [[T]]
  // CHECK-NOT: tensor.collapse_shape
  // CHECK: return %[[C]] : [[T]]
  %0 = field.constant dense<[[[[1, 2], [3, 4], [5, 6]], [[6, 5], [4, 3], [2, 1]]], [[[0, 1], [2, 3], [4, 5]], [[5, 4], [3, 2], [1, 0]]]]> : tensor<2x2x!Fp6>
  %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<2x2x!Fp6> into tensor<4x!Fp6>
  return %1 : tensor<4x!Fp6>
}

// CHECK-LABEL: @test_ext_expand_shape_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_ext_expand_shape_fold() -> tensor<2x2x!QF> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[\[\[}}1, 2], [3, 4{{\]\]}}, {{\[\[}}5, 6], [0, 1{{\]\]\]}}> : [[T]]
  // CHECK-NOT: tensor.expand_shape
  // CHECK: return %[[C]] : [[T]]
  %0 = field.constant dense<[[1, 2], [3, 4], [5, 6], [0, 1]]> : tensor<4x!QF>
  %1 = tensor.expand_shape %0 [[0, 1]] output_shape [2, 2] : tensor<4x!QF> into tensor<2x2x!QF>
  return %1 : tensor<2x2x!QF>
}

// CHECK-LABEL: @test_tower_expand_shape_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tower_expand_shape_fold() -> tensor<2x2x!Fp6> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[\[\[\[}}1, 2], [3, 4], [5, 6{{\]\]}}, {{\[\[}}6, 5], [4, 3], [2, 1{{\]\]\]}}, {{\[\[\[}}0, 1], [2, 3], [4, 5{{\]\]}}, {{\[\[}}5, 4], [3, 2], [1, 0{{\]\]\]\]}}> : [[T]]
  // CHECK-NOT: tensor.expand_shape
  // CHECK: return %[[C]] : [[T]]
  %0 = field.constant dense<[[[1, 2], [3, 4], [5, 6]], [[6, 5], [4, 3], [2, 1]], [[0, 1], [2, 3], [4, 5]], [[5, 4], [3, 2], [1, 0]]]> : tensor<4x!Fp6>
  %1 = tensor.expand_shape %0 [[0, 1]] output_shape [2, 2] : tensor<4x!Fp6> into tensor<2x2x!Fp6>
  return %1 : tensor<2x2x!Fp6>
}

//===----------------------------------------------------------------------===//
// Tensor splat constant folding (reshape, gather, extract_slice)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_ext_reshape_splat_constant_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_ext_reshape_splat_constant_fold() -> tensor<2x2x!QF> {
  %0 = field.constant dense<5> : tensor<4x!QF>
  %shape = arith.constant dense<[2, 2]> : tensor<2xi32>
  %1 = tensor.reshape %0(%shape) : (tensor<4x!QF>, tensor<2xi32>) -> tensor<2x2x!QF>
  // CHECK: %[[C:.*]] = field.constant dense<5> : [[T]]
  // CHECK-NOT: tensor.reshape
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x2x!QF>
}

// CHECK-LABEL: @test_ext_extract_slice_splat_constant_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_ext_extract_slice_splat_constant_fold() -> tensor<2x!QF> {
  %0 = field.constant dense<3> : tensor<4x!QF>
  %1 = tensor.extract_slice %0[1] [2] [1] : tensor<4x!QF> to tensor<2x!QF>
  // CHECK: %[[C:.*]] = field.constant dense<3> : [[T]]
  // CHECK-NOT: tensor.extract_slice
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!QF>
}

// CHECK-LABEL: @test_ext_gather_splat_constant_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_ext_gather_splat_constant_fold() -> tensor<2x1x!QF> {
  %0 = field.constant dense<3> : tensor<4x!QF>
  %indices = arith.constant dense<[[0], [2]]> : tensor<2x1xindex>
  %1 = tensor.gather %0[%indices] gather_dims([0]) : (tensor<4x!QF>, tensor<2x1xindex>) -> tensor<2x1x!QF>
  // CHECK: %[[C:.*]] = field.constant dense<3> : [[T]]
  // CHECK-NOT: tensor.gather
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x1x!QF>
}

//===----------------------------------------------------------------------===//
// Identity operation folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_add_tensor_zero_is_self
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_add_tensor_zero_is_self(%arg0: tensor<2x!QF>) -> tensor<2x!QF> {
  %0 = field.constant dense<0> : tensor<2x!QF>
  %1 = field.add %arg0, %0 : tensor<2x!QF>
  // CHECK: return %[[ARG0]] : [[T]]
  return %1 : tensor<2x!QF>
}

// CHECK-LABEL: @test_mul_tensor_by_zero_is_zero
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_tensor_by_zero_is_zero(%arg0: tensor<2x!QF>) -> tensor<2x!QF> {
  %0 = field.constant dense<0> : tensor<2x!QF>
  %1 = field.mul %arg0, %0 : tensor<2x!QF>
  // CHECK: %[[C:.*]] = field.constant dense<0> : [[T]]
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!QF>
}

// CHECK-LABEL: @test_mul_tensor_by_one_is_self
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_tensor_by_one_is_self(%arg0: tensor<2x!QF>) -> tensor<2x!QF> {
  // Multiplicative identity for QF is [1, 0] (1 in base field, 0 for other coefficients)
  %0 = field.constant dense<[[1, 0], [1, 0]]> : tensor<2x!QF>
  %1 = field.mul %arg0, %0 : tensor<2x!QF>
  // CHECK: return %[[ARG0]] : [[T]]
  return %1 : tensor<2x!QF>
}

//===----------------------------------------------------------------------===//
// Strength reduction (mul by small constant -> double/neg)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_mul_by_two_is_double
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_by_two_is_double(%arg0: !QF) -> !QF {
  // x * 2 -> double(x) where 2 = [2, 0] in EF
  %c2 = field.constant [2, 0] : !QF
  %0 = field.mul %arg0, %c2 : !QF
  // CHECK-NOT: field.mul
  // CHECK: %[[D:.*]] = field.double %[[ARG0]] : [[T]]
  // CHECK: return %[[D]] : [[T]]
  return %0 : !QF
}

// CHECK-LABEL: @test_mul_by_three
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_by_three(%arg0: !QF) -> !QF {
  // x * 3 -> x + double(x)
  %c3 = field.constant [3, 0] : !QF
  %0 = field.mul %arg0, %c3 : !QF
  // CHECK-NOT: field.mul
  // CHECK: %[[D:.*]] = field.double %[[ARG0]] : [[T]]
  // CHECK: %[[R:.*]] = field.add %[[ARG0]], %[[D]] : [[T]]
  // CHECK: return %[[R]] : [[T]]
  return %0 : !QF
}

// CHECK-LABEL: @test_ef_mul_by_ef_splat_not_identity
func.func @test_ef_mul_by_ef_splat_not_identity(%arg0: !QF) -> !QF {
  // [1, 1] is NOT the multiplicative identity (which is [1, 0]).
  // DRR MulByOne must NOT fire — the mul must remain.
  // CHECK: field.mul
  %c = field.constant [1, 1] : !QF
  %0 = field.mul %arg0, %c : !QF
  return %0 : !QF
}

// CHECK-LABEL: @test_tensor_mul_by_two_is_double
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_tensor_mul_by_two_is_double(%arg0: tensor<2x!QF>) -> tensor<2x!QF> {
  // Tensor: x * 2 -> double(x) where all EF elements are [2, 0]
  %c2 = field.constant dense<[[2, 0], [2, 0]]> : tensor<2x!QF>
  %0 = field.mul %arg0, %c2 : tensor<2x!QF>
  // CHECK-NOT: field.mul
  // CHECK: %[[D:.*]] = field.double %[[ARG0]] : [[T]]
  // CHECK: return %[[D]] : [[T]]
  return %0 : tensor<2x!QF>
}

//===----------------------------------------------------------------------===//
// Mixed-type mul canonicalization (EF × PF constant)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_mixed_ef_pf_mul_by_zero
func.func @test_mixed_ef_pf_mul_by_zero(%arg0: !QF) -> !QF {
  // EF * PF(0) -> zero (lowered as ext_from_coeffs of PF zeros)
  %c0 = field.constant 0 : !PF
  %0 = field.mul %arg0, %c0 : !QF, !PF
  // CHECK-NOT: field.mul
  // CHECK: field.ext_from_coeffs
  return %0 : !QF
}

// CHECK-LABEL: @test_mixed_ef_pf_mul_by_two
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mixed_ef_pf_mul_by_two(%arg0: !QF) -> !QF {
  // EF * PF(2) -> double(EF)
  %c2 = field.constant 2 : !PF
  %0 = field.mul %arg0, %c2 : !QF, !PF
  // CHECK-NOT: field.mul
  // CHECK: %[[D:.*]] = field.double %[[ARG0]] : [[T]]
  // CHECK: return %[[D]] : [[T]]
  return %0 : !QF
}

// CHECK-LABEL: @test_mixed_ef_pf_mul_by_three
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mixed_ef_pf_mul_by_three(%arg0: !QF) -> !QF {
  // EF * PF(3) -> EF + double(EF)
  %c3 = field.constant 3 : !PF
  %0 = field.mul %arg0, %c3 : !QF, !PF
  // CHECK-NOT: field.mul
  // CHECK: %[[D:.*]] = field.double %[[ARG0]] : [[T]]
  // CHECK: %[[R:.*]] = field.add %[[ARG0]], %[[D]] : [[T]]
  // CHECK: return %[[R]] : [[T]]
  return %0 : !QF
}

// Montgomery variant of the crash reproducer.

// CHECK-LABEL: @test_mixed_ef_pf_mont_mul_by_two
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mixed_ef_pf_mont_mul_by_two(%arg0: !QFm) -> !QFm {
  %c2 = field.constant 2 : !PFm
  %0 = field.mul %arg0, %c2 : !QFm, !PFm
  // CHECK-NOT: field.mul
  // CHECK: %[[D:.*]] = field.double %[[ARG0]] : [[T]]
  // CHECK: return %[[D]] : [[T]]
  return %0 : !QFm
}

// Tower mixed-type: Fp6 × QF constant.

// CHECK-LABEL: @test_tower_mixed_mul_by_two
func.func @test_tower_mixed_mul_by_two(%arg0: !Fp6) -> !Fp6 {
  // Fp6 * QF([2,0]) should strength-reduce the per-coefficient muls.
  %c2 = field.constant [2, 0] : !QF
  %0 = field.mul %arg0, %c2 : !Fp6, !QF
  // CHECK-NOT: field.mul
  // CHECK-COUNT-3: field.double
  // CHECK-NOT: field.double
  return %0 : !Fp6
}

// CHECK-LABEL: @test_tower_mixed_mul_by_zero
func.func @test_tower_mixed_mul_by_zero(%arg0: !Fp6) -> !Fp6 {
  // Fp6 * QF([0,0]) -> zero
  %c0 = field.constant [0, 0] : !QF
  %0 = field.mul %arg0, %c0 : !Fp6, !QF
  // CHECK-NOT: field.mul
  // CHECK: field.ext_from_coeffs
  return %0 : !Fp6
}

// PF × Tower: constant PF scalar multiplying a tower EF.
// Verifies that PF constants on the LHS also canonicalize correctly.

// CHECK-LABEL: @test_mixed_pf_tower_mul_by_two
func.func @test_mixed_pf_tower_mul_by_two(%arg0: !Fp6) -> !Fp6 {
  %c2 = field.constant [2, 0] : !QF
  %0 = field.mul %c2, %arg0 : !QF, !Fp6
  // CHECK-NOT: field.mul
  // CHECK-COUNT-3: field.double
  // CHECK-NOT: field.double
  return %0 : !Fp6
}

//===----------------------------------------------------------------------===//
// BitcastOp folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_bitcast_scalar_roundtrip
// CHECK-SAME: (%[[ARG:.*]]: !pf7_i32) -> !pf7_i32 {
func.func @test_bitcast_scalar_roundtrip(%arg0: !PF) -> !PF {
  // bitcast(bitcast(x)) -> x when types match
  // CHECK-NOT: field.bitcast
  // CHECK: return %[[ARG]] : !pf7_i32
  %0 = field.bitcast %arg0 : !PF -> i32
  %1 = field.bitcast %0 : i32 -> !PF
  return %1 : !PF
}

// CHECK-LABEL: @test_bitcast_tensor_roundtrip
// CHECK-SAME: (%[[ARG:.*]]: tensor<2x[[T:.*]]>) -> tensor<2x[[T]]> {
func.func @test_bitcast_tensor_roundtrip(%arg0: tensor<2x!QF>) -> tensor<2x!QF> {
  // bitcast(bitcast(x)) -> x when types match
  // CHECK-NOT: field.bitcast
  // CHECK: return %[[ARG]] : tensor<2x[[T]]>
  %0 = field.bitcast %arg0 : tensor<2x!QF> -> tensor<4x!PF>
  %1 = field.bitcast %0 : tensor<4x!PF> -> tensor<2x!QF>
  return %1 : tensor<2x!QF>
}

// CHECK-LABEL: @test_bitcast_chain_simplify
// CHECK-SAME: (%[[ARG:.*]]: tensor<2x[[T:.*]]>) -> tensor<4x!pf7_i32> {
func.func @test_bitcast_chain_simplify(%arg0: tensor<2x!QF>) -> tensor<4x!PF> {
  // The intermediate tensor<4xi32> bitcast should be eliminated
  // CHECK: %[[RES:.*]] = field.bitcast %[[ARG]] : tensor<2x[[T]]> -> tensor<4x!pf7_i32>
  // CHECK: return %[[RES]] : tensor<4x!pf7_i32>
  %0 = field.bitcast %arg0 : tensor<2x!QF> -> tensor<4xi32>
  %1 = field.bitcast %0 : tensor<4xi32> -> tensor<4x!PF>
  return %1 : tensor<4x!PF>
}

//===----------------------------------------------------------------------===//
// ToMont/FromMont cancellation
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_from_mont_to_mont_cancel
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_from_mont_to_mont_cancel(%arg0: !QF) -> !QF {
  // CHECK-NOT: field.to_mont
  // CHECK-NOT: field.from_mont
  // CHECK: return %[[ARG0]] : [[T]]
  %0 = field.to_mont %arg0 : !QFm
  %1 = field.from_mont %0 : !QF
  return %1 : !QF
}

// CHECK-LABEL: @test_from_mont_to_mont_tensor_cancel
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_from_mont_to_mont_tensor_cancel(%arg0: tensor<2x!QF>) -> tensor<2x!QF> {
  // CHECK-NOT: field.to_mont
  // CHECK-NOT: field.from_mont
  // CHECK: return %[[ARG0]] : [[T]]
  %0 = field.to_mont %arg0 : tensor<2x!QFm>
  %1 = field.from_mont %0 : tensor<2x!QF>
  return %1 : tensor<2x!QF>
}

// CHECK-LABEL: @test_to_mont_from_mont_cancel
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_to_mont_from_mont_cancel(%arg0: !QFm) -> !QFm {
  // CHECK-NOT: field.from_mont
  // CHECK-NOT: field.to_mont
  // CHECK: return %[[ARG0]] : [[T]]
  %0 = field.from_mont %arg0 : !QF
  %1 = field.to_mont %0 : !QFm
  return %1 : !QFm
}

// CHECK-LABEL: @test_to_mont_from_mont_tensor_cancel
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_to_mont_from_mont_tensor_cancel(%arg0: tensor<2x!QFm>) -> tensor<2x!QFm> {
  // CHECK-NOT: field.from_mont
  // CHECK-NOT: field.to_mont
  // CHECK: return %[[ARG0]] : [[T]]
  %0 = field.from_mont %arg0 : tensor<2x!QF>
  %1 = field.to_mont %0 : tensor<2x!QFm>
  return %1 : tensor<2x!QFm>
}

//===----------------------------------------------------------------------===//
// Tower Extension Field (Fp4 = (Fp2)^2) constant folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_tower_fold_negate
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tower_fold_negate() -> !Fp6 {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[}}[6, 5], [4, 3], [2, 1]{{\]}}> : [[T]]
  // -[1, 2, 3, 4, 5, 6] mod 7 = [6, 5, 4, 3, 2, 1]
  %0 = field.constant [1, 2, 3, 4, 5, 6] : !Fp6
  %1 = field.negate %0 : !Fp6
  // CHECK-NOT: field.negate
  // CHECK: return %[[C]] : [[T]]
  return %1 : !Fp6
}

// CHECK-LABEL: @test_tower_fold_double
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tower_fold_double() -> !Fp6 {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[}}[2, 4], [6, 1], [3, 5]{{\]}}> : [[T]]
  // 2 * [1, 2, 3, 4, 5, 6] mod 7 = [2, 4, 6, 8, 10, 12] mod 7 = [2, 4, 6, 1, 3, 5]
  %0 = field.constant [1, 2, 3, 4, 5, 6] : !Fp6
  %1 = field.double %0 : !Fp6
  // CHECK-NOT: field.double
  // CHECK: return %[[C]] : [[T]]
  return %1 : !Fp6
}

// CHECK-LABEL: @test_tower_fold_add
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tower_fold_add() -> !Fp6 {
  // CHECK: %[[C:.*]] = field.constant dense<6> : [[T]]
  // [1, 2, 3, 4, 5, 6] + [5, 4, 3, 2, 1, 0] mod 7 = [6, 6, 6, 6, 6, 6] (splat, shape-independent)
  %0 = field.constant [1, 2, 3, 4, 5, 6] : !Fp6
  %1 = field.constant [5, 4, 3, 2, 1, 0] : !Fp6
  %2 = field.add %0, %1 : !Fp6
  // CHECK-NOT: field.add
  // CHECK: return %[[C]] : [[T]]
  return %2 : !Fp6
}

// CHECK-LABEL: @test_tower_fold_sub
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tower_fold_sub() -> !Fp6 {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[}}[3, 5], [0, 2], [4, 6]{{\]}}> : [[T]]
  // [1, 2, 3, 4, 5, 6] - [5, 4, 3, 2, 1, 0] mod 7 = [-4, -2, 0, 2, 4, 6] mod 7 = [3, 5, 0, 2, 4, 6]
  %0 = field.constant [1, 2, 3, 4, 5, 6] : !Fp6
  %1 = field.constant [5, 4, 3, 2, 1, 0] : !Fp6
  %2 = field.sub %0, %1 : !Fp6
  // CHECK-NOT: field.sub
  // CHECK: return %[[C]] : [[T]]
  return %2 : !Fp6
}

// CHECK-LABEL: @test_tower_sub_self_is_zero
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_tower_sub_self_is_zero(%arg0: !Fp6) -> !Fp6 {
  // CHECK: %[[C:.*]] = field.constant dense<0> : [[T]]
  %0 = field.sub %arg0, %arg0 : !Fp6
  // CHECK-NOT: field.sub
  // CHECK: return %[[C]] : [[T]]
  return %0 : !Fp6
}

// CHECK-LABEL: @test_tower_tensor_sub_self_is_zero
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_tower_tensor_sub_self_is_zero(%arg0: tensor<4x!Fp6>) -> tensor<4x!Fp6> {
  // CHECK: %[[C:.*]] = field.constant dense<0> : [[T]]
  %0 = field.sub %arg0, %arg0 : tensor<4x!Fp6>
  // CHECK-NOT: field.sub
  // CHECK: return %[[C]] : [[T]]
  return %0 : tensor<4x!Fp6>
}

// CHECK-LABEL: @test_tower_fold_mul
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tower_fold_mul() -> !Fp6 {
  // Fp6 multiplication: (a + bw + cw²)(d + ew + fw²)
  // where a, b, c, d, e, f ∈ Fp2 and w³ = 2 (non-residue for Fp6/Fp2)
  %0 = field.constant [1, 2, 3, 4, 5, 6] : !Fp6
  %1 = field.constant [5, 4, 3, 2, 1, 0] : !Fp6
  %2 = field.mul %0, %1 : !Fp6
  // CHECK-NOT: field.mul
  // CHECK: return %{{.*}} : [[T]]
  return %2 : !Fp6
}

//===----------------------------------------------------------------------===//
// Tensor of Tower Extension Field (tensor<2x!Fp6>) constant folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_tensor_tower_fold_negate
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_tower_fold_negate() -> tensor<2x!Fp6> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[}}{{\[}}[6, 5], [4, 3], [2, 1]{{\]}}, {{\[}}[1, 2], [3, 4], [5, 6]{{\]}}{{\]}}> : [[T]]
  // -[[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]] mod 7 = [[6, 5, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6]]
  %0 = field.constant dense<[[[1, 2], [3, 4], [5, 6]], [[6, 5], [4, 3], [2, 1]]]> : tensor<2x!Fp6>
  %1 = field.negate %0 : tensor<2x!Fp6>
  // CHECK-NOT: field.negate
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!Fp6>
}

// CHECK-LABEL: @test_tensor_tower_fold_double
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_tower_fold_double() -> tensor<2x!Fp6> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[}}{{\[}}[2, 4], [6, 1], [3, 5]{{\]}}, {{\[}}[5, 3], [1, 6], [4, 2]{{\]}}{{\]}}> : [[T]]
  // 2 * [[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]] mod 7 = [[2, 4, 6, 1, 3, 5], [5, 3, 1, 6, 4, 2]]
  %0 = field.constant dense<[[[1, 2], [3, 4], [5, 6]], [[6, 5], [4, 3], [2, 1]]]> : tensor<2x!Fp6>
  %1 = field.double %0 : tensor<2x!Fp6>
  // CHECK-NOT: field.double
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!Fp6>
}

// CHECK-LABEL: @test_tensor_tower_fold_add
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_tower_fold_add() -> tensor<2x!Fp6> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[}}{{\[}}[6, 6], [6, 6], [6, 6]{{\]}}, {{\[}}[0, 0], [0, 0], [0, 0]{{\]}}{{\]}}> : [[T]]
  // [[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]] + [[5, 4, 3, 2, 1, 0], [1, 2, 3, 4, 5, 6]] mod 7
  // = [[6, 6, 6, 6, 6, 6], [0, 0, 0, 0, 0, 0]]
  %0 = field.constant dense<[[[1, 2], [3, 4], [5, 6]], [[6, 5], [4, 3], [2, 1]]]> : tensor<2x!Fp6>
  %1 = field.constant dense<[[[5, 4], [3, 2], [1, 0]], [[1, 2], [3, 4], [5, 6]]]> : tensor<2x!Fp6>
  %2 = field.add %0, %1 : tensor<2x!Fp6>
  // CHECK-NOT: field.add
  // CHECK: return %[[C]] : [[T]]
  return %2 : tensor<2x!Fp6>
}

// CHECK-LABEL: @test_tensor_tower_fold_sub
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_tower_fold_sub() -> tensor<2x!Fp6> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[}}{{\[}}[3, 5], [0, 2], [4, 6]{{\]}}, {{\[}}[5, 3], [1, 6], [4, 2]{{\]}}{{\]}}> : [[T]]
  // [[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]] - [[5, 4, 3, 2, 1, 0], [1, 2, 3, 4, 5, 6]] mod 7
  // = [[-4, -2, 0, 2, 4, 6], [5, 3, 1, -1, -3, -5]] mod 7 = [[3, 5, 0, 2, 4, 6], [5, 3, 1, 6, 4, 2]]
  %0 = field.constant dense<[[[1, 2], [3, 4], [5, 6]], [[6, 5], [4, 3], [2, 1]]]> : tensor<2x!Fp6>
  %1 = field.constant dense<[[[5, 4], [3, 2], [1, 0]], [[1, 2], [3, 4], [5, 6]]]> : tensor<2x!Fp6>
  %2 = field.sub %0, %1 : tensor<2x!Fp6>
  // CHECK-NOT: field.sub
  // CHECK: return %[[C]] : [[T]]
  return %2 : tensor<2x!Fp6>
}

// CHECK-LABEL: @test_tensor_tower_fold_mul
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_tower_fold_mul() -> tensor<2x!Fp6> {
  // Fp6 multiplication: (a + bw + cw²)(d + ew + fw²)
  // where a, b, c, d, e, f ∈ Fp2 and w³ = 2 (non-residue for Fp6/Fp2)
  %0 = field.constant dense<[[[1, 2], [3, 4], [5, 6]], [[6, 5], [4, 3], [2, 1]]]> : tensor<2x!Fp6>
  %1 = field.constant dense<[[[5, 4], [3, 2], [1, 0]], [[1, 2], [3, 4], [5, 6]]]> : tensor<2x!Fp6>
  %2 = field.mul %0, %1 : tensor<2x!Fp6>
  // CHECK-NOT: field.mul
  // CHECK: return %{{.*}} : [[T]]
  return %2 : tensor<2x!Fp6>
}

//===----------------------------------------------------------------------===//
// Tensor of Tower Extension Field splat constant folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_tower_reshape_splat_constant_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tower_reshape_splat_constant_fold() -> tensor<2x2x!Fp6> {
  %0 = field.constant dense<1> : tensor<4x!Fp6>
  %shape = arith.constant dense<[2, 2]> : tensor<2xi32>
  %1 = tensor.reshape %0(%shape) : (tensor<4x!Fp6>, tensor<2xi32>) -> tensor<2x2x!Fp6>
  // CHECK: %[[C:.*]] = field.constant dense<1> : [[T]]
  // CHECK-NOT: tensor.reshape
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x2x!Fp6>
}

// CHECK-LABEL: @test_tower_extract_slice_splat_constant_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tower_extract_slice_splat_constant_fold() -> tensor<2x!Fp6> {
  %0 = field.constant dense<2> : tensor<4x!Fp6>
  %1 = tensor.extract_slice %0[1] [2] [1] : tensor<4x!Fp6> to tensor<2x!Fp6>
  // CHECK: %[[C:.*]] = field.constant dense<2> : [[T]]
  // CHECK-NOT: tensor.extract_slice
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!Fp6>
}

// CHECK-LABEL: @test_tower_gather_splat_constant_fold
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tower_gather_splat_constant_fold() -> tensor<2x1x!Fp6> {
  %0 = field.constant dense<3> : tensor<4x!Fp6>
  %indices = arith.constant dense<[[0], [2]]> : tensor<2x1xindex>
  %1 = tensor.gather %0[%indices] gather_dims([0]) : (tensor<4x!Fp6>, tensor<2x1xindex>) -> tensor<2x1x!Fp6>
  // CHECK: %[[C:.*]] = field.constant dense<3> : [[T]]
  // CHECK-NOT: tensor.gather
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x1x!Fp6>
}

//===----------------------------------------------------------------------===//
// ExtToCoeffsOp constant folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_ext_to_coeffs_fold_simple
func.func @test_ext_to_coeffs_fold_simple() -> (!PF, !PF) {
  // ext_to_coeffs of a constant should fold to individual PF constants.
  // CHECK-DAG: %[[C3:.*]] = field.constant 3
  // CHECK-DAG: %[[C5:.*]] = field.constant 5
  // CHECK-NOT: field.ext_to_coeffs
  // CHECK: return %[[C3]], %[[C5]]
  %c = field.constant [3, 5] : !QF
  %c0, %c1 = field.ext_to_coeffs %c : (!QF) -> (!PF, !PF)
  return %c0, %c1 : !PF, !PF
}

// CHECK-LABEL: @test_ext_to_coeffs_fold_tower
func.func @test_ext_to_coeffs_fold_tower() -> (!QF, !QF, !QF) {
  // ext_to_coeffs of a tower constant should fold to individual EF constants.
  // CHECK-DAG: %[[C0:.*]] = field.constant dense<[1, 2]>
  // CHECK-DAG: %[[C1:.*]] = field.constant dense<[3, 4]>
  // CHECK-DAG: %[[C2:.*]] = field.constant dense<[5, 6]>
  // CHECK-NOT: field.ext_to_coeffs
  // CHECK: return %[[C0]], %[[C1]], %[[C2]]
  %c = field.constant [1, 2, 3, 4, 5, 6] : !Fp6
  %c0, %c1, %c2 = field.ext_to_coeffs %c : (!Fp6) -> (!QF, !QF, !QF)
  return %c0, %c1, %c2 : !QF, !QF, !QF
}

// CHECK-LABEL: @test_ext_to_coeffs_nonfold
func.func @test_ext_to_coeffs_nonfold(%arg: !QF) -> (!PF, !PF) {
  // Non-constant input should NOT fold.
  // CHECK: field.ext_to_coeffs
  %c0, %c1 = field.ext_to_coeffs %arg : (!QF) -> (!PF, !PF)
  return %c0, %c1 : !PF, !PF
}
