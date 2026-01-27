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
!QF = !field.ef<2x!PF, 6:i32>

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
// Tower Extension Field (Fp4 = (Fp2)^2) constant folding
//===----------------------------------------------------------------------===//

!Fp4 = !field.ef<2x!QF, 6:i32>

// CHECK-LABEL: @test_tower_fold_negate
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tower_fold_negate() -> !Fp4 {
  // CHECK: %[[C:.*]] = field.constant dense<[6, 5, 4, 3]> : [[T]]
  // -[1, 2, 3, 4] mod 7 = [6, 5, 4, 3]
  %0 = field.constant [1, 2, 3, 4] : !Fp4
  %1 = field.negate %0 : !Fp4
  // CHECK-NOT: field.negate
  // CHECK: return %[[C]] : [[T]]
  return %1 : !Fp4
}

// CHECK-LABEL: @test_tower_fold_double
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tower_fold_double() -> !Fp4 {
  // CHECK: %[[C:.*]] = field.constant dense<[2, 4, 6, 1]> : [[T]]
  // 2 * [1, 2, 3, 4] mod 7 = [2, 4, 6, 8 mod 7] = [2, 4, 6, 1]
  %0 = field.constant [1, 2, 3, 4] : !Fp4
  %1 = field.double %0 : !Fp4
  // CHECK-NOT: field.double
  // CHECK: return %[[C]] : [[T]]
  return %1 : !Fp4
}

// CHECK-LABEL: @test_tower_fold_add
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tower_fold_add() -> !Fp4 {
  // CHECK: %[[C:.*]] = field.constant dense<6> : [[T]]
  // [1, 2, 3, 4] + [5, 4, 3, 2] = [6, 6, 6, 6] (splat)
  %0 = field.constant [1, 2, 3, 4] : !Fp4
  %1 = field.constant [5, 4, 3, 2] : !Fp4
  %2 = field.add %0, %1 : !Fp4
  // CHECK-NOT: field.add
  // CHECK: return %[[C]] : [[T]]
  return %2 : !Fp4
}

// CHECK-LABEL: @test_tower_fold_sub
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tower_fold_sub() -> !Fp4 {
  // CHECK: %[[C:.*]] = field.constant dense<[3, 5, 0, 2]> : [[T]]
  // [1, 2, 3, 4] - [5, 4, 3, 2] = [-4, -2, 0, 2] mod 7 = [3, 5, 0, 2]
  %0 = field.constant [1, 2, 3, 4] : !Fp4
  %1 = field.constant [5, 4, 3, 2] : !Fp4
  %2 = field.sub %0, %1 : !Fp4
  // CHECK-NOT: field.sub
  // CHECK: return %[[C]] : [[T]]
  return %2 : !Fp4
}

// CHECK-LABEL: @test_tower_fold_mul
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tower_fold_mul() -> !Fp4 {
  // Fp4 multiplication: (a + bv)(c + dv) = (ac + ξbd) + (ad + bc)v
  // where a, b, c, d ∈ Fp2 and ξ = 6 (non-residue for Fp4/Fp2)
  // Let a = [1, 2], b = [3, 4], c = [5, 4], d = [3, 2]
  // Using Fp2 arithmetic with ξ₂ = 6 (non-residue for Fp2/Fp)
  %0 = field.constant [1, 2, 3, 4] : !Fp4
  %1 = field.constant [5, 4, 3, 2] : !Fp4
  %2 = field.mul %0, %1 : !Fp4
  // CHECK-NOT: field.mul
  // CHECK: return %{{.*}} : [[T]]
  return %2 : !Fp4
}

//===----------------------------------------------------------------------===//
// Tensor of Tower Extension Field (tensor<2x!Fp4>) constant folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_tensor_tower_fold_negate
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_tower_fold_negate() -> tensor<2x!Fp4> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[}}[6, 5, 4, 3], [3, 4, 5, 6]{{\]}}> : [[T]]
  // -[[1, 2, 3, 4], [4, 3, 2, 1]] mod 7 = [[6, 5, 4, 3], [3, 4, 5, 6]]
  %0 = field.constant dense<[[1, 2, 3, 4], [4, 3, 2, 1]]> : tensor<2x!Fp4>
  %1 = field.negate %0 : tensor<2x!Fp4>
  // CHECK-NOT: field.negate
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!Fp4>
}

// CHECK-LABEL: @test_tensor_tower_fold_double
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_tower_fold_double() -> tensor<2x!Fp4> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[}}[2, 4, 6, 1], [1, 6, 4, 2]{{\]}}> : [[T]]
  // 2 * [[1, 2, 3, 4], [4, 3, 2, 1]] mod 7 = [[2, 4, 6, 1], [1, 6, 4, 2]]
  %0 = field.constant dense<[[1, 2, 3, 4], [4, 3, 2, 1]]> : tensor<2x!Fp4>
  %1 = field.double %0 : tensor<2x!Fp4>
  // CHECK-NOT: field.double
  // CHECK: return %[[C]] : [[T]]
  return %1 : tensor<2x!Fp4>
}

// CHECK-LABEL: @test_tensor_tower_fold_add
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_tower_fold_add() -> tensor<2x!Fp4> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[}}[6, 6, 6, 6], [5, 5, 5, 5]{{\]}}> : [[T]]
  // [[1, 2, 3, 4], [4, 3, 2, 1]] + [[5, 4, 3, 2], [1, 2, 3, 4]] = [[6, 6, 6, 6], [5, 5, 5, 5]]
  %0 = field.constant dense<[[1, 2, 3, 4], [4, 3, 2, 1]]> : tensor<2x!Fp4>
  %1 = field.constant dense<[[5, 4, 3, 2], [1, 2, 3, 4]]> : tensor<2x!Fp4>
  %2 = field.add %0, %1 : tensor<2x!Fp4>
  // CHECK-NOT: field.add
  // CHECK: return %[[C]] : [[T]]
  return %2 : tensor<2x!Fp4>
}

// CHECK-LABEL: @test_tensor_tower_fold_sub
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_tower_fold_sub() -> tensor<2x!Fp4> {
  // CHECK: %[[C:.*]] = field.constant dense<{{\[}}[3, 5, 0, 2], [3, 1, 6, 4]{{\]}}> : [[T]]
  // [[1, 2, 3, 4], [4, 3, 2, 1]] - [[5, 4, 3, 2], [1, 2, 3, 4]] mod 7
  // = [[-4, -2, 0, 2], [3, 1, -1, -3]] mod 7 = [[3, 5, 0, 2], [3, 1, 6, 4]]
  %0 = field.constant dense<[[1, 2, 3, 4], [4, 3, 2, 1]]> : tensor<2x!Fp4>
  %1 = field.constant dense<[[5, 4, 3, 2], [1, 2, 3, 4]]> : tensor<2x!Fp4>
  %2 = field.sub %0, %1 : tensor<2x!Fp4>
  // CHECK-NOT: field.sub
  // CHECK: return %[[C]] : [[T]]
  return %2 : tensor<2x!Fp4>
}

// CHECK-LABEL: @test_tensor_tower_fold_mul
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_tensor_tower_fold_mul() -> tensor<2x!Fp4> {
  // Fp4 multiplication: (a + bv)(c + dv) = (ac + ξbd) + (ad + bc)v
  // where a, b, c, d ∈ Fp2 and ξ = 6 (non-residue for Fp4/Fp2)
  %0 = field.constant dense<[[1, 2, 3, 4], [4, 3, 2, 1]]> : tensor<2x!Fp4>
  %1 = field.constant dense<[[5, 4, 3, 2], [1, 2, 3, 4]]> : tensor<2x!Fp4>
  %2 = field.mul %0, %1 : tensor<2x!Fp4>
  // CHECK-NOT: field.mul
  // CHECK: return %{{.*}} : [[T]]
  return %2 : tensor<2x!Fp4>
}
