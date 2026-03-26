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

// RUN: prime-ir-opt -canonicalize %s -split-input-file | FileCheck %s -enable-var-scope

!PF = !field.pf<7:i32>
!QF = !field.ef<2x!PF, 6:i32>

//===----------------------------------------------------------------------===//
// Mixed-type constant folding: ext op base
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_fold_add_ext_base
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_fold_add_ext_base() -> !QF {
  // [1, 2] + 3 = [1 + 3, 2] = [4, 2] (mod 7)
  // CHECK: %[[C:.*]] = field.constant [4, 2] : [[T]]
  // CHECK: return %[[C]]
  %0 = field.constant [1, 2] : !QF
  %1 = field.constant 3 : !PF
  %2 = field.add %0, %1 : !QF, !PF
  return %2 : !QF
}

// CHECK-LABEL: @test_fold_sub_ext_base
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_fold_sub_ext_base() -> !QF {
  // [5, 3] - 2 = [3, 3] (mod 7)
  // CHECK: %[[C:.*]] = field.constant [3, 3] : [[T]]
  // CHECK: return %[[C]]
  %0 = field.constant [5, 3] : !QF
  %1 = field.constant 2 : !PF
  %2 = field.sub %0, %1 : !QF, !PF
  return %2 : !QF
}

// CHECK-LABEL: @test_fold_mul_ext_base
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_fold_mul_ext_base() -> !QF {
  // [2, 3] * 4 = [2 * 4, 3 * 4] = [8, 12] = [1, 5] (mod 7)
  // CHECK: %[[C:.*]] = field.constant [1, 5] : [[T]]
  // CHECK: return %[[C]]
  %0 = field.constant [2, 3] : !QF
  %1 = field.constant 4 : !PF
  %2 = field.mul %0, %1 : !QF, !PF
  return %2 : !QF
}

// -----

!PF = !field.pf<7:i32>
!QF = !field.ef<2x!PF, 6:i32>

//===----------------------------------------------------------------------===//
// Mixed-type constant folding: base op ext (commuted)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_fold_add_base_ext
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_fold_add_base_ext() -> !QF {
  // 3 + [1, 2] = [4, 2] (mod 7)
  // CHECK: %[[C:.*]] = field.constant [4, 2] : [[T]]
  // CHECK: return %[[C]]
  %0 = field.constant 3 : !PF
  %1 = field.constant [1, 2] : !QF
  %2 = field.add %0, %1 : !PF, !QF
  return %2 : !QF
}

// CHECK-LABEL: @test_fold_sub_base_ext
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_fold_sub_base_ext() -> !QF {
  // 3 - [1, 2] = [3 - 1, -2] = [2, 5] (mod 7)
  // CHECK: %[[C:.*]] = field.constant [2, 5] : [[T]]
  // CHECK: return %[[C]]
  %0 = field.constant 3 : !PF
  %1 = field.constant [1, 2] : !QF
  %2 = field.sub %0, %1 : !PF, !QF
  return %2 : !QF
}

// CHECK-LABEL: @test_fold_mul_base_ext
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_fold_mul_base_ext() -> !QF {
  // 4 * [2, 3] = [8, 12] = [1, 5] (mod 7)
  // CHECK: %[[C:.*]] = field.constant [1, 5] : [[T]]
  // CHECK: return %[[C]]
  %0 = field.constant 4 : !PF
  %1 = field.constant [2, 3] : !QF
  %2 = field.mul %0, %1 : !PF, !QF
  return %2 : !QF
}

// -----

!PF = !field.pf<7:i32>
!QF = !field.ef<2x!PF, 6:i32>

//===----------------------------------------------------------------------===//
// Canonicalization expansion (non-constant mixed-type ops)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_expand_add_ext_base
func.func @test_expand_add_ext_base(%ext: !QF, %base: !PF) -> !QF {
  // Should expand to: coeffs = ext_to_coeffs(%ext); coeffs[0] += %base
  // CHECK-NOT: field.add {{.*}} : !field.ef<{{.*}}>, !field.pf<{{.*}}>
  %0 = field.add %ext, %base : !QF, !PF
  return %0 : !QF
}

// CHECK-LABEL: @test_expand_add_base_ext
func.func @test_expand_add_base_ext(%base: !PF, %ext: !QF) -> !QF {
  // base + ext → same as ext + base (commutative)
  // CHECK-NOT: field.add {{.*}} : !field.pf<{{.*}}>, !field.ef<{{.*}}>
  %0 = field.add %base, %ext : !PF, !QF
  return %0 : !QF
}

// CHECK-LABEL: @test_expand_sub_ext_base
func.func @test_expand_sub_ext_base(%ext: !QF, %base: !PF) -> !QF {
  // Should expand to: coeffs = ext_to_coeffs(%ext); coeffs[0] -= %base
  // CHECK-NOT: field.sub {{.*}} : !field.ef<{{.*}}>, !field.pf<{{.*}}>
  %0 = field.sub %ext, %base : !QF, !PF
  return %0 : !QF
}

// CHECK-LABEL: @test_expand_sub_base_ext
func.func @test_expand_sub_base_ext(%base: !PF, %ext: !QF) -> !QF {
  // Should expand to: coeffs = ext_to_coeffs(%ext);
  //                   coeffs[0] = base - coeffs[0]; negate others
  // CHECK-NOT: field.sub {{.*}} : !field.pf<{{.*}}>, !field.ef<{{.*}}>
  %0 = field.sub %base, %ext : !PF, !QF
  return %0 : !QF
}

// CHECK-LABEL: @test_expand_mul_ext_base
func.func @test_expand_mul_ext_base(%ext: !QF, %base: !PF) -> !QF {
  // Should expand to: coeffs = ext_to_coeffs(%ext);
  //                   for each i: coeffs[i] *= %base
  // CHECK-NOT: field.mul {{.*}} : !field.ef<{{.*}}>, !field.pf<{{.*}}>
  %0 = field.mul %ext, %base : !QF, !PF
  return %0 : !QF
}

// CHECK-LABEL: @test_expand_mul_base_ext
func.func @test_expand_mul_base_ext(%base: !PF, %ext: !QF) -> !QF {
  // base * ext → same as ext * base (commutative)
  // CHECK-NOT: field.mul {{.*}} : !field.pf<{{.*}}>, !field.ef<{{.*}}>
  %0 = field.mul %base, %ext : !PF, !QF
  return %0 : !QF
}

// Splat EF constant [2, 2] must not match MulByTwo DRR directly.
// CHECK-LABEL: @test_mixed_mul_splat_ef_const
func.func @test_mixed_mul_splat_ef_const(%x: !PF) -> !QF {
  // CHECK: field.ext_from_coeffs
  %c = field.constant [2, 2] : !QF
  %0 = field.mul %x, %c : !PF, !QF
  return %0 : !QF
}

// -----

!PF = !field.pf<7:i32>
!CF = !field.ef<3x!PF, 5:i32>

//===----------------------------------------------------------------------===//
// Cubic extension field mixed-type ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_fold_add_cubic_ext_base
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_fold_add_cubic_ext_base() -> !CF {
  // [1, 2, 3] + 5 = [6, 2, 3] (mod 7)
  // CHECK: %[[C:.*]] = field.constant [6, 2, 3] : [[T]]
  // CHECK: return %[[C]]
  %0 = field.constant [1, 2, 3] : !CF
  %1 = field.constant 5 : !PF
  %2 = field.add %0, %1 : !CF, !PF
  return %2 : !CF
}

// CHECK-LABEL: @test_fold_mul_cubic_ext_base
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_fold_mul_cubic_ext_base() -> !CF {
  // [1, 2, 3] * 2 = [2, 4, 6] (mod 7)
  // CHECK: %[[C:.*]] = field.constant [2, 4, 6] : [[T]]
  // CHECK: return %[[C]]
  %0 = field.constant [1, 2, 3] : !CF
  %1 = field.constant 2 : !PF
  %2 = field.mul %0, %1 : !CF, !PF
  return %2 : !CF
}

// -----

!PF = !field.pf<7:i32>
!QF = !field.ef<2x!PF, 6:i32>

//===----------------------------------------------------------------------===//
// Mixed-type strength reduction via expansion + ExtToCoeffsOp fold
//===----------------------------------------------------------------------===//
// When a mixed-type op has an EF constant, the pipeline:
// 1. ExpandMixed{Mul,Additive}Op decomposes into ext_to_coeffs + same-type ops
// 2. ExtToCoeffsOp::fold constant-folds the EF constant into PF coefficients
// 3. DRR strength reduction patterns apply to the resulting same-type ops

// CHECK-LABEL: @test_mixed_mul_by_ef_two
func.func @test_mixed_mul_by_ef_two(%pf: !PF) -> !QF {
  // pf * EF(2, 0) -> ext_from_coeffs(double(pf), 0)
  // CHECK-DAG: %[[Z:.*]] = field.constant 0
  // CHECK-DAG: %[[D:.*]] = field.double
  // CHECK: field.ext_from_coeffs
  %c = field.constant [2, 0] : !QF
  %0 = field.mul %pf, %c : !PF, !QF
  return %0 : !QF
}

// CHECK-LABEL: @test_mixed_mul_by_ef_three
func.func @test_mixed_mul_by_ef_three(%pf: !PF) -> !QF {
  // pf * EF(3, 0) -> ext_from_coeffs(pf + double(pf), 0)
  // CHECK-DAG: %[[Z:.*]] = field.constant 0
  // CHECK-DAG: %[[D:.*]] = field.double
  // CHECK: field.add
  // CHECK: field.ext_from_coeffs
  %c = field.constant [3, 0] : !QF
  %0 = field.mul %pf, %c : !PF, !QF
  return %0 : !QF
}

// CHECK-LABEL: @test_mixed_mul_by_ef_neg_one
func.func @test_mixed_mul_by_ef_neg_one(%pf: !PF) -> !QF {
  // pf * EF(6, 0) mod 7 = pf * (-1) -> ext_from_coeffs(double(pf + double(pf)), 0)
  // (MulBySixRhs: x * 6 -> double(x + double(x)))
  // CHECK-DAG: %[[Z:.*]] = field.constant 0
  // CHECK: field.double
  // CHECK: field.ext_from_coeffs
  %c = field.constant [6, 0] : !QF
  %0 = field.mul %pf, %c : !PF, !QF
  return %0 : !QF
}

// CHECK-LABEL: @test_mixed_add_pf_ef_const
func.func @test_mixed_add_pf_ef_const(%pf: !PF) -> !QF {
  // pf + EF(3, 5) -> ext_from_coeffs(pf + 3, 5)
  // CHECK-DAG: %[[C3:.*]] = field.constant 3
  // CHECK-DAG: %[[C5:.*]] = field.constant 5
  // CHECK: field.add
  // CHECK: field.ext_from_coeffs
  %c = field.constant [3, 5] : !QF
  %0 = field.add %pf, %c : !PF, !QF
  return %0 : !QF
}

// CHECK-LABEL: @test_mixed_sub_pf_ef_const
func.func @test_mixed_sub_pf_ef_const(%pf: !PF) -> !QF {
  // pf - EF(1, 2) -> ext_from_coeffs(pf - 1, negate(2))
  // negate(2) mod 7 = 5, constant-folded
  // CHECK-DAG: %[[C5:.*]] = field.constant 5
  // CHECK-DAG: %[[C1:.*]] = field.constant 1
  // CHECK: field.sub
  // CHECK: field.ext_from_coeffs
  %c = field.constant [1, 2] : !QF
  %0 = field.sub %pf, %c : !PF, !QF
  return %0 : !QF
}

// -----

!PF = !field.pf<7:i32>
!QF = !field.ef<2x!PF, 6:i32>

//===----------------------------------------------------------------------===//
// Tensor mixed-type ops (elementwise)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_tensor_add_ext_base
func.func @test_tensor_add_ext_base(
    %ext: tensor<4x!QF>, %base: tensor<4x!PF>) -> tensor<4x!QF> {
  // Tensor mixed-type ops are valid (elementwise mappable).
  // They are not expanded during canonicalization; lowered to scalars first.
  // CHECK: field.add
  // CHECK-SAME: tensor<4x
  %0 = field.add %ext, %base : tensor<4x!QF>, tensor<4x!PF>
  return %0 : tensor<4x!QF>
}

// CHECK-LABEL: @test_tensor_mul_ext_base
func.func @test_tensor_mul_ext_base(
    %ext: tensor<4x!QF>, %base: tensor<4x!PF>) -> tensor<4x!QF> {
  // CHECK: field.mul
  // CHECK-SAME: tensor<4x
  %0 = field.mul %ext, %base : tensor<4x!QF>, tensor<4x!PF>
  return %0 : tensor<4x!QF>
}

// -----

!PF = !field.pf<7:i32>
!QF = !field.ef<2x!PF, 6:i32>

//===----------------------------------------------------------------------===//
// Mixed-type mul DRR canonicalization (EF × PF constant)
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

// -----

!PFm = !field.pf<7:i32, true>
!QFm = !field.ef<2x!PFm, 6:i32>

//===----------------------------------------------------------------------===//
// Montgomery variant
//===----------------------------------------------------------------------===//

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

// -----

!PF = !field.pf<7:i32>
!QF = !field.ef<2x!PF, 6:i32>

// Tower extension: Fp6 = (Fp2)³ where Fp6 = Fp2[w]/(w³ - 2)
!Fp6 = !field.ef<3x!QF, 2:i32>

//===----------------------------------------------------------------------===//
// Tower mixed-type: Fp6 × QF constant
//===----------------------------------------------------------------------===//

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

// -----

!PF = !field.pf<7:i32>
!QF = !field.ef<2x!PF, 6:i32>

//===----------------------------------------------------------------------===//
// Mixed-type tensor constant folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_mixed_tensor_add_pf_ef
// CHECK:         field.constant dense<[3, 0]>
// CHECK-NOT:     field.add
func.func @test_mixed_tensor_add_pf_ef() -> tensor<!QF> {
  %pf = field.constant dense<3> : tensor<!PF>
  %ef = field.constant dense<[0, 0]> : tensor<!QF>
  %r = field.add %pf, %ef : tensor<!PF>, tensor<!QF>
  return %r : tensor<!QF>
}

// CHECK-LABEL: @test_mixed_tensor_sub_ef_pf
// CHECK:         field.constant dense<[4, 2]>
// CHECK-NOT:     field.sub
func.func @test_mixed_tensor_sub_ef_pf() -> tensor<!QF> {
  %ef = field.constant dense<[5, 2]> : tensor<!QF>
  %pf = field.constant dense<1> : tensor<!PF>
  %r = field.sub %ef, %pf : tensor<!QF>, tensor<!PF>
  return %r : tensor<!QF>
}

// CHECK-LABEL: @test_mixed_tensor_mul_pf_ef
// CHECK:         field.constant dense<[6, 3]>
// CHECK-NOT:     field.mul
func.func @test_mixed_tensor_mul_pf_ef() -> tensor<!QF> {
  %pf = field.constant dense<3> : tensor<!PF>
  %ef = field.constant dense<[2, 1]> : tensor<!QF>
  %r = field.mul %pf, %ef : tensor<!PF>, tensor<!QF>
  return %r : tensor<!QF>
}

// Ranked tensor: PF(2) * [1,0] = [2,0] and PF(3) * [0,1] = [0,3].
// CHECK-LABEL: @test_mixed_tensor_2d_mul_pf_ef
// CHECK:         field.constant dense<[2, 0, 0, 3]>
// CHECK-NOT:     field.mul
func.func @test_mixed_tensor_2d_mul_pf_ef() -> tensor<2x!QF> {
  %pf = field.constant dense<[2, 3]> : tensor<2x!PF>
  %ef = field.constant dense<[[1, 0], [0, 1]]> : tensor<2x!QF>
  %r = field.mul %pf, %ef : tensor<2x!PF>, tensor<2x!QF>
  return %r : tensor<2x!QF>
}
