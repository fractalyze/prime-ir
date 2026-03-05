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
  // CHECK: %[[C:.*]] = field.constant dense<[4, 2]> : [[T]]
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
  // CHECK: %[[C:.*]] = field.constant dense<3> : [[T]]
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
  // CHECK: %[[C:.*]] = field.constant dense<[1, 5]> : [[T]]
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
  // CHECK: %[[C:.*]] = field.constant dense<[4, 2]> : [[T]]
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
  // CHECK: %[[C:.*]] = field.constant dense<[2, 5]> : [[T]]
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
  // CHECK: %[[C:.*]] = field.constant dense<[1, 5]> : [[T]]
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
  // CHECK: %[[C:.*]] = field.constant dense<[6, 2, 3]> : [[T]]
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
  // CHECK: %[[C:.*]] = field.constant dense<[2, 4, 6]> : [[T]]
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
