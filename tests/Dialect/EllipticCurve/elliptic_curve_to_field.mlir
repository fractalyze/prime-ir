// Copyright 2025 The ZKIR Authors.
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

// RUN: cat %S/../../bn254_field_defs.mlir %S/../../bn254_ec_mont_defs.mlir %s \
// RUN:   | zkir-opt -elliptic-curve-to-field -split-input-file \
// RUN:   | FileCheck %s -enable-var-scope

// CHECK-LABEL: @test_initialization_and_conversion
func.func @test_initialization_and_conversion() {
  // CHECK: %[[VAR1:.*]] = field.constant [[ATTR1:.*]] : [[PF:.*]]
  %var1 = field.constant 1 : !PFm
  // CHECK: %[[VAR2:.*]] = field.constant [[ATTR2:.*]] : [[PF]]
  %var2 = field.constant 2 : !PFm
  // CHECK: %[[VAR4:.*]] = field.constant [[ATTR4:.*]] : [[PF]]
  %var4 = field.constant 4 : !PFm
  // CHECK: %[[VAR5:.*]] = field.constant [[ATTR5:.*]] : [[PF]]
  %var5 = field.constant 5 : !PFm
  // CHECK: %[[VAR8:.*]] = field.constant [[ATTR8:.*]] : [[PF]]
  %var8 = field.constant 8 : !PFm

  %g2_var1 = field.constant 1, 1 : !QFm
  %affine_g2 = elliptic_curve.point %g2_var1, %g2_var1 : (!QFm, !QFm) -> !g2affine

  // CHECK: elliptic_curve.point
  %affine1 = elliptic_curve.point %var1, %var5 : (!PFm, !PFm) -> !affine
  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : (!PFm, !PFm, !PFm) -> !jacobian
  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : (!PFm, !PFm, !PFm, !PFm) -> !xyzz

  // CHECK-NOT: elliptic_curve.convert_point_type
  // CHECK: elliptic_curve.extract
  // CHECK: elliptic_curve.point
  %jacobian2 = elliptic_curve.convert_point_type %affine1 : !affine -> !jacobian
  %xyzz2 = elliptic_curve.convert_point_type %affine1 : !affine -> !xyzz
  %affine2 = elliptic_curve.convert_point_type %jacobian1 : !jacobian -> !affine
  %xyzz3 = elliptic_curve.convert_point_type %jacobian1 : !jacobian -> !xyzz
  %affine3 = elliptic_curve.convert_point_type %xyzz1 : !xyzz -> !affine
  %jacobian3 = elliptic_curve.convert_point_type %xyzz1 : !xyzz -> !jacobian
  return
}

// CHECK-LABEL: @test_addition
func.func @test_addition(%affine1: !affine, %affine2: !affine, %jacobian1: !jacobian, %jacobian2: !jacobian, %xyzz1: !xyzz, %xyzz2: !xyzz) {
  // CHECK-NOT: elliptic_curve.add
  // affine, affine -> jacobian
  %affine3 = elliptic_curve.add %affine1, %affine2 : !affine, !affine -> !jacobian
  // affine, jacobian -> jacobian
  %jacobian3 = elliptic_curve.add %affine1, %jacobian1 : !affine, !jacobian -> !jacobian
  %jacobian4 = elliptic_curve.add %jacobian1, %affine1 : !jacobian, !affine -> !jacobian
  // affine, xyzz -> xyzz
  %xyzz3 = elliptic_curve.add %affine1, %xyzz1 : !affine, !xyzz -> !xyzz
  %xyzz4 = elliptic_curve.add %xyzz1, %affine1 : !xyzz, !affine -> !xyzz
  // jacobian, jacobian -> jacobian
  %jacobian5 = elliptic_curve.add %jacobian1, %jacobian2 : !jacobian, !jacobian -> !jacobian
  // xyzz, xyzz -> xyzz
  %xyzz5 = elliptic_curve.add %xyzz1, %xyzz2 : !xyzz, !xyzz -> !xyzz
  return
}

// CHECK-LABEL: @test_double
func.func @test_double(%affine1: !affine, %jacobian1: !jacobian, %xyzz1: !xyzz) {
  // CHECK-NOT: elliptic_curve.double
  %affine2 = elliptic_curve.double %affine1 : !affine -> !jacobian
  %jacobian2 = elliptic_curve.double %jacobian1 : !jacobian -> !jacobian
  %xyzz2 = elliptic_curve.double %xyzz1 : !xyzz -> !xyzz
  return
}

// CHECK-LABEL: @test_negation
func.func @test_negation(%affine1: !affine, %jacobian1: !jacobian, %xyzz1: !xyzz) {

  // CHECK-NOT: elliptic_curve.negate
  %affine2 = elliptic_curve.negate %affine1 : !affine
  %jacobian2 = elliptic_curve.negate %jacobian1 : !jacobian
  %xyzz2 = elliptic_curve.negate %xyzz1 : !xyzz
  return
}

// CHECK-LABEL: @test_subtraction
func.func @test_subtraction(%affine1: !affine, %affine2: !affine, %jacobian1: !jacobian, %jacobian2: !jacobian, %xyzz1: !xyzz, %xyzz2: !xyzz) {
  // CHECK-NOT: elliptic_curve.sub
  // affine, affine -> jacobian
  %affine3 = elliptic_curve.sub %affine1, %affine2 : !affine, !affine -> !jacobian
  // affine, jacobian -> jacobian
  %jacobian3 = elliptic_curve.sub %affine1, %jacobian1 : !affine, !jacobian -> !jacobian
  %jacobian4 = elliptic_curve.sub %jacobian1, %affine1 : !jacobian, !affine -> !jacobian
  // affine, xyzz -> xyzz
  %xyzz3 = elliptic_curve.sub %affine1, %xyzz1 : !affine, !xyzz -> !xyzz
  %xyzz4 = elliptic_curve.sub %xyzz1, %affine1 : !xyzz, !affine -> !xyzz
  // jacobian, jacobian -> jacobian
  %jacobian5 = elliptic_curve.sub %jacobian1, %jacobian2 : !jacobian, !jacobian -> !jacobian
  // xyzz, xyzz -> xyzz
  %xyzz5 = elliptic_curve.sub %xyzz1, %xyzz2 : !xyzz, !xyzz -> !xyzz
  return
}

// CHECK-LABEL: @test_scalar_mul
func.func @test_scalar_mul(%affine1: !affine, %jacobian1: !jacobian, %xyzz1: !xyzz, %var1: !SFm, %var2: !SFm) {
  // CHECK-NOT: elliptic_curve.scalar_mul
  %jacobian2 = elliptic_curve.scalar_mul %var1, %affine1 : !SFm, !affine -> !jacobian
  %jacobian3 = elliptic_curve.scalar_mul %var2, %affine1 : !SFm, !affine -> !jacobian

  %jacobian4 = elliptic_curve.scalar_mul %var1, %jacobian1 : !SFm, !jacobian -> !jacobian
  %jacobian5 = elliptic_curve.scalar_mul %var2, %jacobian1 : !SFm, !jacobian -> !jacobian

  %xyzz2 = elliptic_curve.scalar_mul %var1, %xyzz1 : !SFm, !xyzz -> !xyzz
  %xyzz3 = elliptic_curve.scalar_mul %var2, %xyzz1 : !SFm, !xyzz -> !xyzz
  return
}

// CHECK-LABEL: @test_cmp
func.func @test_cmp(%xyzz1: !xyzz, %xyzz2: !xyzz) {
  // CHECK-NOT: elliptic_curve.cmp
  %cmp1 = elliptic_curve.cmp eq, %xyzz1, %xyzz2 : !xyzz
  return
}

func.func @test_msm(%scalars: tensor<3x!SFm>, %points: tensor<3x!affine>) {
  %msm_result = elliptic_curve.msm %scalars, %points degree=2 : tensor<3x!SFm>, tensor<3x!affine> -> !jacobian
  return
}

func.func @test_g2_msm(%scalars: tensor<3x!SFm>, %points: tensor<3x!g2affine>) {
  %msm_result = elliptic_curve.msm %scalars, %points degree=2 : tensor<3x!SFm>, tensor<3x!g2affine> -> !g2jacobian
  return
}

func.func @test_memref(%arg0: memref<3x!affine>, %arg1: memref<3x!affine>) {
  %c0 = arith.constant 0 : index
  %p0 = memref.load %arg0[%c0] : memref<3x!affine>
  %p1 = memref.load %arg1[%c0] : memref<3x!affine>
  %sum = elliptic_curve.add %p0, %p1 : !affine, !affine -> !jacobian
  %affine = elliptic_curve.convert_point_type %sum : !jacobian -> !affine
  memref.store %affine, %arg0[%c0] : memref<3x!affine>
  return
}

// result bucket_indices = <(#numScalars * #numWindows) x index>
// result point_indices = <(#numScalars * #numWindows) x index>
func.func @test_scalar_decomp(%scalars: tensor<6x!SFm>) {
  %bucket_indices, %point_indices = elliptic_curve.scalar_decomp %scalars {bitsPerWindow = 2 : i16, scalarMaxBits = 4 : i16} : (tensor<6x!SFm>) -> (tensor<12xindex>, tensor<12xindex>)
  return
}

// result buckets = <#totalBuckets x pointType>
func.func @test_bucket_acc(%points: tensor<3x!affine>, %sorted_point_indices: tensor<6xindex>, %sorted_unique_bucket_indices: tensor<4xindex>, %bucket_offsets: tensor<5xindex>) {
  %msm_result = elliptic_curve.bucket_acc %points, %sorted_point_indices, %sorted_unique_bucket_indices, %bucket_offsets: (tensor<3x!affine>, tensor<6xindex>, tensor<4xindex>, tensor<5xindex>) -> tensor<8x!jacobian>
  return
}

// input buckets = <#windows x #bucketsPerWindow x pointType>
func.func @test_bucket_reduce(%buckets: tensor<?x?x!jacobian>) {
  %msm_result = elliptic_curve.bucket_reduce %buckets {scalarType = !SF}: (tensor<?x?x!jacobian>) -> tensor<?x!jacobian>
  return
}

// input windows = <#windows x pointType>
// CHECK-LABEL: @test_window_reduce
func.func @test_window_reduce(%windows: tensor<128x!jacobian>) {
  %msm_result = elliptic_curve.window_reduce %windows {bitsPerWindow = 2 : i16, scalarType = !SF}: (tensor<128x!jacobian>) -> !jacobian
  return
}
