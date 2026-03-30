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

// RUN: cat %S/../../bn254_defs.mlir %S/../../bn254_ec_mont_helpers.mlir %s \
// RUN:   | prime-ir-opt -elliptic-curve-to-field -split-input-file \
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

  %g2_var1 = field.constant [1, 1] : !QFm
  %affine_g2 = elliptic_curve.from_coords %g2_var1, %g2_var1 : (!QFm, !QFm) -> !g2affinem

  // CHECK: elliptic_curve.from_coords
  %affine1 = elliptic_curve.from_coords %var1, %var5 : (!PFm, !PFm) -> !affinem
  %jacobian1 = elliptic_curve.from_coords %var1, %var5, %var2 : (!PFm, !PFm, !PFm) -> !jacobianm
  %xyzz1 = elliptic_curve.from_coords %var1, %var5, %var4, %var8 : (!PFm, !PFm, !PFm, !PFm) -> !xyzzm

  // CHECK-NOT: elliptic_curve.convert_point_type
  // CHECK: elliptic_curve.to_coords
  // CHECK: elliptic_curve.from_coords
  %jacobian2 = elliptic_curve.convert_point_type %affine1 : !affinem -> !jacobianm
  %xyzz2 = elliptic_curve.convert_point_type %affine1 : !affinem -> !xyzzm
  %affine2 = elliptic_curve.convert_point_type %jacobian1 : !jacobianm -> !affinem
  %xyzz3 = elliptic_curve.convert_point_type %jacobian1 : !jacobianm -> !xyzzm
  %affine3 = elliptic_curve.convert_point_type %xyzz1 : !xyzzm -> !affinem
  %jacobian3 = elliptic_curve.convert_point_type %xyzz1 : !xyzzm -> !jacobianm
  return
}

// CHECK-LABEL: @test_constant
func.func @test_constant() {
  // CHECK-NOT: elliptic_curve.constant
  // G1 constants (prime field - single integers)
  %affine1 = elliptic_curve.constant dense<[1, 2]> : !affinem
  %jacobian1 = elliptic_curve.constant dense<[1, 2, 1]> : !jacobianm
  %xyzz1 = elliptic_curve.constant dense<[1, 2, 1, 1]> : !xyzzm
  // G2 constants (extension field - integer arrays)
  %g2_affine1 = elliptic_curve.constant dense<[[1, 1], [2, 3]]> : !g2affinem
  %g2_jacobian1 = elliptic_curve.constant dense<[[1, 1], [2, 3], [1, 0]]> : !g2jacobianm
  %g2_xyzz1 = elliptic_curve.constant dense<[[1, 1], [2, 3], [1, 0], [1, 0]]> : !g2xyzzm
  return
}

// CHECK-LABEL: @test_tensor_constant
func.func @test_tensor_constant() -> (tensor<2x!jacobianm>, tensor<2x!g2jacobianm>) {
  // CHECK-NOT: elliptic_curve.constant
  // CHECK: arith.constant
  // CHECK: field.bitcast
  // CHECK: elliptic_curve.bitcast
  // G1 tensor constants (prime field)
  %tensor_jacobian = elliptic_curve.constant dense<[[1, 2, 1], [3, 4, 1]]> : tensor<2x!jacobianm>
  // G2 tensor constants (extension field)
  %tensor_g2_jacobian = elliptic_curve.constant dense<[[[1, 1], [2, 2], [1, 0]], [[3, 3], [4, 4], [1, 0]]]> : tensor<2x!g2jacobianm>
  return %tensor_jacobian, %tensor_g2_jacobian : tensor<2x!jacobianm>, tensor<2x!g2jacobianm>
}

// CHECK-LABEL: @test_addition
func.func @test_addition(%affine1: !affinem, %affine2: !affinem, %jacobian1: !jacobianm, %jacobian2: !jacobianm, %xyzz1: !xyzzm, %xyzz2: !xyzzm) {
  // CHECK-NOT: elliptic_curve.add
  // affine, affine -> jacobian
  %affine3 = elliptic_curve.add %affine1, %affine2 : !affinem, !affinem -> !jacobianm
  // affine, jacobian -> jacobian
  %jacobian3 = elliptic_curve.add %affine1, %jacobian1 : !affinem, !jacobianm -> !jacobianm
  %jacobian4 = elliptic_curve.add %jacobian1, %affine1 : !jacobianm, !affinem -> !jacobianm
  // affine, xyzz -> xyzz
  %xyzz3 = elliptic_curve.add %affine1, %xyzz1 : !affinem, !xyzzm -> !xyzzm
  %xyzz4 = elliptic_curve.add %xyzz1, %affine1 : !xyzzm, !affinem -> !xyzzm
  // jacobian, jacobian -> jacobian
  %jacobian5 = elliptic_curve.add %jacobian1, %jacobian2 : !jacobianm, !jacobianm -> !jacobianm
  // xyzz, xyzz -> xyzz
  %xyzz5 = elliptic_curve.add %xyzz1, %xyzz2 : !xyzzm, !xyzzm -> !xyzzm
  return
}

// CHECK-LABEL: @test_double
func.func @test_double(%affine1: !affinem, %jacobian1: !jacobianm, %xyzz1: !xyzzm) {
  // CHECK-NOT: elliptic_curve.double
  %affine2 = elliptic_curve.double %affine1 : !affinem -> !jacobianm
  %jacobian2 = elliptic_curve.double %jacobian1 : !jacobianm -> !jacobianm
  %xyzz2 = elliptic_curve.double %xyzz1 : !xyzzm -> !xyzzm
  return
}

// CHECK-LABEL: @test_addition_cross_kind
func.func @test_addition_cross_kind(%affine1: !affinem, %affine2: !affinem) {
  // CHECK-NOT: elliptic_curve.add
  // affine, affine -> xyzz (natural result is jacobian, requires conversion)
  %xyzz1 = elliptic_curve.add %affine1, %affine2 : !affinem, !affinem -> !xyzzm
  return
}

// CHECK-LABEL: @test_double_cross_kind
func.func @test_double_cross_kind(%affine1: !affinem) {
  // CHECK-NOT: elliptic_curve.double
  // affine -> xyzz (natural result is jacobian, requires conversion)
  %xyzz1 = elliptic_curve.double %affine1 : !affinem -> !xyzzm
  return
}

// CHECK-LABEL: @test_negation
func.func @test_negation(%affine1: !affinem, %jacobian1: !jacobianm, %xyzz1: !xyzzm) {

  // CHECK-NOT: elliptic_curve.negate
  %affine2 = elliptic_curve.negate %affine1 : !affinem
  %jacobian2 = elliptic_curve.negate %jacobian1 : !jacobianm
  %xyzz2 = elliptic_curve.negate %xyzz1 : !xyzzm
  return
}

// CHECK-LABEL: @test_subtraction
func.func @test_subtraction(%affine1: !affinem, %affine2: !affinem, %jacobian1: !jacobianm, %jacobian2: !jacobianm, %xyzz1: !xyzzm, %xyzz2: !xyzzm) {
  // CHECK-NOT: elliptic_curve.sub
  // affine, affine -> jacobian
  %affine3 = elliptic_curve.sub %affine1, %affine2 : !affinem, !affinem -> !jacobianm
  // affine, jacobian -> jacobian
  %jacobian3 = elliptic_curve.sub %affine1, %jacobian1 : !affinem, !jacobianm -> !jacobianm
  %jacobian4 = elliptic_curve.sub %jacobian1, %affine1 : !jacobianm, !affinem -> !jacobianm
  // affine, xyzz -> xyzz
  %xyzz3 = elliptic_curve.sub %affine1, %xyzz1 : !affinem, !xyzzm -> !xyzzm
  %xyzz4 = elliptic_curve.sub %xyzz1, %affine1 : !xyzzm, !affinem -> !xyzzm
  // jacobian, jacobian -> jacobian
  %jacobian5 = elliptic_curve.sub %jacobian1, %jacobian2 : !jacobianm, !jacobianm -> !jacobianm
  // xyzz, xyzz -> xyzz
  %xyzz5 = elliptic_curve.sub %xyzz1, %xyzz2 : !xyzzm, !xyzzm -> !xyzzm
  return
}

// CHECK-LABEL: @test_scalar_mul
func.func @test_scalar_mul(%affine1: !affinem, %jacobian1: !jacobianm, %xyzz1: !xyzzm, %var1: !SFm, %var2: !SFm) {
  // CHECK-NOT: elliptic_curve.scalar_mul
  %jacobian2 = elliptic_curve.scalar_mul %var1, %affine1 : !SFm, !affinem -> !jacobianm
  %jacobian3 = elliptic_curve.scalar_mul %var2, %affine1 : !SFm, !affinem -> !jacobianm

  %jacobian4 = elliptic_curve.scalar_mul %var1, %jacobian1 : !SFm, !jacobianm -> !jacobianm
  %jacobian5 = elliptic_curve.scalar_mul %var2, %jacobian1 : !SFm, !jacobianm -> !jacobianm

  %xyzz2 = elliptic_curve.scalar_mul %var1, %xyzz1 : !SFm, !xyzzm -> !xyzzm
  %xyzz3 = elliptic_curve.scalar_mul %var2, %xyzz1 : !SFm, !xyzzm -> !xyzzm
  return
}

// CHECK-LABEL: @test_cmp
func.func @test_cmp(%xyzz1: !xyzzm, %xyzz2: !xyzzm) {
  // CHECK-NOT: elliptic_curve.cmp
  %cmp1 = elliptic_curve.cmp eq, %xyzz1, %xyzz2 : !xyzzm
  return
}

func.func @test_msm(%scalars: tensor<3x!SFm>, %points: tensor<3x!affinem>) {
  %msm_result = elliptic_curve.msm %scalars, %points degree=2 : tensor<3x!SFm>, tensor<3x!affinem> -> !jacobianm
  return
}

func.func @test_g2_msm(%scalars: tensor<3x!SFm>, %points: tensor<3x!g2affinem>) {
  %msm_result = elliptic_curve.msm %scalars, %points degree=2 : tensor<3x!SFm>, tensor<3x!g2affinem> -> !g2jacobianm
  return
}

func.func @test_memref(%arg0: memref<3x!affinem>, %arg1: memref<3x!affinem>) {
  %c0 = arith.constant 0 : index
  %p0 = memref.load %arg0[%c0] : memref<3x!affinem>
  %p1 = memref.load %arg1[%c0] : memref<3x!affinem>
  %sum = elliptic_curve.add %p0, %p1 : !affinem, !affinem -> !jacobianm
  %affine = elliptic_curve.convert_point_type %sum : !jacobianm -> !affinem
  memref.store %affine, %arg0[%c0] : memref<3x!affinem>
  return
}
