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

// RUN: cat %S/../../bn254_field_defs.mlir %S/../../bn254_ec_mont_defs.mlir %s \
// RUN:   | prime-ir-opt -convert-to-llvm -split-input-file \
// RUN:   | FileCheck %s -enable-var-scope

// CHECK-LABEL: @test_ext_from_coord
func.func @test_ext_from_coord(%var1: i256, %var2: i256, %var3: i256) {
  // CHECK-NOT: elliptic_curve.ext_from_coord
  %affine = elliptic_curve.ext_from_coord %var1, %var2 : (i256, i256) -> !affine
  %jacobian = elliptic_curve.ext_from_coord %var1, %var2, %var3 : (i256, i256, i256) -> !jacobian
  %xyzz = elliptic_curve.ext_from_coord %var1, %var2, %var3, %var1 : (i256, i256, i256, i256) -> !xyzz
  return
}

// CHECK-LABEL: @test_ext_to_coords
func.func @test_ext_to_coords(%affine: !affine, %jacobian: !jacobian, %xyzz: !xyzz) {
  // CHECK-NOT: elliptic_curve.ext_to_coords
  %affine_coords:2 = elliptic_curve.ext_to_coords %affine : !affine -> i256, i256
  %jacobian_coords:3 = elliptic_curve.ext_to_coords %jacobian : !jacobian -> i256, i256, i256
  %xyzz_coords:4 = elliptic_curve.ext_to_coords %xyzz : !xyzz -> i256, i256, i256, i256
  return
}

// CHECK-LABEL: @test_ext_from_coord_g2
func.func @test_ext_from_coord_g2(%var1: !llvm.struct<(i256, i256)>, %var2: !llvm.struct<(i256, i256)>, %var3: !llvm.struct<(i256, i256)>) {
  // CHECK-NOT: elliptic_curve.ext_from_coord
  %affine = elliptic_curve.ext_from_coord %var1, %var2 : (!llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>) -> !g2affine
  %jacobian = elliptic_curve.ext_from_coord %var1, %var2, %var3 : (!llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>) -> !g2jacobian
  %xyzz = elliptic_curve.ext_from_coord %var1, %var2, %var3, %var1 : (!llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>) -> !g2xyzz
  return
}

// CHECK-LABEL: @test_ext_to_coords_g2
func.func @test_ext_to_coords_g2(%affine: !g2affine, %jacobian: !g2jacobian, %xyzz: !g2xyzz) {
  // CHECK-NOT: elliptic_curve.ext_to_coords
  %affine_coords:2 = elliptic_curve.ext_to_coords %affine : !g2affine -> !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>
  %jacobian_coords:3 = elliptic_curve.ext_to_coords %jacobian : !g2jacobian -> !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>
  %xyzz_coords:4 = elliptic_curve.ext_to_coords %xyzz : !g2xyzz -> !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>
  return
}
