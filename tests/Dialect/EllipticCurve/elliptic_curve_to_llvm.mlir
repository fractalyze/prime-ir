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
// RUN:   | prime-ir-opt -convert-to-llvm -split-input-file \
// RUN:   | FileCheck %s -enable-var-scope

// CHECK-LABEL: @test_from_coords
func.func @test_from_coords(%var1: i256, %var2: i256, %var3: i256) {
  // CHECK-NOT: elliptic_curve.from_coords
  %affine = elliptic_curve.from_coords %var1, %var2 : (i256, i256) -> !affinem
  %jacobian = elliptic_curve.from_coords %var1, %var2, %var3 : (i256, i256, i256) -> !jacobianm
  %xyzz = elliptic_curve.from_coords %var1, %var2, %var3, %var1 : (i256, i256, i256, i256) -> !xyzzm
  return
}

// CHECK-LABEL: @test_to_coords
func.func @test_to_coords(%affine: !affinem, %jacobian: !jacobianm, %xyzz: !xyzzm) {
  // CHECK-NOT: elliptic_curve.to_coords
  %affine_coords:2 = elliptic_curve.to_coords %affine : (!affinem) -> (i256, i256)
  %jacobian_coords:3 = elliptic_curve.to_coords %jacobian : (!jacobianm) -> (i256, i256, i256)
  %xyzz_coords:4 = elliptic_curve.to_coords %xyzz : (!xyzzm) -> (i256, i256, i256, i256)
  return
}

// CHECK-LABEL: @test_from_coords_g2
func.func @test_from_coords_g2(%var1: !llvm.struct<(i256, i256)>, %var2: !llvm.struct<(i256, i256)>, %var3: !llvm.struct<(i256, i256)>) {
  // CHECK-NOT: elliptic_curve.from_coords
  %affine = elliptic_curve.from_coords %var1, %var2 : (!llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>) -> !g2affinem
  %jacobian = elliptic_curve.from_coords %var1, %var2, %var3 : (!llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>) -> !g2jacobianm
  %xyzz = elliptic_curve.from_coords %var1, %var2, %var3, %var1 : (!llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>) -> !g2xyzzm
  return
}

// CHECK-LABEL: @test_to_coords_g2
func.func @test_to_coords_g2(%affine: !g2affinem, %jacobian: !g2jacobianm, %xyzz: !g2xyzzm) {
  // CHECK-NOT: elliptic_curve.to_coords
  %affine_coords:2 = elliptic_curve.to_coords %affine : (!g2affinem) -> (!llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>)
  %jacobian_coords:3 = elliptic_curve.to_coords %jacobian : (!g2jacobianm) -> (!llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>)
  %xyzz_coords:4 = elliptic_curve.to_coords %xyzz : (!g2xyzzm) -> (!llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>)
  return
}

// -----

// Memref bitcast descriptor rebuild tests.
//
// After bufferization, elliptic_curve.bitcast operates on memrefs, reinterpreting
// one buffer as N points <-> N*K contiguous coordinates.  ConvertBitcast must
// build a NEW descriptor with sizes/strides/offset in the OUTPUT element's units.
// Before the fix, an unrealized_conversion_cast forwarded the input descriptor
// unchanged — leaving sizes in input-element units — so a later dealloc/copy
// computed the wrong byte count (heap corruption: munmap_chunk invalid pointer).
// Mirrors tests/Dialect/Field/ext_field_to_llvm.mlir.
//
// -split-input-file parses each chunk independently, so the standard-form G1
// types are redefined locally here (the bn254_defs.mlir aliases live in the
// first chunk only).

!PF = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PF
!affine = !elliptic_curve.affine<#curve>
!jacobian = !elliptic_curve.jacobian<#curve>

// Pack: memref<8x!PF> -> memref<4x!affine>  (2 coords per affine point)
//
// Descriptor fields:
//   allocPtr / alignedPtr — passed through from input
//   offset   = sdiv(input_offset, 2)
//   sizes[0] = 4  (static, from output type)
//   strides[0] = 1  (identity layout)
//
// CHECK-LABEL: @bitcast_pf_to_affine
//       CHECK:   llvm.mlir.poison
//       CHECK:   llvm.insertvalue {{%.+}}, {{%.+}}[0]
//       CHECK:   llvm.insertvalue {{%.+}}, {{%.+}}[1]
//       CHECK:   [[R2_U:%.+]] = llvm.mlir.constant(2 : index) : i64
//       CHECK:   llvm.sdiv {{%.+}}, [[R2_U]] : i64
//       CHECK:   llvm.insertvalue {{%.+}}, {{%.+}}[2]
//       CHECK:   [[SZ4:%.+]] = llvm.mlir.constant(4 : index) : i64
//       CHECK:   llvm.insertvalue [[SZ4]], {{%.+}}[3, 0]
//       CHECK:   [[ST1_U:%.+]] = llvm.mlir.constant(1 : index) : i64
//       CHECK:   llvm.insertvalue [[ST1_U]], {{%.+}}[4, 0]
func.func @bitcast_pf_to_affine(%arg0: memref<8x!PF>) -> memref<4x!affine> {
  %0 = elliptic_curve.bitcast %arg0 : memref<8x!PF> -> memref<4x!affine>
  return %0 : memref<4x!affine>
}

// Unpack: memref<4x!affine> -> memref<8x!PF>  (2 coords per affine point)
//
// offset = mul(input_offset, 2),  sizes[0] = 8,  strides[0] = 1
//
// CHECK-LABEL: @bitcast_affine_to_pf
//       CHECK:   [[R2_D:%.+]] = llvm.mlir.constant(2 : index) : i64
//       CHECK:   llvm.mul {{%.+}}, [[R2_D]] : i64
//       CHECK:   llvm.insertvalue {{%.+}}, {{%.+}}[2]
//       CHECK:   [[SZ8:%.+]] = llvm.mlir.constant(8 : index) : i64
//       CHECK:   llvm.insertvalue [[SZ8]], {{%.+}}[3, 0]
//       CHECK:   [[ST1_D:%.+]] = llvm.mlir.constant(1 : index) : i64
//       CHECK:   llvm.insertvalue [[ST1_D]], {{%.+}}[4, 0]
func.func @bitcast_affine_to_pf(%arg0: memref<4x!affine>) -> memref<8x!PF> {
  %0 = elliptic_curve.bitcast %arg0 : memref<4x!affine> -> memref<8x!PF>
  return %0 : memref<8x!PF>
}

// Jacobian: memref<6x!PF> -> memref<2x!jacobian>  (3 coords per point)
//
// CHECK-LABEL: @bitcast_pf_to_jacobian
//       CHECK:   [[R3:%.+]] = llvm.mlir.constant(3 : index) : i64
//       CHECK:   llvm.sdiv {{%.+}}, [[R3]] : i64
//       CHECK:   [[SZ2:%.+]] = llvm.mlir.constant(2 : index) : i64
//       CHECK:   llvm.insertvalue [[SZ2]], {{%.+}}[3, 0]
func.func @bitcast_pf_to_jacobian(%arg0: memref<6x!PF>) -> memref<2x!jacobian> {
  %0 = elliptic_curve.bitcast %arg0 : memref<6x!PF> -> memref<2x!jacobian>
  return %0 : memref<2x!jacobian>
}
