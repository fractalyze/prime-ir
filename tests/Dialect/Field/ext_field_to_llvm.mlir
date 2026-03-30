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

// RUN: cat %S/../../bn254_defs.mlir %s \
// RUN:   | prime-ir-opt -convert-to-llvm -split-input-file \
// RUN:   | FileCheck %s -enable-var-scope

// CHECK-LABEL: @test_ext_from_coeffs
func.func @test_ext_from_coeffs(%var1: i256, %var2: i256) -> !QFm {
  // CHECK-NOT: field.ext_from_coeffs
  %ext = field.ext_from_coeffs %var1, %var2 : (i256, i256) -> !QFm
  return %ext : !QFm
}

// CHECK-LABEL: @test_ext_to_coeffs
func.func @test_ext_to_coeffs(%ext: !QFm) -> i256 {
  // CHECK-NOT: field.ext_to_coeffs
  %coeffs:2 = field.ext_to_coeffs %ext : (!QFm) -> (i256, i256)
  return %coeffs#1 : i256
}

// -----

// Memref bitcast descriptor rebuild tests.
//
// After bufferization, field.bitcast operates on memrefs.  ConvertBitcast must
// build a NEW descriptor with correct sizes/strides/offset for the output
// element type.  Before the fix, an unrealized_conversion_cast forwarded the
// input descriptor unchanged, leaving sizes in input-element units.  That made
// memref.copy compute:
//   PF→EF2:  4 × sizeof(struct{i32,i32}) = 32 bytes  (should be 16)
//   EF2→PF:  2 × sizeof(i32) = 8 bytes               (should be 16)

!PF  = !field.pf<7:i32>
!EF2 = !field.ef<2x!PF, 6:i32>
!EF6 = !field.ef<3x!EF2, 2:i32>

// Upward: memref<4xi32> → memref<2x!EF2>  (degree ratio = 2)
//
// Descriptor fields:
//   allocPtr / alignedPtr — passed through from input
//   offset   = sdiv(input_offset, 2)
//   sizes[0] = 2  (static, from output type)
//   strides[0] = 1  (identity layout)
//
// Before the fix, an unrealized_conversion_cast forwarded the descriptor
// unchanged (sizes=[4], stride=1).  memref.copy then computed
//   4 × sizeof(!llvm.struct<(i32, i32)>) = 32 bytes
// instead of the correct 2 × 8 = 16.
//
// CHECK-LABEL: @bitcast_i32_to_ef2
//       CHECK:   llvm.mlir.poison
//       CHECK:   llvm.insertvalue {{%.+}}, {{%.+}}[0]
//       CHECK:   llvm.insertvalue {{%.+}}, {{%.+}}[1]
//       CHECK:   [[R2_U:%.+]] = llvm.mlir.constant(2 : index) : i64
//       CHECK:   llvm.sdiv {{%.+}}, [[R2_U]] : i64
//       CHECK:   llvm.insertvalue {{%.+}}, {{%.+}}[2]
//       CHECK:   [[SZ2:%.+]] = llvm.mlir.constant(2 : index) : i64
//       CHECK:   llvm.insertvalue [[SZ2]], {{%.+}}[3, 0]
//       CHECK:   [[ST1_U:%.+]] = llvm.mlir.constant(1 : index) : i64
//       CHECK:   llvm.insertvalue [[ST1_U]], {{%.+}}[4, 0]
func.func @bitcast_i32_to_ef2(%arg0: memref<4xi32>) -> memref<2x!EF2> {
  %0 = field.bitcast %arg0 : memref<4xi32> -> memref<2x!EF2>
  return %0 : memref<2x!EF2>
}

// Downward: memref<2x!EF2> → memref<4xi32>  (degree ratio = 2)
//
// offset = mul(input_offset, 2),  sizes[0] = 4,  strides[0] = 1
//
// CHECK-LABEL: @bitcast_ef2_to_i32
//       CHECK:   [[R2_D:%.+]] = llvm.mlir.constant(2 : index) : i64
//       CHECK:   llvm.mul {{%.+}}, [[R2_D]] : i64
//       CHECK:   llvm.insertvalue {{%.+}}, {{%.+}}[2]
//       CHECK:   [[SZ4:%.+]] = llvm.mlir.constant(4 : index) : i64
//       CHECK:   llvm.insertvalue [[SZ4]], {{%.+}}[3, 0]
//       CHECK:   [[ST1_D:%.+]] = llvm.mlir.constant(1 : index) : i64
//       CHECK:   llvm.insertvalue [[ST1_D]], {{%.+}}[4, 0]
func.func @bitcast_ef2_to_i32(%arg0: memref<2x!EF2>) -> memref<4xi32> {
  %0 = field.bitcast %arg0 : memref<2x!EF2> -> memref<4xi32>
  return %0 : memref<4xi32>
}

// Tower upward: memref<6xi32> → memref<1x!EF6>  (degree ratio = 6)
//
// CHECK-LABEL: @bitcast_i32_to_ef6
//       CHECK:   [[R6:%.+]] = llvm.mlir.constant(6 : index) : i64
//       CHECK:   llvm.sdiv {{%.+}}, [[R6]] : i64
//       CHECK:   [[SZ1_T:%.+]] = llvm.mlir.constant(1 : index) : i64
//       CHECK:   llvm.insertvalue [[SZ1_T]], {{%.+}}[3, 0]
func.func @bitcast_i32_to_ef6(%arg0: memref<6xi32>) -> memref<1x!EF6> {
  %0 = field.bitcast %arg0 : memref<6xi32> -> memref<1x!EF6>
  return %0 : memref<1x!EF6>
}

// Tower-to-tower: memref<3x!EF2> → memref<1x!EF6>  (degree ratio = 3)
//
// CHECK-LABEL: @bitcast_ef2_to_ef6
//       CHECK:   [[R3:%.+]] = llvm.mlir.constant(3 : index) : i64
//       CHECK:   llvm.sdiv {{%.+}}, [[R3]] : i64
//       CHECK:   [[SZ1_TT:%.+]] = llvm.mlir.constant(1 : index) : i64
//       CHECK:   llvm.insertvalue [[SZ1_TT]], {{%.+}}[3, 0]
func.func @bitcast_ef2_to_ef6(%arg0: memref<3x!EF2>) -> memref<1x!EF6> {
  %0 = field.bitcast %arg0 : memref<3x!EF2> -> memref<1x!EF6>
  return %0 : memref<1x!EF6>
}
