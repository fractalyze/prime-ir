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

// RUN: prime-ir-opt -canonicalize -mod-arith-to-arith  -split-input-file %s | FileCheck %s --check-prefix=CHECK-LOWERING

// RUN: prime-ir-opt %s -canonicalize -mod-arith-to-arith -convert-to-llvm \
// RUN:   | mlir-runner -e main -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

!mod = !mod_arith.int<2013265921 : i32>
!modm = !mod_arith.int<2013265921 : i32, true>

func.func @test_halve(%arg0 : !modm) -> !modm {
  %c2 = mod_arith.constant 2 : !mod
  %two = mod_arith.to_mont %c2 : !modm
  %inv_two = mod_arith.inverse %two : !modm

  // CHECK-LOWERING: arith.shrui
  %res = mod_arith.mul %arg0, %inv_two : !modm
  return %res : !modm
}

func.func @test_mul_by_inv_2_16(%arg0 : !modm) -> !modm {
  %c2 = mod_arith.constant 2 : !mod
  %two = mod_arith.to_mont %c2 : !modm
  %two_2 = mod_arith.square %two : !modm
  %two_4 = mod_arith.square %two_2 : !modm
  %two_8 = mod_arith.square %two_4 : !modm
  %two_16 = mod_arith.square %two_8 : !modm
  %inv_two_16 = mod_arith.inverse %two_16 : !modm

  // CHECK-LOWERING: arith.shli
  %res = mod_arith.mul %arg0, %inv_two_16 : !modm
  return %res : !modm
}

func.func @test_mul_by_neg_inv_2_16(%arg0 : !modm) -> !modm {
  %c2 = mod_arith.constant 2 : !mod
  %two = mod_arith.to_mont %c2 : !modm
  %two_2 = mod_arith.square %two : !modm
  %two_4 = mod_arith.square %two_2 : !modm
  %two_8 = mod_arith.square %two_4 : !modm
  %two_16 = mod_arith.square %two_8 : !modm
  %inv_two_16 = mod_arith.inverse %two_16 : !modm
  %neg_inv_two_16 = mod_arith.negate %inv_two_16 : !modm

  // CHECK-LOWERING: arith.shli
  %res = mod_arith.mul %arg0, %neg_inv_two_16 : !modm
  return %res : !modm
}

func.func @test_mul_by_inv_2_27(%arg0 : !modm) -> !modm {
  %c2 = mod_arith.constant 2 : !mod
  %two = mod_arith.to_mont %c2 : !modm
  %two_2 = mod_arith.square %two : !modm
  %two_4 = mod_arith.square %two_2 : !modm
  %two_8 = mod_arith.square %two_4 : !modm
  %two_16 = mod_arith.square %two_8 : !modm
  %two_24 = mod_arith.mul %two_8, %two_16 : !modm
  %two_26 = mod_arith.mul %two_24, %two_2 : !modm
  %two_27 = mod_arith.mul %two_26, %two : !modm
  %inv_two_27 = mod_arith.inverse %two_27 : !modm

  // CHECK-LOWERING: arith.shli
  %res = mod_arith.mul %arg0, %inv_two_27 : !modm
  return %res : !modm
}

func.func @test_mul_by_neg_inv_2_27(%arg0 : !modm) -> !modm {
  %c2 = mod_arith.constant 2 : !mod
  %two = mod_arith.to_mont %c2 : !modm
  %two_2 = mod_arith.square %two : !modm
  %two_4 = mod_arith.square %two_2 : !modm
  %two_8 = mod_arith.square %two_4 : !modm
  %two_16 = mod_arith.square %two_8 : !modm
  %two_24 = mod_arith.mul %two_8, %two_16 : !modm
  %two_26 = mod_arith.mul %two_24, %two_2 : !modm
  %two_27 = mod_arith.mul %two_26, %two : !modm
  %inv_two_27 = mod_arith.inverse %two_27 : !modm
  %neg_inv_two_27 = mod_arith.negate %inv_two_27 : !modm

  // CHECK-LOWERING: arith.shli
  %res = mod_arith.mul %arg0, %neg_inv_two_27 : !modm
  return %res : !modm
}

func.func @main() {
  %c2 = mod_arith.constant 2 : !mod
  %two = mod_arith.to_mont %c2 : !modm
  %two_2 = mod_arith.square %two : !modm
  %two_4 = mod_arith.square %two_2 : !modm
  %two_8 = mod_arith.square %two_4 : !modm
  %two_16 = mod_arith.square %two_8 : !modm
  %two_17 = mod_arith.mul %two, %two_16 : !modm
  %two_32 = mod_arith.square %two_16 : !modm

  %mem = memref.alloca() : memref<1xi32>
  %mem_cast = memref.cast %mem : memref<1xi32> to memref<*xi32>
  %idx = arith.constant 0 : index

  %res = func.call @test_halve(%two) : (!modm) -> !modm
  %res_std = mod_arith.from_mont %res : !mod
  %res_int = mod_arith.bitcast %res_std : !mod -> i32
  memref.store %res_int, %mem[%idx] : memref<1xi32>
  func.call @printMemrefI32(%mem_cast) : (memref<*xi32>) -> ()

  %res1 = func.call @test_mul_by_inv_2_16(%two_17) : (!modm) -> !modm
  %res1_std = mod_arith.from_mont %res1 : !mod
  %res1_int = mod_arith.bitcast %res1_std : !mod -> i32
  memref.store %res1_int, %mem[%idx] : memref<1xi32>
  func.call @printMemrefI32(%mem_cast) : (memref<*xi32>) -> ()

  %res2 = func.call @test_mul_by_neg_inv_2_16(%two_16) : (!modm) -> !modm
  %res2_std = mod_arith.from_mont %res2 : !mod
  %res2_int = mod_arith.bitcast %res2_std : !mod -> i32
  memref.store %res2_int, %mem[%idx] : memref<1xi32>
  func.call @printMemrefI32(%mem_cast) : (memref<*xi32>) -> ()

  %res3 = func.call @test_mul_by_inv_2_27(%two_32) : (!modm) -> !modm
  %res3_std = mod_arith.from_mont %res3 : !mod
  %res3_int = mod_arith.bitcast %res3_std : !mod -> i32
  memref.store %res3_int, %mem[%idx] : memref<1xi32>
  func.call @printMemrefI32(%mem_cast) : (memref<*xi32>) -> ()

  %res4 = func.call @test_mul_by_neg_inv_2_27(%two_32) : (!modm) -> !modm
  %res4_std = mod_arith.from_mont %res4 : !mod
  %res4_int = mod_arith.bitcast %res4_std : !mod -> i32
  memref.store %res4_int, %mem[%idx] : memref<1xi32>
  func.call @printMemrefI32(%mem_cast) : (memref<*xi32>) -> ()
  return
}

// CHECK: [1]
// CHECK: [2]
// CHECK: [2013265920]
// CHECK: [32]
// CHECK: [2013265889]
