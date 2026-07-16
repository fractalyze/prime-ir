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

// Executes the specialized Fermat-chain inverse (inverse-algorithm=fermat) for
// each registered field — Goldilocks, BabyBear, KoalaBear — and checks
// a * inv(a) == 1 (canonical, and Montgomery for Goldilocks), the p-1 boundary,
// and inv(0) == 0 (matching safegcd's convention, so 0 * inv(0) == 0). The
// Goldilocks case also covers the cubic extension's internal base inverse.

// RUN: prime-ir-opt %s -field-to-mod-arith="inverse-algorithm=fermat" -mod-arith-to-arith -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_goldilocks_inverse -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_INV < %t

// RUN: prime-ir-opt %s -field-to-mod-arith="inverse-algorithm=fermat" -mod-arith-to-arith -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_goldilocks_mont_inverse -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_MONT_INV < %t

// RUN: prime-ir-opt %s -field-to-mod-arith="inverse-algorithm=fermat" -mod-arith-to-arith -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_goldilocks_inverse_edge -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_EDGE < %t

// RUN: prime-ir-opt %s -field-to-mod-arith="inverse-algorithm=fermat" -mod-arith-to-arith -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_goldilocks_inverse_zero -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_ZERO < %t

// RUN: prime-ir-opt %s -field-to-mod-arith="inverse-algorithm=fermat" -mod-arith-to-arith -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_goldilocks_cubic_inverse -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_CUBIC < %t

// RUN: prime-ir-opt %s -field-to-mod-arith="inverse-algorithm=fermat" -mod-arith-to-arith -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_babybear_inverse -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_BB < %t

// RUN: prime-ir-opt %s -field-to-mod-arith="inverse-algorithm=fermat" -mod-arith-to-arith -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_koalabear_inverse -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_KB < %t

// RUN: prime-ir-opt %s -field-to-mod-arith="inverse-algorithm=fermat" -mod-arith-to-arith -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_mersenne31_inverse -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_M31 < %t

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

// Goldilocks: p = 2^64 - 2^32 + 1 = 18446744069414584321
!G = !field.pf<18446744069414584321 : i64>
!Gm = !field.pf<18446744069414584321 : i64, true>
!C = !field.ef<3x!G, 7:i64>

func.func @check(%v: !G) {
  %vi = field.bitcast %v : !G -> i64
  %v32 = arith.trunci %vi : i64 to i32
  %t = tensor.from_elements %v32 : tensor<1xi32>
  %m = bufferization.to_buffer %t : tensor<1xi32> to memref<1xi32>
  %U = memref.cast %m : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

func.func @test_goldilocks_inverse() {
  %ai = arith.constant 123456789 : i64
  %a = field.bitcast %ai : i64 -> !G
  %inv = field.inverse %a : !G
  %check = field.mul %inv, %a : !G
  func.call @check(%check) : (!G) -> ()
  return
}
// CHECK_INV: [1]

func.func @test_goldilocks_mont_inverse() {
  %ai = arith.constant 987654321 : i64
  %a = field.bitcast %ai : i64 -> !G
  %a_mont = field.to_mont %a : !Gm
  %inv_mont = field.inverse %a_mont : !Gm
  %check_mont = field.mul %inv_mont, %a_mont : !Gm
  %check = field.from_mont %check_mont : !G
  func.call @check(%check) : (!G) -> ()
  return
}
// CHECK_MONT_INV: [1]

func.func @test_goldilocks_inverse_edge() {
  // a = p - 1, whose inverse is itself; a * inv(a) must still be 1.
  %ai = arith.constant 18446744069414584320 : i64
  %a = field.bitcast %ai : i64 -> !G
  %inv = field.inverse %a : !G
  %check = field.mul %inv, %a : !G
  func.call @check(%check) : (!G) -> ()
  return
}
// CHECK_EDGE: [1]

func.func @test_goldilocks_inverse_zero() {
  // inv(0) = 0^(p-2) = 0 (matches safegcd's convention for a non-invertible 0);
  // a * inv(a) = 0.
  %ai = arith.constant 0 : i64
  %a = field.bitcast %ai : i64 -> !G
  %inv = field.inverse %a : !G
  %check = field.mul %inv, %a : !G
  func.call @check(%check) : (!G) -> ()
  return
}
// CHECK_ZERO: [0]

func.func @check3(%v: !C) {
  %v0, %v1, %v2 = field.ext_to_coeffs %v : (!C) -> (!G, !G, !G)
  %i0 = field.bitcast %v0 : !G -> i64
  %i1 = field.bitcast %v1 : !G -> i64
  %i2 = field.bitcast %v2 : !G -> i64
  %c0 = arith.trunci %i0 : i64 to i32
  %c1 = arith.trunci %i1 : i64 to i32
  %c2 = arith.trunci %i2 : i64 to i32
  %t = tensor.from_elements %c0, %c1, %c2 : tensor<3xi32>
  %m = bufferization.to_buffer %t : tensor<3xi32> to memref<3xi32>
  %U = memref.cast %m : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

// Cubic (Goldilocks^3) inverse: exercises the norm-trick's base inverse running
// as the Fermat chain; a * inv(a) must be the identity (1, 0, 0).
func.func @test_goldilocks_cubic_inverse() {
  %a = field.constant [7, 11, 13] : !C
  %inv = field.inverse %a : !C
  %check = field.mul %inv, %a : !C
  func.call @check3(%check) : (!C) -> ()
  return
}
// CHECK_CUBIC: [1, 0, 0]

// BabyBear: p = 2^31 - 2^27 + 1 = 2013265921. Each result element is
// a * inv(a): [canonical, p-1 boundary, zero] = [1, 1, 0].
!BB = !field.pf<2013265921 : i32>

func.func @test_babybear_inverse() {
  %c0 = arith.constant 123456 : i32
  %a0 = field.bitcast %c0 : i32 -> !BB
  %i0 = field.inverse %a0 : !BB
  %m0 = field.mul %i0, %a0 : !BB
  %r0 = field.bitcast %m0 : !BB -> i32

  %c1 = arith.constant 2013265920 : i32 // p - 1
  %a1 = field.bitcast %c1 : i32 -> !BB
  %i1 = field.inverse %a1 : !BB
  %m1 = field.mul %i1, %a1 : !BB
  %r1 = field.bitcast %m1 : !BB -> i32

  %c2 = arith.constant 0 : i32
  %a2 = field.bitcast %c2 : i32 -> !BB
  %i2 = field.inverse %a2 : !BB
  %m2 = field.mul %i2, %a2 : !BB
  %r2 = field.bitcast %m2 : !BB -> i32

  %t = tensor.from_elements %r0, %r1, %r2 : tensor<3xi32>
  %m = bufferization.to_buffer %t : tensor<3xi32> to memref<3xi32>
  %U = memref.cast %m : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}
// CHECK_BB: [1, 1, 0]

// KoalaBear: p = 2^31 - 2^24 + 1 = 2130706433. [canonical, p-1, zero] = [1, 1, 0].
!KB = !field.pf<2130706433 : i32>

func.func @test_koalabear_inverse() {
  %c0 = arith.constant 654321 : i32
  %a0 = field.bitcast %c0 : i32 -> !KB
  %i0 = field.inverse %a0 : !KB
  %m0 = field.mul %i0, %a0 : !KB
  %r0 = field.bitcast %m0 : !KB -> i32

  %c1 = arith.constant 2130706432 : i32 // p - 1
  %a1 = field.bitcast %c1 : i32 -> !KB
  %i1 = field.inverse %a1 : !KB
  %m1 = field.mul %i1, %a1 : !KB
  %r1 = field.bitcast %m1 : !KB -> i32

  %c2 = arith.constant 0 : i32
  %a2 = field.bitcast %c2 : i32 -> !KB
  %i2 = field.inverse %a2 : !KB
  %m2 = field.mul %i2, %a2 : !KB
  %r2 = field.bitcast %m2 : !KB -> i32

  %t = tensor.from_elements %r0, %r1, %r2 : tensor<3xi32>
  %m = bufferization.to_buffer %t : tensor<3xi32> to memref<3xi32>
  %U = memref.cast %m : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}
// CHECK_KB: [1, 1, 0]

// Mersenne31: p = 2^31 - 1 = 2147483647. Auto-detected Solinas (a=31, b=1),
// lowered via the general chain (no hand-tuned override). [canonical, p-1, zero]
// = [1, 1, 0] confirms the general chain computes a^(p-2) correctly.
!M31 = !field.pf<2147483647 : i32>

func.func @test_mersenne31_inverse() {
  %c0 = arith.constant 424242 : i32
  %a0 = field.bitcast %c0 : i32 -> !M31
  %i0 = field.inverse %a0 : !M31
  %m0 = field.mul %i0, %a0 : !M31
  %r0 = field.bitcast %m0 : !M31 -> i32

  %c1 = arith.constant 2147483646 : i32 // p - 1
  %a1 = field.bitcast %c1 : i32 -> !M31
  %i1 = field.inverse %a1 : !M31
  %m1 = field.mul %i1, %a1 : !M31
  %r1 = field.bitcast %m1 : !M31 -> i32

  %c2 = arith.constant 0 : i32
  %a2 = field.bitcast %c2 : i32 -> !M31
  %i2 = field.inverse %a2 : !M31
  %m2 = field.mul %i2, %a2 : !M31
  %r2 = field.bitcast %m2 : !M31 -> i32

  %t = tensor.from_elements %r0, %r1, %r2 : tensor<3xi32>
  %m = bufferization.to_buffer %t : tensor<3xi32> to memref<3xi32>
  %U = memref.cast %m : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}
// CHECK_M31: [1, 1, 0]
