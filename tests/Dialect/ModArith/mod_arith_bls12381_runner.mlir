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

// Runner tests for the BLS12-381 base field (384-bit modulus). Exercises the
// Bernstein-Yang inverter codegen path with `divsteps > 64`, which goes
// through the `limbBitWidth = n` branch in BYInverter.cpp. Catches the bug
// where modular inverse computed via the lowered MLIR code disagreed with
// zk_dtypes::bls12_381::FqMont::Inverse() for any non-trivial denominator.

// RUN: prime-ir-opt %s -mod-arith-to-arith -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_bls12381_inverse -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_INVERSE < %t

// RUN: prime-ir-opt %s -mod-arith-to-arith -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_bls12381_mont_inverse -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_MONT_INVERSE < %t

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

// BLS12-381 base field prime:
// p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f624
//     1eabfffeb153ffffb9feffffffffaaab
!Fp = !mod_arith.int<4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787 : i384>
!Fpm = !mod_arith.int<4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787 : i384, true>

// Test inverse: inv(a) * a == 1 in the canonical (non-Montgomery) domain.
func.func @test_bls12381_inverse() {
  %a = mod_arith.constant 3723 : !Fp
  %inv = mod_arith.inverse %a : !Fp
  %check = mod_arith.mul %inv, %a : !Fp
  %v = mod_arith.bitcast %check : !Fp -> i384
  %v32 = arith.trunci %v : i384 to i32
  %t = tensor.from_elements %v32 : tensor<1xi32>
  %m = bufferization.to_buffer %t : tensor<1xi32> to memref<1xi32>
  %U = memref.cast %m : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

// CHECK_INVERSE: data =
// CHECK_INVERSE-NEXT: [1]

// Test mont_inverse: inv(a) * a == 1 in the Montgomery domain.
func.func @test_bls12381_mont_inverse() {
  %a = mod_arith.constant 3723 : !Fp
  %a_mont = mod_arith.to_mont %a : !Fpm
  %inv_mont = mod_arith.mont_inverse %a_mont : !Fpm
  %check_mont = mod_arith.mul %inv_mont, %a_mont : !Fpm
  %check = mod_arith.from_mont %check_mont : !Fp
  %v = mod_arith.bitcast %check : !Fp -> i384
  %v32 = arith.trunci %v : i384 to i32
  %t = tensor.from_elements %v32 : tensor<1xi32>
  %m = bufferization.to_buffer %t : tensor<1xi32> to memref<1xi32>
  %U = memref.cast %m : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

// CHECK_MONT_INVERSE: data =
// CHECK_MONT_INVERSE-NEXT: [1]
