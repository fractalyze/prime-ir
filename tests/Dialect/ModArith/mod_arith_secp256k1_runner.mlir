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

// Runner tests for secp256k1 base field (256-bit modulus with NO spare bit).
// secp256k1: p = 2²⁵⁶ - 2³² - 977, so p > 2²⁵⁵ and 2p > 2²⁵⁶.
// This exercises the REDC overflow path in reduceMultiLimb.

// RUN: prime-ir-opt %s -mod-arith-to-arith -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_secp256k1_mont_mul -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_MONT_MUL < %t

// RUN: prime-ir-opt %s -mod-arith-to-arith -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_secp256k1_mont_square -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_MONT_SQUARE < %t

// RUN: prime-ir-opt %s -mod-arith-to-arith -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_secp256k1_inverse -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_INVERSE < %t

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

// secp256k1 base field: p = 115792089237316195423570985008687907853269984665640564039457584007908834671663
!Fp = !mod_arith.int<115792089237316195423570985008687907853269984665640564039457584007908834671663 : i256>
!Fpm = !mod_arith.int<115792089237316195423570985008687907853269984665640564039457584007908834671663 : i256, true>

// Test mont_mul: a * b mod p
// a = 0xDEADBEEFCAFEBABE1234567890ABCDEF0123456789ABCDEF0FEDCBA987654321
// b = 0xAAAABBBBCCCCDDDD1111222233334444555566667777888899990000AAAA1111
func.func @test_secp256k1_mont_mul() {
  %a = mod_arith.constant 100720434724302814904028779430100746620855462830062608636120953741433830196001 : !Fp
  %b = mod_arith.constant 77194843949812472705866951895890002278064767165192463917288049357869258772753 : !Fp
  %a_mont = mod_arith.to_mont %a : !Fpm
  %b_mont = mod_arith.to_mont %b : !Fpm
  %ab_mont = mod_arith.mont_mul %a_mont, %b_mont : !Fpm
  %ab = mod_arith.from_mont %ab_mont : !Fp

  %v = mod_arith.bitcast %ab : !Fp -> i256
  %vec = vector.from_elements %v : vector<1xi256>
  %i32vec = vector.bitcast %vec : vector<1xi256> to vector<8xi32>
  %mem = memref.alloc() : memref<8xi32>
  %c0 = arith.constant 0 : index
  vector.store %i32vec, %mem[%c0] : memref<8xi32>, vector<8xi32>
  %U = memref.cast %mem : memref<8xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

// a*b mod p = 60982711392264587969682996417164316263579529872982963815222662697680270146463
// CHECK_MONT_MUL: [-317572193, 234547340, -933347341, 937312186, 925532681, -193108322, -782384468, -2032992815]

// Test mont_square: a² mod p
func.func @test_secp256k1_mont_square() {
  %a = mod_arith.constant 100720434724302814904028779430100746620855462830062608636120953741433830196001 : !Fp
  %a_mont = mod_arith.to_mont %a : !Fpm
  %a2_mont = mod_arith.mont_square %a_mont : !Fpm
  %a2 = mod_arith.from_mont %a2_mont : !Fp

  %v = mod_arith.bitcast %a2 : !Fp -> i256
  %vec = vector.from_elements %v : vector<1xi256>
  %i32vec = vector.bitcast %vec : vector<1xi256> to vector<8xi32>
  %mem = memref.alloc() : memref<8xi32>
  %c0 = arith.constant 0 : index
  vector.store %i32vec, %mem[%c0] : memref<8xi32>, vector<8xi32>
  %U = memref.cast %mem : memref<8xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

// a² mod p = 31420896780135511602482624633964787057105052135029862436193230832710629143834
// CHECK_MONT_SQUARE: [410727706, -880396485, -539256400, -604627336, 995880535, 1108136652, -129139622, 1165465835]

// Test inverse: inv(a) * a == 1
func.func @test_secp256k1_inverse() {
  %a = mod_arith.constant 100720434724302814904028779430100746620855462830062608636120953741433830196001 : !Fp
  %a_mont = mod_arith.to_mont %a : !Fpm
  %inv_mont = mod_arith.mont_inverse %a_mont : !Fpm
  %check_mont = mod_arith.mul %inv_mont, %a_mont : !Fpm
  %check = mod_arith.from_mont %check_mont : !Fp
  %v = mod_arith.bitcast %check : !Fp -> i256
  %v32 = arith.trunci %v : i256 to i32
  %t = tensor.from_elements %v32 : tensor<1xi32>
  %m = bufferization.to_buffer %t : tensor<1xi32> to memref<1xi32>
  %U = memref.cast %m : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

// inv(a) * a mod p == 1
// CHECK_INVERSE: [1]
