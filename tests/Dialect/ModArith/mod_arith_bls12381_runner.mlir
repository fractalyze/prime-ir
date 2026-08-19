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
// Bernstein-Yang inverter codegen path for a modulus where divsteps > 62, so
// the limb type produced by BYAttr::getLimbBitWidth() is wider than 64 bits.
// Catches the bug where modular inverse computed via the lowered MLIR code
// disagreed with zk_dtypes::bls12_381::FqMont::Inverse() for any non-trivial
// denominator (because the limb type had no signed headroom for `md`).

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
// Compares the full i384 result against 1 (not just its low 32 bits) so a
// mismatch in any limb fails the test.
func.func @test_bls12381_inverse() {
  %a = mod_arith.constant 3723 : !Fp
  %inv = mod_arith.inverse %a : !Fp
  %check = mod_arith.mul %inv, %a : !Fp
  %v = mod_arith.bitcast %check : !Fp -> i384
  %one = arith.constant 1 : i384
  %ok = arith.cmpi eq, %v, %one : i384
  %ok32 = arith.extui %ok : i1 to i32
  %t = tensor.from_elements %ok32 : tensor<1xi32>
  %m = bufferization.to_buffer %t : tensor<1xi32> to memref<1xi32>
  %U = memref.cast %m : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

// CHECK_INVERSE: data =
// CHECK_INVERSE-NEXT: [1]

// Test mont_inverse: inv(a) * a == 1 in the Montgomery domain. Same full
// i384 comparison as above.
func.func @test_bls12381_mont_inverse() {
  %a = mod_arith.constant 3723 : !Fp
  %a_mont = mod_arith.to_mont %a : !Fpm
  %inv_mont = mod_arith.mont_inverse %a_mont : !Fpm
  %check_mont = mod_arith.mul %inv_mont, %a_mont : !Fpm
  %check = mod_arith.from_mont %check_mont : !Fp
  %v = mod_arith.bitcast %check : !Fp -> i384
  %one = arith.constant 1 : i384
  %ok = arith.cmpi eq, %v, %one : i384
  %ok32 = arith.extui %ok : i1 to i32
  %t = tensor.from_elements %ok32 : tensor<1xi32>
  %m = bufferization.to_buffer %t : tensor<1xi32> to memref<1xi32>
  %U = memref.cast %m : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

// CHECK_MONT_INVERSE: data =
// CHECK_MONT_INVERSE-NEXT: [1]

// RUN: prime-ir-opt %s -mod-arith-to-arith -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_bls12381_fr_mont_mul_small -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_FR_MUL_SMALL < %t

// RUN: prime-ir-opt %s -mod-arith-to-arith -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_bls12381_fr_mont_mul_wide -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_FR_MUL_WIDE < %t

// BLS12-381 scalar field:
// r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
// Bit 64 of r is CLEAR (limb1 = 0x53bda402fffe5bfe is even), which used to
// corrupt every Montgomery constant derived from bReduced = 2⁶⁴ mod r
// (MontgomeryAttrStorage::construct truncated the modulus to 65 bits before
// the urem), so even 3 · 5 came out wrong. r also has b⁻¹ mod r ≈ r, the
// worst case for the former shift-then-add b⁻¹ REDC variant whose result
// bound V < T/2ʷ + b⁻¹ + r exceeded 2ʷ for ~1% of random products.
!Fr = !mod_arith.int<52435875175126190479447740508185965837690552500527637822603658699938581184513 : i256>
!Frm = !mod_arith.int<52435875175126190479447740508185965837690552500527637822603658699938581184513 : i256, true>

// Small values through to_mont → mont_mul → from_mont. Catches the corrupted
// R² / b⁻¹ constants (bit-64-clear modulus): with them even this is garbage.
func.func @test_bls12381_fr_mont_mul_small() {
  %a = mod_arith.constant 3 : !Fr
  %b = mod_arith.constant 5 : !Fr
  %a_mont = mod_arith.to_mont %a : !Frm
  %b_mont = mod_arith.to_mont %b : !Frm
  %ab_mont = mod_arith.mont_mul %a_mont, %b_mont : !Frm
  %ab = mod_arith.from_mont %ab_mont : !Fr

  %v = mod_arith.bitcast %ab : !Fr -> i256
  %vec = vector.from_elements %v : vector<1xi256>
  %i32vec = vector.bitcast %vec : vector<1xi256> to vector<8xi32>
  %mem = memref.alloc() : memref<8xi32>
  %c0 = arith.constant 0 : index
  vector.store %i32vec, %mem[%c0] : memref<8xi32>, vector<8xi32>
  %U = memref.cast %mem : memref<8xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

// 3 * 5 mod r = 15
// CHECK_FR_MUL_SMALL: [15, 0, 0, 0, 0, 0, 0, 0]

// A random product whose REDC pre-canonical value exceeds 2²⁵⁶ under the
// former b⁻¹ REDC variant even with correct constants (found by search; the
// classical per-limb REDC keeps V < 2r < 2²⁵⁶ so the fixed lowering is
// immune by construction).
func.func @test_bls12381_fr_mont_mul_wide() {
  %a = mod_arith.constant 18623444530724716917715066816556456062935249299391794592104980440890991110953 : !Fr
  %b = mod_arith.constant 5351036188861986030282903639234331999548142798982450849968728322547002645315 : !Fr
  %a_mont = mod_arith.to_mont %a : !Frm
  %b_mont = mod_arith.to_mont %b : !Frm
  %ab_mont = mod_arith.mont_mul %a_mont, %b_mont : !Frm
  %ab = mod_arith.from_mont %ab_mont : !Fr

  %v = mod_arith.bitcast %ab : !Fr -> i256
  %vec = vector.from_elements %v : vector<1xi256>
  %i32vec = vector.bitcast %vec : vector<1xi256> to vector<8xi32>
  %mem = memref.alloc() : memref<8xi32>
  %c0 = arith.constant 0 : index
  vector.store %i32vec, %mem[%c0] : memref<8xi32>, vector<8xi32>
  %U = memref.cast %mem : memref<8xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

// a * b mod r =
// 50706946945572298874017667503635483751308595243764330383439609392027904712325
// CHECK_FR_MUL_WIDE: [-535617915, 1299169884, -586082370, -1308919744, 124195241, -1719860803, -296459002, 1880825194]
