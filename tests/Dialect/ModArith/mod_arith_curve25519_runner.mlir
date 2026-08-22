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

// Runner test for the curve25519 base field, p = 2²⁵⁵ - 19. This is the
// tightest narrow-modulus witness for the multi-limb REDC result bound:
//
//   2p = 2²⁵⁶ - 38
//
// so the "V < 2n < 2ʷ" headroom the narrow-modulus tail of
// MontReducer::reduceMultiLimb relies on is only 38. The tail folds tHigh
// into tLow with `tHigh << 192` in a w-bit register, which silently drops
// any bit of V at or above 2²⁵⁶ — so the moment V crosses 2²⁵⁶ the result
// loses exactly 2²⁵⁶.
//
// Unlike the BLS12-381 scalar field, bit 64 of p is SET, so the truncated
// bReduced never corrupted this modulus' Montgomery constants. That makes
// this a witness for the REDC bound alone.
//
// The former shift-then-add b⁻¹ REDC variant had b⁻¹ ≈ 0.53p here, and its
// bound V < T/2ʷ + b⁻¹ + p crosses 2²⁵⁶ for roughly 1 random square in
// 10⁵. The error is invisible at a glance because the lost 2²⁵⁶ ≡ R (mod p):
// an error of exactly R in the Montgomery domain is an error of exactly
// R·R⁻¹ = 1 once from_mont converts out, so the result is always low by
// exactly 1 and never by anything else.
//
// A rate that low is undetectable by fixed-vector testing — this operand was
// found from the consumer side, where an X25519 ladder (~2800 multiplies per
// call) over the dtype passed RFC 7748 §5.2's fixed vectors and six random
// ones, then diverged from the reference at iteration 82 of the iterated
// test. See fractalyze/xla#542.

// RUN: prime-ir-opt %s -mod-arith-to-arith -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_curve25519_mont_square_wrap -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_WRAP < %t

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

// curve25519 base field: p = 2²⁵⁵ - 19
!Fp = !mod_arith.int<57896044618658097711785492504343953926634992332820282019728792003956564819949 : i256>
!Fpm = !mod_arith.int<57896044618658097711785492504343953926634992332820282019728792003956564819949 : i256, true>

// Square an operand whose REDC pre-canonical value crosses 2²⁵⁶ under the
// former b⁻¹ variant. Returned a result low by exactly 1 before the fix; the
// classical per-limb REDC keeps V < 2p < 2²⁵⁶, so the tail's tHigh == 0
// assumption holds and the lowering is immune by construction.
func.func @test_curve25519_mont_square_wrap() {
  %a = mod_arith.constant 16745370599590454985819630277347750976000815978402662724786347922478530828282 : !Fp
  %a_mont = mod_arith.to_mont %a : !Fpm
  %aa_mont = mod_arith.mont_mul %a_mont, %a_mont : !Fpm
  %aa = mod_arith.from_mont %aa_mont : !Fp

  %v = mod_arith.bitcast %aa : !Fp -> i256
  %vec = vector.from_elements %v : vector<1xi256>
  %i32vec = vector.bitcast %vec : vector<1xi256> to vector<8xi32>
  %mem = memref.alloc() : memref<8xi32>
  %c0 = arith.constant 0 : index
  vector.store %i32vec, %mem[%c0] : memref<8xi32>, vector<8xi32>
  %U = memref.cast %mem : memref<8xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

// a² mod p =
// 6472717142822227947744657607057095016600357453940632462193194648066349579
// The pre-fix result differed only in the low word: -403154422.
// CHECK_WRAP: [-403154421, -999763932, 1167814559, -2006176652, 995992490, -352753903, 1814100388, 240086]
