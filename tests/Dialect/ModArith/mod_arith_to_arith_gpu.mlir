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

// RUN: prime-ir-opt -mod-arith-to-arith=target=gpu -split-input-file %s | FileCheck %s -enable-var-scope --check-prefix=GPU
// The default (no target) RUN proves the flag actually gates: the same
// spare-bit BLS377 case keeps the wide-int path.
// RUN: prime-ir-opt -mod-arith-to-arith -split-input-file %s | FileCheck %s -enable-var-scope --check-prefix=CPU

!BLS377m = !mod_arith.int<258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177 : i384, true>

// GPU-LABEL: @gpu_mont_mul_multi_limb
// 32-bit-limb CIOS: limb products are (i64)a32*b32 accumulations; the wide
// product/reduce of the cpu path must be gone.
// GPU: arith.muli {{.*}} : i64
// GPU-NOT: arith.mului_extended {{.*}} : i384
// GPU: return {{.*}} : i384
// CPU-LABEL: @gpu_mont_mul_multi_limb
// CPU: arith.mului_extended {{.*}} : i384
func.func @gpu_mont_mul_multi_limb(%a : !BLS377m, %b : !BLS377m) -> !BLS377m {
  %r = mod_arith.mont_mul %a, %b : !BLS377m
  return %r : !BLS377m
}

// -----

// secp256k1 base field (p > 2^255, sign bit set): CIOS gate rejects sign-bit-set
// moduli, so even under target=gpu this falls back to the wide-int path.
!Secpm = !mod_arith.int<115792089237316195423570985008687907853269984665640564039457584007908834671663 : i256, true>

// GPU-LABEL: @gpu_mont_mul_sign_bit_set
// GPU: arith.mului_extended {{.*}} : i256
func.func @gpu_mont_mul_sign_bit_set(%a : !Secpm, %b : !Secpm) -> !Secpm {
  %r = mod_arith.mont_mul %a, %b : !Secpm
  return %r : !Secpm
}

// -----

!BLS377m = !mod_arith.int<258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177 : i384, true>

// mont_square routes through emitMontMul(input, input): 32-bit-limb CIOS, so
// the wide product/reduce must be gone.
// GPU-LABEL: @gpu_mont_square_multi_limb
// GPU: arith.muli {{.*}} : i64
// GPU-NOT: arith.mului_extended {{.*}} : i384
// GPU: return {{.*}} : i384
// CPU-LABEL: @gpu_mont_square_multi_limb
// CPU: arith.mului_extended {{.*}} : i384
func.func @gpu_mont_square_multi_limb(%a : !BLS377m) -> !BLS377m {
  %r = mod_arith.mont_square %a : !BLS377m
  return %r : !BLS377m
}

// -----

!BLS377m = !mod_arith.int<258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177 : i384, true>
!BLS377 = !mod_arith.int<258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177 : i384>

// mont_reduce routes through emitRedc: 32-bit-limb CIOS REDC, so the wide
// mont reducer's product/reduce must be gone.
// GPU-LABEL: @gpu_mont_reduce_multi_limb
// GPU: arith.muli {{.*}} : i64
// GPU-NOT: arith.mului_extended {{.*}} : i384
// GPU: return {{.*}} : i384
// CPU-LABEL: @gpu_mont_reduce_multi_limb
// CPU: arith.mului_extended {{.*}} : i384
func.func @gpu_mont_reduce_multi_limb(%lo : i384, %hi : i384) -> !BLS377 {
  %r = mod_arith.mont_reduce %lo, %hi : i384 -> !BLS377
  return %r : !BLS377
}

// -----

!BLS377m = !mod_arith.int<258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177 : i384, true>
!BLS377 = !mod_arith.int<258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177 : i384>

// from_mont lowers to mont_reduce(input, 0), inheriting the CIOS REDC path.
// GPU-LABEL: @gpu_from_mont_multi_limb
// GPU: arith.muli {{.*}} : i64
// GPU-NOT: arith.mului_extended {{.*}} : i384
// GPU: return {{.*}} : i384
// CPU-LABEL: @gpu_from_mont_multi_limb
// CPU: arith.mului_extended {{.*}} : i384
func.func @gpu_from_mont_multi_limb(%a : !BLS377m) -> !BLS377 {
  %r = mod_arith.from_mont %a : !BLS377
  return %r : !BLS377
}

// -----

!BLS377m = !mod_arith.int<258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177 : i384, true>
!BLS377 = !mod_arith.int<258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177 : i384>

// to_mont lowers to mont_mul(input, R²), inheriting the CIOS path.
// GPU-LABEL: @gpu_to_mont_multi_limb
// GPU: arith.muli {{.*}} : i64
// GPU-NOT: arith.mului_extended {{.*}} : i384
// GPU: return {{.*}} : i384
// CPU-LABEL: @gpu_to_mont_multi_limb
// CPU: arith.mului_extended {{.*}} : i384
func.func @gpu_to_mont_multi_limb(%a : !BLS377) -> !BLS377m {
  %r = mod_arith.to_mont %a : !BLS377m
  return %r : !BLS377m
}

// -----

!BLS377m = !mod_arith.int<258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177 : i384, true>

// square lowers to mont_square (Montgomery type), inheriting the CIOS path.
// GPU-LABEL: @gpu_square_multi_limb
// GPU: arith.muli {{.*}} : i64
// GPU-NOT: arith.mului_extended {{.*}} : i384
// GPU: return {{.*}} : i384
// CPU-LABEL: @gpu_square_multi_limb
// CPU: arith.mului_extended {{.*}} : i384
func.func @gpu_square_multi_limb(%a : !BLS377m) -> !BLS377m {
  %r = mod_arith.square %a : !BLS377m
  return %r : !BLS377m
}

// -----

// BabyBear (p = 2013265921, i32, Montgomery): single-limb type never routes
// through the 12-limb CIOS path — reduceSingleLimb is used unchanged under
// target=gpu. The lowering uses i32 extended multiplies, not the i64 limb
// accumulations of CIOS.
!BBm = !mod_arith.int<2013265921 : i32, true>

// GPU-LABEL: @gpu_mont_mul_single_limb_babybear
// Single-limb path: i32 extended mul present, no i64 CIOS limb products.
// GPU: arith.mului_extended {{.*}} : i32
// GPU-NOT: arith.muli {{.*}} : i64
func.func @gpu_mont_mul_single_limb_babybear(%a : !BBm, %b : !BBm) -> !BBm {
  %r = mod_arith.mont_mul %a, %b : !BBm
  return %r : !BBm
}
