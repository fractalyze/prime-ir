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

// Executable correctness cross-check for the BLS12-377 base field (384-bit,
// width%64==0, bit-64==0 — the class whose Montgomery bInv constant was wrong,
// issue #344). mont_mul operands are loaded from a memref so the op does NOT
// constant-fold via zk_dtypes before lowering; the CIOS/wide-int arith is
// actually emitted and executed. The same CHECK pins both the default
// (wide-int REDC) and the target=gpu (32-bit-limb CIOS) lowerings, so a green
// run simultaneously proves the bInv fix (default path correct), proves the
// CIOS lowering, and cross-checks the two against each other.
//
// Expected values computed in Python (R = 2^384):
//
//   p = 258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177
//   Rinv = pow(R, -1, p)
//   to_mont(x)   = x * R % p
//   from_mont(x) = x * Rinv % p
//   mont_mul(x,y) = x * y * Rinv % p
//   a = 999999999999999999999999999;  b = 888888888888888888888888888
//   Am = to_mont(a);  Bm = to_mont(b)
//   # mont_mul(Am, Bm) == a*b*R % p; from_mont(mont_mul(Am,Bm)) == a*b % p
//   def limbs(x): return [((x>>(32*i))&0xffffffff)-(2**32 if ((x>>(32*i))&0xffffffff)>=2**31 else 0) for i in range(12)]
//   Am limbs: [-993027604, -388498365, -1470266380, 258085974, 1271394216, -1006981957, -1774979521, 1779210631, 1700853052, 920988029, -85444264, 1398356]
//   Bm limbs: [-405472615, 857098982, -262987786, -1159288151, 1476924906, 1071475599, -621537805, 198586088, -1626159698, 1188184566, -31640358, 4375805]
//   a  limbs: [-402653185, -1613725636, 54210108, 0, 0, 0, 0, 0, 0, 0, 0, 0]
//   b  limbs: [2028179000, 951670155, 48186763, 0, 0, 0, 0, 0, 0, 0, 0, 0]
//   mont_mul(Am,Bm) limbs: [-2085231504, -944653634, 788851044, 435972004, -2052965250, 763951016, 195537406, 351631468, -670383268, 1389461931, -1947203231, 21569816]
//   a*b % p     limbs: [1193046472, -1631176585, -403608459, 1030493760, 1969132180, 608202, 0, 0, 0, 0, 0, 0]

// RUN: prime-ir-opt %s -mod-arith-to-arith -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_bls377_mont_mul -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_MONT_MUL < %t

// RUN: prime-ir-opt %s -mod-arith-to-arith=target=gpu -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_bls377_mont_mul -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_MONT_MUL < %t

// RUN: prime-ir-opt %s -mod-arith-to-arith -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_bls377_roundtrip -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_ROUNDTRIP < %t

// RUN: prime-ir-opt %s -mod-arith-to-arith=target=gpu -convert-elementwise-to-linalg -one-shot-bufferize -convert-linalg-to-parallel-loops -convert-scf-to-cf -convert-cf-to-llvm -convert-to-llvm -convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_bls377_roundtrip -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_ROUNDTRIP < %t

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

// BLS12-377 base field (spare bit set: p < 2^383, so 2p < 2^384).
!BLS377m = !mod_arith.int<258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177 : i384, true>
!BLS377 = !mod_arith.int<258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177 : i384>

// Store 12 i32 limbs and read them back as an opaque i384. Routing the value
// through memref load/store keeps it non-constant at fold time, so mont_mul
// (and to_mont/from_mont) lower to real arith instead of folding.
func.func private @load_i384(%limbs : memref<12xi32>) -> i384 {
  %c0 = arith.constant 0 : index
  %vec = vector.load %limbs[%c0] : memref<12xi32>, vector<12xi32>
  %wide = vector.bitcast %vec : vector<12xi32> to vector<1xi384>
  %v = vector.extract %wide[0] : i384 from vector<1xi384>
  return %v : i384
}

func.func private @print_i384(%v : i384) {
  %c0 = arith.constant 0 : index
  %vec = vector.from_elements %v : vector<1xi384>
  %i32vec = vector.bitcast %vec : vector<1xi384> to vector<12xi32>
  %mem = memref.alloc() : memref<12xi32>
  vector.store %i32vec, %mem[%c0] : memref<12xi32>, vector<12xi32>
  %U = memref.cast %mem : memref<12xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  memref.dealloc %mem : memref<12xi32>
  return
}

func.func private @store_limbs(%m : memref<12xi32>, %v0 : i32, %v1 : i32,
    %v2 : i32, %v3 : i32, %v4 : i32, %v5 : i32, %v6 : i32, %v7 : i32,
    %v8 : i32, %v9 : i32, %v10 : i32, %v11 : i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index
  memref.store %v0, %m[%c0] : memref<12xi32>
  memref.store %v1, %m[%c1] : memref<12xi32>
  memref.store %v2, %m[%c2] : memref<12xi32>
  memref.store %v3, %m[%c3] : memref<12xi32>
  memref.store %v4, %m[%c4] : memref<12xi32>
  memref.store %v5, %m[%c5] : memref<12xi32>
  memref.store %v6, %m[%c6] : memref<12xi32>
  memref.store %v7, %m[%c7] : memref<12xi32>
  memref.store %v8, %m[%c8] : memref<12xi32>
  memref.store %v9, %m[%c9] : memref<12xi32>
  memref.store %v10, %m[%c10] : memref<12xi32>
  memref.store %v11, %m[%c11] : memref<12xi32>
  return
}

// mont_mul(Am, Bm) with Am, Bm already in Montgomery form, fed as opaque i384
// (bitcast i384 -> !BLS377m reinterprets bits, matching what mod_arith.constant
// stores). Result = a*b*R % p in Montgomery form.
func.func @test_bls377_mont_mul() {
  %am_mem = memref.alloc() : memref<12xi32>
  %bm_mem = memref.alloc() : memref<12xi32>
  %am0 = arith.constant -993027604 : i32
  %am1 = arith.constant -388498365 : i32
  %am2 = arith.constant -1470266380 : i32
  %am3 = arith.constant 258085974 : i32
  %am4 = arith.constant 1271394216 : i32
  %am5 = arith.constant -1006981957 : i32
  %am6 = arith.constant -1774979521 : i32
  %am7 = arith.constant 1779210631 : i32
  %am8 = arith.constant 1700853052 : i32
  %am9 = arith.constant 920988029 : i32
  %am10 = arith.constant -85444264 : i32
  %am11 = arith.constant 1398356 : i32
  func.call @store_limbs(%am_mem, %am0, %am1, %am2, %am3, %am4, %am5, %am6,
    %am7, %am8, %am9, %am10, %am11)
    : (memref<12xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
  %bm0 = arith.constant -405472615 : i32
  %bm1 = arith.constant 857098982 : i32
  %bm2 = arith.constant -262987786 : i32
  %bm3 = arith.constant -1159288151 : i32
  %bm4 = arith.constant 1476924906 : i32
  %bm5 = arith.constant 1071475599 : i32
  %bm6 = arith.constant -621537805 : i32
  %bm7 = arith.constant 198586088 : i32
  %bm8 = arith.constant -1626159698 : i32
  %bm9 = arith.constant 1188184566 : i32
  %bm10 = arith.constant -31640358 : i32
  %bm11 = arith.constant 4375805 : i32
  func.call @store_limbs(%bm_mem, %bm0, %bm1, %bm2, %bm3, %bm4, %bm5, %bm6,
    %bm7, %bm8, %bm9, %bm10, %bm11)
    : (memref<12xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()

  %ai = func.call @load_i384(%am_mem) : (memref<12xi32>) -> i384
  %bi = func.call @load_i384(%bm_mem) : (memref<12xi32>) -> i384
  %am = mod_arith.bitcast %ai : i384 -> !BLS377m
  %bm = mod_arith.bitcast %bi : i384 -> !BLS377m
  %r = mod_arith.mont_mul %am, %bm : !BLS377m
  %v = mod_arith.bitcast %r : !BLS377m -> i384
  func.call @print_i384(%v) : (i384) -> ()
  memref.dealloc %am_mem : memref<12xi32>
  memref.dealloc %bm_mem : memref<12xi32>
  return
}

// CHECK_MONT_MUL: data =
// CHECK_MONT_MUL-NEXT: [-2085231504, -944653634, 788851044, 435972004, -2052965250, 763951016, 195537406, 351631468, -670383268, 1389461931, -1947203231, 21569816]

// from_mont(mont_mul(to_mont(a), to_mont(b))) with a, b plain residues fed as
// opaque i384. Exercises the full to_mont -> mont_mul -> from_mont chain on the
// runtime path; result = a*b % p in canonical (non-Montgomery) form.
func.func @test_bls377_roundtrip() {
  %a_mem = memref.alloc() : memref<12xi32>
  %b_mem = memref.alloc() : memref<12xi32>
  %z = arith.constant 0 : i32
  %a0 = arith.constant -402653185 : i32
  %a1 = arith.constant -1613725636 : i32
  %a2 = arith.constant 54210108 : i32
  func.call @store_limbs(%a_mem, %a0, %a1, %a2, %z, %z, %z, %z, %z, %z, %z, %z, %z)
    : (memref<12xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
  %b0 = arith.constant 2028179000 : i32
  %b1 = arith.constant 951670155 : i32
  %b2 = arith.constant 48186763 : i32
  func.call @store_limbs(%b_mem, %b0, %b1, %b2, %z, %z, %z, %z, %z, %z, %z, %z, %z)
    : (memref<12xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()

  %ai = func.call @load_i384(%a_mem) : (memref<12xi32>) -> i384
  %bi = func.call @load_i384(%b_mem) : (memref<12xi32>) -> i384
  %a = mod_arith.bitcast %ai : i384 -> !BLS377
  %b = mod_arith.bitcast %bi : i384 -> !BLS377
  %a_mont = mod_arith.to_mont %a : !BLS377m
  %b_mont = mod_arith.to_mont %b : !BLS377m
  %r_mont = mod_arith.mont_mul %a_mont, %b_mont : !BLS377m
  %r = mod_arith.from_mont %r_mont : !BLS377
  %v = mod_arith.bitcast %r : !BLS377 -> i384
  func.call @print_i384(%v) : (i384) -> ()
  memref.dealloc %a_mem : memref<12xi32>
  memref.dealloc %b_mem : memref<12xi32>
  return
}

// CHECK_ROUNDTRIP: data =
// CHECK_ROUNDTRIP-NEXT: [1193046472, -1631176585, -403608459, 1030493760, 1969132180, 608202, 0, 0, 0, 0, 0, 0]
