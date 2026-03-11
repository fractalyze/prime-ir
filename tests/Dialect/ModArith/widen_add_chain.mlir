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

// RUN: prime-ir-opt '-mod-arith-to-arith=min-chain-for-widen=3' -split-input-file %s | FileCheck %s -enable-var-scope
// RUN: prime-ir-opt '-mod-arith-to-arith=min-chain-for-widen=0' -split-input-file %s | FileCheck %s --check-prefix=NOWIDEN -enable-var-scope

// BabyBear: 31-bit modulus in i32 storage.
// modWidth=31, storageWidth=32. A 3-leaf chain needs ceil(log₂(3))=2 extra
// bits → neededWidth=33, widerWidth=bit_ceil(33)=64 > 32 → widening triggers.
!Zp = !mod_arith.int<2013265921 : i32>

// 3 leaves (2 adds): a + b + c
// With widening: extui to i64, two adds in i64, binary reduction, trunci.
// CHECK-LABEL: @test_widen_3_leaves
// CHECK-SAME: (%[[A:.*]]: i32, %[[B:.*]]: i32, %[[C:.*]]: i32) -> i32
// CHECK-DAG: %[[A_EXT:.*]] = arith.extui %[[A]] : i32 to i64
// CHECK-DAG: %[[B_EXT:.*]] = arith.extui %[[B]] : i32 to i64
// CHECK: %[[ADD0:.*]] = arith.addi %[[A_EXT]], %[[B_EXT]] : i64
// CHECK: %[[C_EXT:.*]] = arith.extui %[[C]] : i32 to i64
// CHECK: %[[ADD1:.*]] = arith.addi %[[ADD0]], %[[C_EXT]] : i64
//   Binary reduction: ceil(log₂(3)) = 2 steps → 2 minui ops.
// CHECK: arith.subi {{.*}} : i64
// CHECK: arith.minui {{.*}} : i64
// CHECK: arith.subi {{.*}} : i64
// CHECK: arith.minui {{.*}} : i64
// CHECK: arith.trunci {{.*}} : i64 to i32
// NOWIDEN-LABEL: @test_widen_3_leaves
// NOWIDEN-NOT: i64
func.func @test_widen_3_leaves(%a : !Zp, %b : !Zp, %c : !Zp) -> !Zp {
  %ab = mod_arith.add %a, %b : !Zp
  %abc = mod_arith.add %ab, %c : !Zp
  return %abc : !Zp
}

// -----

!Zp = !mod_arith.int<2013265921 : i32>

// 4 leaves (3 adds): a + b + c + d
// CHECK-LABEL: @test_widen_4_leaves
// CHECK-SAME: (%[[A:.*]]: i32, %[[B:.*]]: i32, %[[C:.*]]: i32, %[[D:.*]]: i32) -> i32
// CHECK: arith.extui {{.*}} : i32 to i64
// CHECK: arith.addi {{.*}} : i64
// CHECK: arith.extui {{.*}} : i32 to i64
// CHECK: arith.addi {{.*}} : i64
// CHECK: arith.extui {{.*}} : i32 to i64
// CHECK: arith.addi {{.*}} : i64
// CHECK: arith.trunci {{.*}} : i64 to i32
func.func @test_widen_4_leaves(%a : !Zp, %b : !Zp, %c : !Zp, %d : !Zp) -> !Zp {
  %ab = mod_arith.add %a, %b : !Zp
  %abc = mod_arith.add %ab, %c : !Zp
  %abcd = mod_arith.add %abc, %d : !Zp
  return %abcd : !Zp
}

// -----

!Zp = !mod_arith.int<2013265921 : i32>

// Single add (2 leaves): below threshold → no widening, fallback to per-add.
// BabyBear: storageWidth - modWidth = 1, so only nuw flag.
// CHECK-LABEL: @test_no_widen_single_add
// CHECK-SAME: (%[[A:.*]]: i32, %[[B:.*]]: i32) -> i32
// CHECK-NOT: i64
// CHECK: arith.addi %[[A]], %[[B]] overflow<nuw>
func.func @test_no_widen_single_add(%a : !Zp, %b : !Zp) -> !Zp {
  %res = mod_arith.add %a, %b : !Zp
  return %res : !Zp
}

// -----

!Zp = !mod_arith.int<2013265921 : i32>

// Chain break: intermediate result has multiple uses.
// (a + b) has two uses → it's a chain root with 2 leaves, not intermediate.
// (a + b) + c has 2 leaves (the root (a+b) and c) → no widening.
// CHECK-LABEL: @test_chain_break_multi_use
// CHECK-SAME: (%[[A:.*]]: i32, %[[B:.*]]: i32, %[[C:.*]]: i32)
// CHECK-NOT: i64
func.func @test_chain_break_multi_use(%a : !Zp, %b : !Zp, %c : !Zp) -> (!Zp, !Zp) {
  %ab = mod_arith.add %a, %b : !Zp
  %abc = mod_arith.add %ab, %c : !Zp
  return %ab, %abc : !Zp, !Zp
}

// -----

// Full-width prime: 32-bit prime (4294967291 = 2³² - 5) in i32 storage.
// modWidth=32, storageWidth=32. Even a single add can overflow.
// For 3 leaves: extraBits=2, neededWidth=34, widerWidth=64 > 32 → widening.
!Zp_full = !mod_arith.int<4294967291 : i32>

// CHECK-LABEL: @test_widen_full_width
// CHECK-SAME: (%[[A:.*]]: i32, %[[B:.*]]: i32, %[[C:.*]]: i32) -> i32
// CHECK: arith.extui {{.*}} : i32 to i64
// CHECK: arith.addi {{.*}} : i64
// CHECK: arith.trunci {{.*}} : i64 to i32
func.func @test_widen_full_width(%a : !Zp_full, %b : !Zp_full, %c : !Zp_full) -> !Zp_full {
  %ab = mod_arith.add %a, %b : !Zp_full
  %abc = mod_arith.add %ab, %c : !Zp_full
  return %abc : !Zp_full
}
