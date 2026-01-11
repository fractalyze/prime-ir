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

// RUN: prime-ir-opt -mod-arith-to-arith -sccp -split-input-file %s | FileCheck %s -enable-var-scope

!Zp = !mod_arith.int<21888242871839275222246405745257275088696311157297823662689037894645226208583 : i256>

// CHECK-LABEL: @test_bn254_add
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_bn254_add() -> !Zp {
  %0 = mod_arith.constant 21888242871839275222246405745257275088696311157297823662689037894645226208580 :!Zp // -3
  %1 = mod_arith.constant 1 : !Zp
  %2 = mod_arith.constant 3 : !Zp
  %3 = mod_arith.constant 5 : !Zp

  %i0 = mod_arith.add %0, %1 : !Zp // -2
  %i1 = mod_arith.add %0, %2 : !Zp // 0
  %i2 = mod_arith.add %2, %2 : !Zp // 6
  %i3 = mod_arith.add %0, %3 : !Zp // 2

  %res0 = mod_arith.add %i0, %i1 : !Zp // -2
  %res1 = mod_arith.add %res0, %i2 : !Zp // 4
  %res2 = mod_arith.add %res1, %i3 : !Zp // 6

  // CHECK: %[[RES:.*]] = arith.constant 6 : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %res2 : !Zp // 6
}

// CHECK-LABEL: @test_bn254_sub
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_bn254_sub() -> !Zp {
  %0 = mod_arith.constant 2 : !Zp
  %1 = mod_arith.constant 2 : !Zp
  %2 = mod_arith.constant 4 : !Zp
  %3 = mod_arith.constant 21888242871839275222246405745257275088696311157297823662689037894645226208581 :!Zp // -2

  %i0 = mod_arith.sub %0, %0 : !Zp // 0
  %i1 = mod_arith.sub %0, %1 : !Zp // 0
  %i2 = mod_arith.sub %0, %2 : !Zp // -2

  %res0 = mod_arith.sub %i0, %i1 : !Zp // 0
  %res1 = mod_arith.sub %res0, %i2 : !Zp // 2
  %res2 = mod_arith.sub %res1, %3 : !Zp // 4

  // CHECK: %[[RES:.*]] = arith.constant 4 : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %res2 : !Zp
}

// CHECK-LABEL: @test_bn254_mult
// @ashjeong: this works, but changing to "-2" gives different direct result
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_bn254_mult() -> !Zp {
  %0 = mod_arith.constant 2 : !Zp
  %1 = mod_arith.constant 0 : !Zp
  %2 = mod_arith.constant 1 : !Zp
  %3 = mod_arith.constant 21888242871839275222246405745257275088696311157297823662689037894645226208581 :!Zp // -2

  %i0 = mod_arith.mul %0, %0 : !Zp // 4
  %i1 = mod_arith.mul %1, %1 : !Zp // 0
  %i2 = mod_arith.mul %1, %2 : !Zp // 0
  %i3 = mod_arith.mul %2, %3 : !Zp // -2
  %i4 = mod_arith.mul %3, %3 : !Zp // 4

  %res0 = mod_arith.add %i0, %i1 : !Zp // 4
  %res1 = mod_arith.add %i2, %i3 : !Zp // -2
  %res2 = mod_arith.add %res0, %i4 : !Zp // 8
  %res3 = mod_arith.add %res1, %res2 : !Zp // 6

  // CHECK: %[[RES:.*]] = arith.constant 6 : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %res3 : !Zp
}
