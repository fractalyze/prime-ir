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

// RUN: prime-ir-opt -field-vectorize -split-input-file %s | FileCheck %s
// RUN: prime-ir-opt -field-vectorize="vector-width=8" -split-input-file %s | FileCheck %s --check-prefix=WIDTH8

!pf = !field.pf<2013265921 : i32, true>
!pf_std = !field.pf<2013265921 : i32>

// CHECK-LABEL: @test_basic_add
// CHECK-SAME: (%[[ARG0:.*]]: vector<16x!{{.*}}>, %[[ARG1:.*]]: vector<16x!{{.*}}>)
// CHECK-SAME: -> vector<16x!{{.*}}>
// WIDTH8-LABEL: @test_basic_add
// WIDTH8-SAME: (%[[ARG0:.*]]: vector<8x!{{.*}}>, %[[ARG1:.*]]: vector<8x!{{.*}}>)
func.func @test_basic_add(%a: !pf, %b: !pf) -> !pf {
  // CHECK: field.add %[[ARG0]], %[[ARG1]] : vector<16x!{{.*}}>
  %result = field.add %a, %b : !pf
  return %result : !pf
}

// -----

!pf = !field.pf<2013265921 : i32, true>

// CHECK-LABEL: @test_sub_mul
// CHECK-SAME: (%[[ARG0:.*]]: vector<16x!{{.*}}>, %[[ARG1:.*]]: vector<16x!{{.*}}>)
func.func @test_sub_mul(%a: !pf, %b: !pf) -> !pf {
  // CHECK: %[[SUB:.*]] = field.sub %[[ARG0]], %[[ARG1]] : vector<16x!{{.*}}>
  %sub = field.sub %a, %b : !pf
  // CHECK: %[[MUL:.*]] = field.mul %[[SUB]], %[[ARG1]] : vector<16x!{{.*}}>
  %mul = field.mul %sub, %b : !pf
  return %mul : !pf
}

// -----

!pf = !field.pf<2013265921 : i32, true>

// CHECK-LABEL: @test_unary_ops
// CHECK-SAME: (%[[ARG:.*]]: vector<16x!{{.*}}>)
func.func @test_unary_ops(%a: !pf) -> !pf {
  // CHECK: %[[DBL:.*]] = field.double %[[ARG]] : vector<16x!{{.*}}>
  %dbl = field.double %a : !pf
  // CHECK: %[[SQ:.*]] = field.square %[[DBL]] : vector<16x!{{.*}}>
  %sq = field.square %dbl : !pf
  // CHECK: %[[NEG:.*]] = field.negate %[[SQ]] : vector<16x!{{.*}}>
  %neg = field.negate %sq : !pf
  return %neg : !pf
}

// -----

!pf = !field.pf<2013265921 : i32, true>

// CHECK-LABEL: @test_inverse
// CHECK-SAME: (%[[ARG:.*]]: vector<16x!{{.*}}>)
func.func @test_inverse(%a: !pf) -> !pf {
  // CHECK: field.inverse %[[ARG]] : vector<16x!{{.*}}>
  %inv = field.inverse %a : !pf
  return %inv : !pf
}

// -----

!pf = !field.pf<2013265921 : i32, true>

// CHECK-LABEL: @test_powui
// CHECK-SAME: (%[[ARG:.*]]: vector<16x!{{.*}}>)
func.func @test_powui(%a: !pf) -> !pf {
  %c7 = arith.constant 7 : i32
  // CHECK: field.powui %[[ARG]], %{{.*}} : vector<16x!{{.*}}>, i32
  %pow = field.powui %a, %c7 : !pf, i32
  return %pow : !pf
}

// -----

!pf = !field.pf<2013265921 : i32, true>
!state = memref<16x!pf>

// CHECK-LABEL: @test_memref_conversion
// CHECK-SAME: (%[[STATE:.*]]: memref<16xvector<16x!{{.*}}>>)
func.func @test_memref_conversion(%state: !state) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[LOAD:.*]] = memref.load %[[STATE]][%{{.*}}] : memref<16xvector<16x!{{.*}}>>
  %val = memref.load %state[%c0] : !state
  // CHECK: %[[DBL:.*]] = field.double %[[LOAD]] : vector<16x!{{.*}}>
  %dbl = field.double %val : !pf
  // CHECK: memref.store %[[DBL]], %[[STATE]][%{{.*}}] : memref<16xvector<16x!{{.*}}>>
  memref.store %dbl, %state[%c0] : !state
  return
}

// -----

!pf = !field.pf<2013265921 : i32, true>

// CHECK-LABEL: @test_constant_splat
func.func @test_constant_splat() -> !pf {
  // CHECK: arith.constant dense<42> : vector<16xi32>
  // CHECK: field.bitcast {{.*}} : vector<16xi32> -> vector<16x!{{.*}}>
  %c = field.constant 42 : !pf
  return %c : !pf
}

// -----

!pf = !field.pf<2013265921 : i32, true>
!pf_std = !field.pf<2013265921 : i32>

// CHECK-LABEL: @test_to_mont
// CHECK-SAME: (%[[ARG:.*]]: vector<16x!{{.*}}>)
func.func @test_to_mont(%a: !pf_std) -> !pf {
  // CHECK: field.to_mont %[[ARG]] : vector<16x!{{.*}}>
  %mont = field.to_mont %a : !pf
  return %mont : !pf
}

// -----

!pf = !field.pf<2013265921 : i32, true>
!state = memref<16x!pf>

// CHECK-LABEL: @test_affine_ops
// CHECK-SAME: (%[[STATE:.*]]: memref<16xvector<16x!{{.*}}>>)
func.func @test_affine_ops(%state: !state) {
  // CHECK: affine.for %[[I:.*]] = 0 to 16 {
  affine.for %i = 0 to 16 {
    // CHECK: %[[LOAD:.*]] = affine.load %[[STATE]][%[[I]]] : memref<16xvector<16x!{{.*}}>>
    %val = affine.load %state[%i] : !state
    // CHECK: %[[SQ:.*]] = field.square %[[LOAD]] : vector<16x!{{.*}}>
    %sq = field.square %val : !pf
    // CHECK: affine.store %[[SQ]], %[[STATE]][%[[I]]] : memref<16xvector<16x!{{.*}}>>
    affine.store %sq, %state[%i] : !state
  }
  return
}
