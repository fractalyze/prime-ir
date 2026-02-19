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

// RUN: prime-ir-opt -mod-arith-to-arith %s | FileCheck %s -enable-var-scope

!mod = !mod_arith.int<7:i32, true>

// CHECK-LABEL: @test_vector_splat
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[VEC:.*]] {
func.func @test_vector_splat(%input : !mod) -> vector<4x!mod> {
  // CHECK: %[[SPLAT:.*]] = vector.splat %[[INPUT]] : [[VEC]]
  %splat = vector.splat %input : vector<4x!mod>
  // CHECK: return %[[SPLAT]] : [[VEC]]
  return %splat : vector<4x!mod>
}

// CHECK-LABEL: @test_vector_insert
// CHECK-SAME: (%[[VAL:.*]]: [[ELEM:.*]], %[[VEC:.*]]: [[VEC_TYPE:.*]]) -> [[VEC_TYPE]] {
func.func @test_vector_insert(%val : !mod, %vec : vector<4x!mod>) -> vector<4x!mod> {
  // CHECK: %[[RES:.*]] = vector.insert %[[VAL]], %[[VEC]] [2] : [[ELEM]] into [[VEC_TYPE]]
  %res = vector.insert %val, %vec [2] : !mod into vector<4x!mod>
  // CHECK: return %[[RES]] : [[VEC_TYPE]]
  return %res : vector<4x!mod>
}

// CHECK-LABEL: @test_vector_extract
// CHECK-SAME: (%[[VEC:.*]]: [[VEC_TYPE:.*]]) -> [[ELEM:.*]] {
func.func @test_vector_extract(%vec : vector<4x!mod>) -> !mod {
  // CHECK: %[[RES:.*]] = vector.extract %[[VEC]][1] : [[ELEM]] from [[VEC_TYPE]]
  %res = vector.extract %vec[1] : !mod from vector<4x!mod>
  // CHECK: return %[[RES]] : [[ELEM]]
  return %res : !mod
}
