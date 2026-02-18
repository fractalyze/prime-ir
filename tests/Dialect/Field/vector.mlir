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

// RUN: prime-ir-opt -field-to-mod-arith %s | FileCheck %s -enable-var-scope

!PF = !field.pf<7:i32, true>

// CHECK-LABEL: @test_vector_splat
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]]) -> [[VEC:.*]] {
func.func @test_vector_splat(%input : !PF) -> vector<4x!PF> {
  // CHECK: %[[SPLAT:.*]] = vector.splat %[[INPUT]] : [[VEC]]
  %splat = vector.splat %input : vector<4x!PF>
  // CHECK: return %[[SPLAT]] : [[VEC]]
  return %splat : vector<4x!PF>
}

// CHECK-LABEL: @test_vector_insert
// CHECK-SAME: (%[[VAL:.*]]: [[ELEM:.*]], %[[VEC:.*]]: [[VEC_TYPE:.*]]) -> [[VEC_TYPE]] {
func.func @test_vector_insert(%val : !PF, %vec : vector<4x!PF>) -> vector<4x!PF> {
  // CHECK: %[[RES:.*]] = vector.insert %[[VAL]], %[[VEC]] [2] : [[ELEM]] into [[VEC_TYPE]]
  %res = vector.insert %val, %vec [2] : !PF into vector<4x!PF>
  // CHECK: return %[[RES]] : [[VEC_TYPE]]
  return %res : vector<4x!PF>
}

// CHECK-LABEL: @test_vector_extract
// CHECK-SAME: (%[[VEC:.*]]: [[VEC_TYPE:.*]]) -> [[ELEM:.*]] {
func.func @test_vector_extract(%vec : vector<4x!PF>) -> !PF {
  // CHECK: %[[RES:.*]] = vector.extract %[[VEC]][1] : [[ELEM]] from [[VEC_TYPE]]
  %res = vector.extract %vec[1] : !PF from vector<4x!PF>
  // CHECK: return %[[RES]] : [[ELEM]]
  return %res : !PF
}
