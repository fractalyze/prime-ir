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

// RUN: prime-ir-opt -mod-arith-to-arith %s | FileCheck %s

// Test: IntrReduceOp and IntrMontReduceOp are preserved by mod-arith-to-arith pass
// This enables the two-stage lowering pipeline with optimization in between

// CHECK-LABEL: @test_preserve_intr_reduce
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_preserve_intr_reduce(%input: i32) -> i32 {
  // CHECK: mod_arith.intr.reduce %[[INPUT]]
  // CHECK-SAME: input_range = #llvm.constant_range<i32, 0, 131074>
  // CHECK-SAME: modulus = 65537
  %result = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 131074>
  } : i32 -> i32
  return %result : i32
}

// CHECK-LABEL: @test_preserve_intr_reduce_negative_range
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_preserve_intr_reduce_negative_range(%input: i32) -> i32 {
  // CHECK: mod_arith.intr.reduce %[[INPUT]]
  // CHECK-SAME: input_range = #llvm.constant_range<i32, -65536, 65537>
  // CHECK-SAME: modulus = 65537
  %result = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, -65536, 65537>
  } : i32 -> i32
  return %result : i32
}

// CHECK-LABEL: @test_preserve_intr_mont_reduce
// CHECK-SAME: (%[[LOW:.*]]: [[T:.*]], %[[HIGH:.*]]: [[T]]) -> [[T]]
func.func @test_preserve_intr_mont_reduce(%low: i32, %high: i32) -> i32 {
  // CHECK: mod_arith.intr.mont_reduce %[[LOW]], %[[HIGH]]
  // CHECK-SAME: modulus = 65537
  // CHECK-SAME: n_inv = -65535
  %result = mod_arith.intr.mont_reduce %low, %high {
    modulus = 65537 : i32,
    n_inv = -65535 : i32
  } : (i32, i32) -> i32
  return %result : i32
}

// CHECK-LABEL: @test_preserve_intr_reduce_vector
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_preserve_intr_reduce_vector(%input: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: mod_arith.intr.reduce %[[INPUT]]
  %result = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 131074>
  } : tensor<4xi32> -> tensor<4xi32>
  return %result : tensor<4xi32>
}
