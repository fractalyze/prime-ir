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

// RUN: prime-ir-opt -intr-reduce-to-arith %s | FileCheck %s

// -----
// Test: Single-operand reduction with non-negative range [0, 2*mod)
// Should generate: if (input >= mod) input -= mod

// CHECK-LABEL: @test_reduce_extended
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_reduce_extended(%input: i32) -> i32 {
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[T]]
  // CHECK: %[[CMP:.*]] = arith.cmpi ult, %[[INPUT]], %[[CMOD]] : [[T]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[INPUT]], %[[CMOD]] : [[T]]
  // CHECK: %[[SELECT:.*]] = arith.select %[[CMP]], %[[INPUT]], %[[SUB]] : [[T]]
  // CHECK: return %[[SELECT]] : [[T]]
  %result = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 131074>
  } : i32 -> i32
  return %result : i32
}

// -----
// Test: Single-operand reduction with negative range [-mod+1, mod)
// Should generate: if (input < 0) input += mod

// CHECK-LABEL: @test_reduce_negative
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_reduce_negative(%input: i32) -> i32 {
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[T]]
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : [[T]]
  // CHECK: %[[CMP:.*]] = arith.cmpi slt, %[[INPUT]], %[[ZERO]] : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[INPUT]], %[[CMOD]] : [[T]]
  // CHECK: %[[SELECT:.*]] = arith.select %[[CMP]], %[[ADD]], %[[INPUT]] : [[T]]
  // CHECK: return %[[SELECT]] : [[T]]
  %result = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, -65536, 65537>
  } : i32 -> i32
  return %result : i32
}

// -----
// Test: Single-operand reduction with tensor type (non-negative range)
// Tensor types use cmpi + select (minui is only for VectorType)

// CHECK-LABEL: @test_reduce_extended_tensor
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_reduce_extended_tensor(%input: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[T]]
  // CHECK: %[[CMP:.*]] = arith.cmpi ult, %[[INPUT]], %[[CMOD]] : [[T]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[INPUT]], %[[CMOD]] : [[T]]
  // CHECK: %[[SELECT:.*]] = arith.select %[[CMP]], %[[INPUT]], %[[SUB]]
  // CHECK: return %[[SELECT]] : [[T]]
  %result = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 131074>
  } : tensor<4xi32> -> tensor<4xi32>
  return %result : tensor<4xi32>
}

// -----
// Test: Single-operand reduction with vector type (non-negative range)
// VectorType uses arith.minui for branchless comparison

// CHECK-LABEL: @test_reduce_extended_vector
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_reduce_extended_vector(%input: vector<4xi32>) -> vector<4xi32> {
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[T]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[INPUT]], %[[CMOD]] : [[T]]
  // CHECK: %[[MIN:.*]] = arith.minui %[[SUB]], %[[INPUT]] : [[T]]
  // CHECK: return %[[MIN]] : [[T]]
  %result = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 131074>
  } : vector<4xi32> -> vector<4xi32>
  return %result : vector<4xi32>
}

// -----
// Test: Montgomery reduction (single-limb)
// Should use n_inv attribute

// CHECK-LABEL: @test_montgomery_single_limb
// CHECK-SAME: (%[[LOW:.*]]: [[T:.*]], %[[HIGH:.*]]: [[T]]) -> [[T]]
func.func @test_montgomery_single_limb(%low: i32, %high: i32) -> i32 {
  // CHECK: %[[NINV:.*]] = arith.constant -65535 : [[T]]
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[T]]
  // CHECK: %[[M:.*]] = arith.muli %[[LOW]], %[[NINV]] : [[T]]
  // CHECK: arith.mului_extended %[[M]], %[[CMOD]] : [[T]]
  %result = mod_arith.intr.mont_reduce %low, %high {
    modulus = 65537 : i32,
    n_inv = -65535 : i32
  } : (i32, i32) -> i32
  return %result : i32
}

// -----
// Test: Reduction with Babybear modulus

// CHECK-LABEL: @test_reduce_babybear
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_reduce_babybear(%input: i32) -> i32 {
  // CHECK: %[[CMOD:.*]] = arith.constant 2013265921 : [[T]]
  // CHECK: arith.cmpi ult
  // CHECK: arith.subi
  // CHECK: arith.select
  %result = mod_arith.intr.reduce %input {
    modulus = 2013265921 : i32,
    input_range = #llvm.constant_range<i32, 0, -268435454>
  } : i32 -> i32
  return %result : i32
}

// -----
// Test: Two-stage lowering - Montgomery then canonicalization

// CHECK-LABEL: @test_two_stage_montgomery
// CHECK-SAME: (%[[LOW:.*]]: [[T:.*]], %[[HIGH:.*]]: [[T]]) -> [[T]]
func.func @test_two_stage_montgomery(%low: i32, %high: i32) -> i32 {
  // CHECK: arith.muli
  // CHECK: arith.mului_extended
  // Stage 1: Montgomery reduction
  %mont = mod_arith.intr.mont_reduce %low, %high {
    modulus = 65537 : i32,
    n_inv = -65535 : i32
  } : (i32, i32) -> i32
  // CHECK: arith.cmpi ult
  // CHECK: arith.subi
  // CHECK: arith.select
  // Stage 2: Canonicalization
  %result = mod_arith.intr.reduce %mont {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 131074>
  } : i32 -> i32
  return %result : i32
}

// -----
// Test: Montgomery reduction with vector type

// CHECK-LABEL: @test_montgomery_vector
// CHECK-SAME: (%[[LOW:.*]]: [[T:.*]], %[[HIGH:.*]]: [[T]]) -> [[T]]
func.func @test_montgomery_vector(%low: vector<4xi32>, %high: vector<4xi32>) -> vector<4xi32> {
  // CHECK: arith.muli
  // CHECK: arith.mului_extended
  %result = mod_arith.intr.mont_reduce %low, %high {
    modulus = 65537 : i32,
    n_inv = -65535 : i32
  } : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
  return %result : vector<4xi32>
}

// -----
// Test: Montgomery reduction with tensor type

// CHECK-LABEL: @test_montgomery_tensor
// CHECK-SAME: (%[[LOW:.*]]: [[T:.*]], %[[HIGH:.*]]: [[T]]) -> [[T]]
func.func @test_montgomery_tensor(%low: tensor<4xi32>, %high: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: arith.muli
  // CHECK: arith.mului_extended
  %result = mod_arith.intr.mont_reduce %low, %high {
    modulus = 65537 : i32,
    n_inv = -65535 : i32
  } : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}

// -----
// Test: Single-operand reduction with 64-bit integers

// CHECK-LABEL: @test_reduce_i64
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_reduce_i64(%input: i64) -> i64 {
  // CHECK: %[[CMOD:.*]] = arith.constant 4611686018427387903 : [[T]]
  // CHECK: arith.cmpi ult
  // CHECK: arith.subi
  // CHECK: arith.select
  %result = mod_arith.intr.reduce %input {
    modulus = 4611686018427387903 : i64,
    input_range = #llvm.constant_range<i64, 0, 9223372036854775806>
  } : i64 -> i64
  return %result : i64
}

// -----
// Test: Montgomery reduction with 64-bit integers

// CHECK-LABEL: @test_montgomery_i64
// CHECK-SAME: (%[[LOW:.*]]: [[T:.*]], %[[HIGH:.*]]: [[T]]) -> [[T]]
func.func @test_montgomery_i64(%low: i64, %high: i64) -> i64 {
  // CHECK: arith.muli
  // CHECK: arith.mului_extended
  %result = mod_arith.intr.mont_reduce %low, %high {
    modulus = 4611686018427387903 : i64,
    n_inv = 4611686018427387905 : i64
  } : (i64, i64) -> i64
  return %result : i64
}

// -----
// Test: Vector negative range reduction

// CHECK-LABEL: @test_reduce_negative_vector
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_reduce_negative_vector(%input: vector<4xi32>) -> vector<4xi32> {
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[T]]
  // CHECK: %[[ZERO:.*]] = arith.constant dense<0> : [[T]]
  // CHECK: arith.cmpi slt
  // CHECK: arith.addi
  // CHECK: arith.select
  %result = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, -65536, 65537>
  } : vector<4xi32> -> vector<4xi32>
  return %result : vector<4xi32>
}

// -----
// Test: Tensor negative range reduction

// CHECK-LABEL: @test_reduce_negative_tensor
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_reduce_negative_tensor(%input: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[T]]
  // CHECK: %[[ZERO:.*]] = arith.constant dense<0> : [[T]]
  // CHECK: arith.cmpi slt
  // CHECK: arith.addi
  // CHECK: arith.select
  %result = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, -65536, 65537>
  } : tensor<4xi32> -> tensor<4xi32>
  return %result : tensor<4xi32>
}

// -----
// Test: Full chain - mont_reduce followed by reduce (common pattern)

// CHECK-LABEL: @test_mont_mul_pattern
// CHECK-SAME: (%[[LOW:.*]]: [[T:.*]], %[[HIGH:.*]]: [[T]]) -> [[T]]
func.func @test_mont_mul_pattern(%low: i32, %high: i32) -> i32 {
  // First: Montgomery reduction (output in [0, 2*mod))
  // CHECK: %[[NINV:.*]] = arith.constant -65535 : [[T]]
  // CHECK: %[[CMOD1:.*]] = arith.constant 65537 : [[T]]
  // CHECK: %[[M:.*]] = arith.muli %[[LOW]], %[[NINV]] : [[T]]
  // CHECK: arith.mului_extended %[[M]], %[[CMOD1]] : [[T]]
  %mont = mod_arith.intr.mont_reduce %low, %high {
    modulus = 65537 : i32,
    n_inv = -65535 : i32
  } : (i32, i32) -> i32

  // Second: Canonicalization (output in [0, mod))
  // CHECK: %[[CMOD2:.*]] = arith.constant 65537 : [[T]]
  // CHECK: arith.cmpi ult
  // CHECK: arith.subi
  // CHECK: arith.select
  %result = mod_arith.intr.reduce %mont {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 131074>
  } : i32 -> i32
  return %result : i32
}

// -----
// Test: Goldilocks field (64-bit prime)

// CHECK-LABEL: @test_goldilocks_reduce
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_goldilocks_reduce(%input: i64) -> i64 {
  // Goldilocks prime: 2^64 - 2^32 + 1 = 18446744069414584321
  // When printed as signed i64, this is -4294967295
  // CHECK: arith.constant -4294967295 : [[T]]
  // CHECK: arith.cmpi ult
  // CHECK: arith.subi
  // CHECK: arith.select
  %result = mod_arith.intr.reduce %input {
    modulus = 18446744069414584321 : i64,
    input_range = #llvm.constant_range<i64, 0, 36893488138829168642>
  } : i64 -> i64
  return %result : i64
}

// -----
// Test: 2D tensor reduction

// CHECK-LABEL: @test_reduce_2d_tensor
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_reduce_2d_tensor(%input: tensor<2x4xi32>) -> tensor<2x4xi32> {
  // CHECK: arith.constant dense<65537> : [[T]]
  // CHECK: arith.cmpi ult
  // CHECK: arith.subi
  // CHECK: arith.select
  %result = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 131074>
  } : tensor<2x4xi32> -> tensor<2x4xi32>
  return %result : tensor<2x4xi32>
}
