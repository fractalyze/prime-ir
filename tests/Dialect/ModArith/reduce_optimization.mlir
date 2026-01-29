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

// RUN: prime-ir-opt -mod-arith-reduce-opt %s | FileCheck %s

// -----
// Test: Eliminate reduce when input is already in canonical range [0, mod)

// CHECK-LABEL: @test_eliminate_canonical_reduce
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_eliminate_canonical_reduce(%input: i32) -> i32 {
  // CHECK-NOT: mod_arith.intr.reduce
  // CHECK: return %[[INPUT]] : [[T]]
  %result = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 65537>
  } : i32 -> i32
  return %result : i32
}

// -----
// Test: Eliminate reduce when input range is subset of [0, mod)

// CHECK-LABEL: @test_eliminate_subset_range
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_eliminate_subset_range(%input: i32) -> i32 {
  // CHECK-NOT: mod_arith.intr.reduce
  // CHECK: return %[[INPUT]] : [[T]]
  %result = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 1000>
  } : i32 -> i32
  return %result : i32
}

// -----
// Test: Keep reduce when input range exceeds modulus

// CHECK-LABEL: @test_keep_extended_reduce
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_keep_extended_reduce(%input: i32) -> i32 {
  // CHECK: mod_arith.intr.reduce
  %result = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 131074>
  } : i32 -> i32
  return %result : i32
}

// -----
// Test: Merge consecutive reduces - second reduce is eliminated

// CHECK-LABEL: @test_merge_consecutive_reduces
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_merge_consecutive_reduces(%input: i32) -> i32 {
  // CHECK: %[[R1:.*]] = mod_arith.intr.reduce %[[INPUT]]
  // CHECK-NOT: mod_arith.intr.reduce %[[R1]]
  // CHECK: return %[[R1]] : [[T]]
  %r1 = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 131074>
  } : i32 -> i32
  %r2 = mod_arith.intr.reduce %r1 {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 65537>
  } : i32 -> i32
  return %r2 : i32
}

// -----
// Test: Vector type - eliminate canonical reduce

// CHECK-LABEL: @test_eliminate_canonical_reduce_vector
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_eliminate_canonical_reduce_vector(%input: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK-NOT: mod_arith.intr.reduce
  // CHECK: return %[[INPUT]] : [[T]]
  %result = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 65537>
  } : tensor<4xi32> -> tensor<4xi32>
  return %result : tensor<4xi32>
}

// -----
// Test: Don't merge reduces with different moduli

// CHECK-LABEL: @test_no_merge_different_moduli
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_no_merge_different_moduli(%input: i32) -> i32 {
  // Both reduces should be kept (different moduli)
  // CHECK: mod_arith.intr.reduce
  // CHECK: mod_arith.intr.reduce
  %r1 = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 131074>
  } : i32 -> i32
  %r2 = mod_arith.intr.reduce %r1 {
    modulus = 65521 : i32,
    input_range = #llvm.constant_range<i32, 0, 65537>
  } : i32 -> i32
  return %r2 : i32
}

// -----
// Test: Full pipeline - mod-arith-reduce-opt then intr-reduce-to-arith

// RUN: prime-ir-opt -mod-arith-reduce-opt -intr-reduce-to-arith %s | FileCheck %s --check-prefix=PIPELINE

// PIPELINE-LABEL: @test_full_pipeline
// PIPELINE-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_full_pipeline(%input: i32) -> i32 {
  // First reduce is kept, second is eliminated
  // Then lowered to arith
  // PIPELINE-NOT: mod_arith.intr.reduce
  // PIPELINE: arith.cmpi
  // PIPELINE: arith.subi
  // PIPELINE: arith.select
  %r1 = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 131074>
  } : i32 -> i32
  %r2 = mod_arith.intr.reduce %r1 {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 65537>
  } : i32 -> i32
  return %r2 : i32
}

// -----
// Test: Triple consecutive reduces - only first is kept

// CHECK-LABEL: @test_triple_reduce
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_triple_reduce(%input: i32) -> i32 {
  // CHECK: %[[R1:.*]] = mod_arith.intr.reduce %[[INPUT]]
  // CHECK-NOT: mod_arith.intr.reduce %[[R1]]
  // CHECK: return %[[R1]] : [[T]]
  %r1 = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 131074>
  } : i32 -> i32
  %r2 = mod_arith.intr.reduce %r1 {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 65537>
  } : i32 -> i32
  %r3 = mod_arith.intr.reduce %r2 {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 65537>
  } : i32 -> i32
  return %r3 : i32
}

// -----
// Test: Eliminate reduce with zero range

// CHECK-LABEL: @test_eliminate_zero_range
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_eliminate_zero_range(%input: i32) -> i32 {
  // CHECK-NOT: mod_arith.intr.reduce
  // CHECK: return %[[INPUT]] : [[T]]
  %result = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 1>
  } : i32 -> i32
  return %result : i32
}

// -----
// Test: Eliminate reduce with single value range

// CHECK-LABEL: @test_eliminate_single_value
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_eliminate_single_value(%input: i32) -> i32 {
  // Input is known to be exactly 100 (single value range)
  // CHECK-NOT: mod_arith.intr.reduce
  // CHECK: return %[[INPUT]] : [[T]]
  %result = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 100, 101>
  } : i32 -> i32
  return %result : i32
}

// -----
// Test: Keep reduce when range touches modulus boundary

// CHECK-LABEL: @test_keep_boundary_reduce
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_keep_boundary_reduce(%input: i32) -> i32 {
  // Range is [0, 65538) which exceeds modulus 65537
  // CHECK: mod_arith.intr.reduce
  %result = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 65538>
  } : i32 -> i32
  return %result : i32
}

// -----
// Test: 64-bit optimization

// CHECK-LABEL: @test_eliminate_i64
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_eliminate_i64(%input: i64) -> i64 {
  // CHECK-NOT: mod_arith.intr.reduce
  // CHECK: return %[[INPUT]] : [[T]]
  %result = mod_arith.intr.reduce %input {
    modulus = 4611686018427387903 : i64,
    input_range = #llvm.constant_range<i64, 0, 4611686018427387903>
  } : i64 -> i64
  return %result : i64
}

// -----
// Test: Multi-use value - reduce is not eliminated if other uses exist

// CHECK-LABEL: @test_multi_use
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_multi_use(%input: i32) -> i32 {
  // The first reduce has multiple uses
  // CHECK: %[[R1:.*]] = mod_arith.intr.reduce %[[INPUT]]
  // CHECK: arith.addi %[[R1]], %[[R1]]
  %r1 = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 131074>
  } : i32 -> i32
  %sum = arith.addi %r1, %r1 : i32
  return %sum : i32
}

// -----
// Test: Negative range that spans zero

// CHECK-LABEL: @test_keep_spanning_negative
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_keep_spanning_negative(%input: i32) -> i32 {
  // Range [-1000, 1000) spans zero and includes negatives
  // CHECK: mod_arith.intr.reduce
  %result = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, -1000, 1000>
  } : i32 -> i32
  return %result : i32
}

// -----
// Test: Eliminate negative reduce when range is small positive

// CHECK-LABEL: @test_eliminate_small_positive
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_eliminate_small_positive(%input: i32) -> i32 {
  // Range [10, 100) is well within [0, mod)
  // CHECK-NOT: mod_arith.intr.reduce
  // CHECK: return %[[INPUT]] : [[T]]
  %result = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 10, 100>
  } : i32 -> i32
  return %result : i32
}

// -----
// Test: 2D tensor optimization

// CHECK-LABEL: @test_eliminate_2d_tensor
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_eliminate_2d_tensor(%input: tensor<2x4xi32>) -> tensor<2x4xi32> {
  // CHECK-NOT: mod_arith.intr.reduce
  // CHECK: return %[[INPUT]] : [[T]]
  %result = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 65537>
  } : tensor<2x4xi32> -> tensor<2x4xi32>
  return %result : tensor<2x4xi32>
}

// -----
// Test: Keep 2D tensor reduce when needed

// CHECK-LABEL: @test_keep_2d_tensor
// CHECK-SAME: (%[[INPUT:.*]]: [[T:.*]]) -> [[T]]
func.func @test_keep_2d_tensor(%input: tensor<2x4xi32>) -> tensor<2x4xi32> {
  // CHECK: mod_arith.intr.reduce
  %result = mod_arith.intr.reduce %input {
    modulus = 65537 : i32,
    input_range = #llvm.constant_range<i32, 0, 131074>
  } : tensor<2x4xi32> -> tensor<2x4xi32>
  return %result : tensor<2x4xi32>
}
