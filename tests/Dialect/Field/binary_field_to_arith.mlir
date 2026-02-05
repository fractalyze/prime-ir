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

// RUN: prime-ir-opt --binary-field-to-arith %s | FileCheck %s

// Test binary field to arith lowering

!BF8 = !field.bf<3>  // GF(2^8)

// CHECK-LABEL: @test_bf_constant
// CHECK-SAME: () -> i8
func.func @test_bf_constant() -> !BF8 {
  // CHECK: arith.constant 42 : i8
  %c = field.constant 42 : !BF8
  return %c : !BF8
}

// CHECK-LABEL: @test_bf_add
// CHECK-SAME: (%arg0: i8, %arg1: i8) -> i8
func.func @test_bf_add(%a: !BF8, %b: !BF8) -> !BF8 {
  // CHECK: arith.xori
  %c = field.add %a, %b : !BF8
  return %c : !BF8
}

// CHECK-LABEL: @test_bf_sub
// CHECK-SAME: (%arg0: i8, %arg1: i8) -> i8
func.func @test_bf_sub(%a: !BF8, %b: !BF8) -> !BF8 {
  // CHECK: arith.xori
  %c = field.sub %a, %b : !BF8
  return %c : !BF8
}

// CHECK-LABEL: @test_bf_negate
// CHECK-SAME: (%arg0: i8) -> i8
func.func @test_bf_negate(%a: !BF8) -> !BF8 {
  // CHECK: return %arg0 : i8
  %c = field.negate %a : !BF8
  return %c : !BF8
}

// CHECK-LABEL: @test_bf_double
// CHECK-SAME: (%arg0: i8) -> i8
func.func @test_bf_double(%a: !BF8) -> !BF8 {
  // CHECK: arith.constant 0 : i8
  %c = field.double %a : !BF8
  return %c : !BF8
}

// CHECK-LABEL: @test_bf_mul
// CHECK-SAME: (%arg0: i8, %arg1: i8) -> i8
func.func @test_bf_mul(%a: !BF8, %b: !BF8) -> !BF8 {
  // CHECK: arith.trunci
  // CHECK: arith.shrui
  // CHECK: arith.xori
  // CHECK: arith.ori
  %c = field.mul %a, %b : !BF8
  return %c : !BF8
}

// CHECK-LABEL: @test_bf_square
// CHECK-SAME: (%arg0: i8) -> i8
func.func @test_bf_square(%a: !BF8) -> !BF8 {
  // CHECK: arith.trunci
  // CHECK: arith.shrui
  // CHECK: arith.xori
  // CHECK: arith.ori
  %c = field.square %a : !BF8
  return %c : !BF8
}
