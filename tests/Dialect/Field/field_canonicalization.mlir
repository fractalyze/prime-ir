// Copyright 2025 The ZKIR Authors.
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

// RUN: zkir-opt -canonicalize %s | FileCheck %s -enable-var-scope

!PF17 = !field.pf<17:i32>

// TODO(batzor): add tests related with constant folding after constant folding
// is enabled
//===----------------------------------------------------------------------===//
// AddOp patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_add_self_is_double
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_add_self_is_double(%arg0: !PF17) -> !PF17 {
  %add = field.add %arg0, %arg0 : !PF17
  // CHECK: %[[RES:.*]] = field.double %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %add : !PF17
}

// CHECK-LABEL: @test_add_both_negated
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_add_both_negated(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  %neg0 = field.negate %arg0 : !PF17
  %neg1 = field.negate %arg1 : !PF17
  %add = field.add %neg0, %neg1 : !PF17
  // CHECK: %[[ADD:.*]] = field.add %[[ARG0]], %[[ARG1]] : [[T]]
  // CHECK: %[[RES:.*]] = field.negate %[[ADD]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %add : !PF17
}

// CHECK-LABEL: @test_add_after_sub
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_add_after_sub(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  %sub = field.sub %arg0, %arg1 : !PF17
  %add = field.add %sub, %arg1 : !PF17
  // CHECK: return %[[ARG0]] : [[T]]
  return %add : !PF17
}

// CHECK-LABEL: @test_add_after_neg_lhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_add_after_neg_lhs(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  %neg = field.negate %arg0 : !PF17
  %add = field.add %neg, %arg1 : !PF17
  // CHECK: %[[RES:.*]] = field.sub %[[ARG1]], %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %add : !PF17
}

// CHECK-LABEL: @test_add_after_neg_rhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_add_after_neg_rhs(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  %neg = field.negate %arg1 : !PF17
  %add = field.add %arg0, %neg : !PF17
  // CHECK: %[[RES:.*]] = field.sub %[[ARG0]], %[[ARG1]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %add : !PF17
}

// CHECK-LABEL: @test_factor_mul_add
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]], %[[ARG2:.*]]: [[T]]) -> [[T]]
func.func @test_factor_mul_add(%arg0: !PF17, %arg1: !PF17, %arg2: !PF17) -> !PF17 {
  %mul1 = field.mul %arg0, %arg1 : !PF17
  %mul2 = field.mul %arg0, %arg2 : !PF17
  %add = field.add %mul1, %mul2 : !PF17
  // CHECK: %[[ADD:.*]] = field.add %[[ARG1]], %[[ARG2]] : [[T]]
  // CHECK: %[[RES:.*]] = field.mul %[[ARG0]], %[[ADD]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %add : !PF17
}

//===----------------------------------------------------------------------===//
// SubOp patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_sub_lhs_after_add
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_lhs_after_add(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  %add = field.add %arg0, %arg1 : !PF17
  %sub = field.sub %add, %arg0 : !PF17
  // CHECK: return %[[ARG1]] : [[T]]
  return %sub : !PF17
}

// CHECK-LABEL: @test_sub_rhs_after_add
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_rhs_after_add(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  %add = field.add %arg0, %arg1 : !PF17
  %sub = field.sub %add, %arg1 : !PF17
  // CHECK: return %[[ARG0]] : [[T]]
  return %sub : !PF17
}

// CHECK-LABEL: @test_sub_lhs_after_sub
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_lhs_after_sub(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  %sub1 = field.sub %arg0, %arg1 : !PF17
  %sub2 = field.sub %sub1, %arg0 : !PF17
  // CHECK: %[[RES:.*]] = field.negate %[[ARG1]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %sub2 : !PF17
}

// CHECK-LABEL: @test_sub_after_neg_lhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_after_neg_lhs(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  %neg = field.negate %arg0 : !PF17
  %sub = field.sub %neg, %arg1 : !PF17
  // CHECK: %[[SUM:.*]] = field.add %[[ARG0]], %[[ARG1]] : [[T]]
  // CHECK: %[[RES:.*]] = field.negate %[[SUM]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %sub : !PF17
}

// CHECK-LABEL: @test_sub_after_neg_rhs
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_after_neg_rhs(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  %neg = field.negate %arg1 : !PF17
  %sub = field.sub %arg0, %neg : !PF17
  // CHECK: %[[RES:.*]] = field.add %[[ARG0]], %[[ARG1]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %sub : !PF17
}

// CHECK-LABEL: @test_sub_both_negated
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_both_negated(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  %neg0 = field.negate %arg0 : !PF17
  %neg1 = field.negate %arg1 : !PF17
  %sub = field.sub %neg0, %neg1 : !PF17
  // CHECK: %[[RES:.*]] = field.sub %[[ARG1]], %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %sub : !PF17
}

// CHECK-LABEL: @test_sub_after_square_both
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_after_square_both(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  %sq0 = field.square %arg0 : !PF17
  %sq1 = field.square %arg1 : !PF17
  %sub = field.sub %sq0, %sq1 : !PF17
  // CHECK: %[[SUB:.*]] = field.sub %[[ARG0]], %[[ARG1]] : [[T]]
  // CHECK: %[[ADD:.*]] = field.add %[[ARG0]], %[[ARG1]] : [[T]]
  // CHECK: %[[RES:.*]] = field.mul %[[SUB]], %[[ADD]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %sub : !PF17
}

// CHECK-LABEL: @test_factor_mul_sub
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]], %[[ARG2:.*]]: [[T]]) -> [[T]]
func.func @test_factor_mul_sub(%arg0: !PF17, %arg1: !PF17, %arg2: !PF17) -> !PF17 {
  %mul1 = field.mul %arg0, %arg1 : !PF17
  %mul2 = field.mul %arg0, %arg2 : !PF17
  %sub = field.sub %mul1, %mul2 : !PF17
  // CHECK: %[[SUB:.*]] = field.sub %[[ARG1]], %[[ARG2]] : [[T]]
  // CHECK: %[[RES:.*]] = field.mul %[[ARG0]], %[[SUB]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %sub : !PF17
}

//===----------------------------------------------------------------------===//
// MulOp patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_mul_self_is_square
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_self_is_square(%arg0: !PF17) -> !PF17 {
  %mul = field.mul %arg0, %arg0 : !PF17
  // CHECK: %[[RES:.*]] = field.square %[[ARG0]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  return %mul : !PF17
}
