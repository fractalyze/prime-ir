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

// RUN: zkir-opt -inline %s | FileCheck %s

!PF17 = !field.pf<17:i32>

//===----------------------------------------------------------------------===//
// Inline with multiple operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_inline_multiple_ops
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_inline_multiple_ops(%arg0: !PF17, %arg1: !PF17) -> !PF17 {
  // CHECK-NOT: func.call @helper_complex
  // CHECK: %[[C3:.*]] = field.constant 3
  // CHECK: %[[MUL:.*]] = field.mul %[[ARG0]], %[[C3]] : [[T]]
  // CHECK: %[[ADD:.*]] = field.add %[[MUL]], %[[ARG1]] : [[T]]
  // CHECK: %[[RES:.*]] = field.square %[[ADD]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  %0 = func.call @helper_complex(%arg0, %arg1) : (!PF17, !PF17) -> (!PF17)
  return %0 : !PF17
}

func.func private @helper_complex(%x: !PF17, %y: !PF17) -> !PF17 {
  %c3 = field.constant 3 : !PF17
  %mul = field.mul %x, %c3 : !PF17
  %add = field.add %mul, %y : !PF17
  %square = field.square %add : !PF17
  return %square : !PF17
}

//===----------------------------------------------------------------------===//
// Inline with nested calls
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_inline_nested
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_inline_nested(%arg0: !PF17) -> !PF17 {
  // CHECK-NOT: func.call @helper_nested
  // CHECK-NOT: func.call @helper_complex
  // CHECK: %[[C5:.*]] = field.constant 5
  // CHECK: %[[C3:.*]] = field.constant 3
  // CHECK: %[[MUL:.*]] = field.mul %[[ARG0]], %[[C3]] : [[T]]
  // CHECK: %[[ADD:.*]] = field.add %[[MUL]], %[[ARG0]] : [[T]]
  // CHECK: %[[SQUARE:.*]] = field.square %[[ADD]] : [[T]]
  // CHECK: %[[ADD2:.*]] = field.add %[[SQUARE]], %[[C5]] : [[T]]
  // CHECK: return %[[ADD2]] : [[T]]
  %0 = func.call @helper_nested(%arg0) : (!PF17) -> (!PF17)
  return %0 : !PF17
}

func.func private @helper_nested(%x: !PF17) -> !PF17 {
  %0 = func.call @helper_complex(%x, %x) : (!PF17, !PF17) -> (!PF17)
  %c5 = field.constant 5 : !PF17
  %1 = field.add %0, %c5 : !PF17
  return %1 : !PF17
}
