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

// RUN: prime-ir-opt -inline %s | FileCheck %s

!Zp = !mod_arith.int<37 : i32>

//===----------------------------------------------------------------------===//
// Inline with multiple operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_inline_multiple_ops
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]], %[[ARG1:.*]]: [[T]]) -> [[T]]
func.func @test_inline_multiple_ops(%arg0: !Zp, %arg1: !Zp) -> !Zp {
  // CHECK-NOT: func.call @helper_complex
  // CHECK: %[[C3:.*]] = mod_arith.constant 3 : [[T]]
  // CHECK: %[[MUL:.*]] = mod_arith.mul %[[ARG0]], %[[C3]] : [[T]]
  // CHECK: %[[ADD:.*]] = mod_arith.add %[[MUL]], %[[ARG1]] : [[T]]
  // CHECK: %[[RES:.*]] = mod_arith.square %[[ADD]] : [[T]]
  // CHECK: return %[[RES]] : [[T]]
  %0 = func.call @helper_complex(%arg0, %arg1) : (!Zp, !Zp) -> (!Zp)
  return %0 : !Zp
}

func.func private @helper_complex(%x: !Zp, %y: !Zp) -> !Zp {
  %c3 = mod_arith.constant 3 : !Zp
  %mul = mod_arith.mul %x, %c3 : !Zp
  %add = mod_arith.add %mul, %y : !Zp
  %square = mod_arith.square %add : !Zp
  return %square : !Zp
}

//===----------------------------------------------------------------------===//
// Inline with nested calls
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_inline_nested
// CHECK-SAME: (%[[ARG0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_inline_nested(%arg0: !Zp) -> !Zp {
  // CHECK-NOT: func.call @helper_nested
  // CHECK-NOT: func.call @helper_complex
  // CHECK: %[[C5:.*]] = mod_arith.constant 5 : [[T]]
  // CHECK: %[[C3:.*]] = mod_arith.constant 3 : [[T]]
  // CHECK: %[[MUL:.*]] = mod_arith.mul %[[ARG0]], %[[C3]] : [[T]]
  // CHECK: %[[ADD:.*]] = mod_arith.add %[[MUL]], %[[ARG0]] : [[T]]
  // CHECK: %[[SQUARE:.*]] = mod_arith.square %[[ADD]] : [[T]]
  // CHECK: %[[ADD2:.*]] = mod_arith.add %[[SQUARE]], %[[C5]] : [[T]]
  // CHECK: return %[[ADD2]] : [[T]]
  %0 = func.call @helper_nested(%arg0) : (!Zp) -> (!Zp)
  return %0 : !Zp
}

func.func private @helper_nested(%x: !Zp) -> !Zp {
  %0 = func.call @helper_complex(%x, %x) : (!Zp, !Zp) -> (!Zp)
  %c5 = mod_arith.constant 5 : !Zp
  %1 = mod_arith.add %0, %c5 : !Zp
  return %1 : !Zp
}
