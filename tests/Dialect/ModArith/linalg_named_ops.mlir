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

// RUN: prime-ir-opt -mod-arith-to-arith %s | FileCheck %s -enable-var-scope

!Zp = !mod_arith.int<65537 : i32>

// CHECK-LABEL: @test_linalg_dot
// Named linalg ops are generalized to linalg.generic
// CHECK: linalg.generic
func.func @test_linalg_dot(%a: tensor<4x!Zp>, %b: tensor<4x!Zp>, %init: tensor<!Zp>) -> tensor<!Zp> {
  %res = linalg.dot ins(%a, %b : tensor<4x!Zp>, tensor<4x!Zp>) outs(%init : tensor<!Zp>) -> tensor<!Zp>
  return %res : tensor<!Zp>
}

// CHECK-LABEL: @test_linalg_matvec
// CHECK: linalg.generic
func.func @test_linalg_matvec(%A: tensor<3x4x!Zp>, %v: tensor<4x!Zp>, %init: tensor<3x!Zp>) -> tensor<3x!Zp> {
  %res = linalg.matvec ins(%A, %v : tensor<3x4x!Zp>, tensor<4x!Zp>) outs(%init : tensor<3x!Zp>) -> tensor<3x!Zp>
  return %res : tensor<3x!Zp>
}

// CHECK-LABEL: @test_linalg_matmul
// CHECK: linalg.generic
func.func @test_linalg_matmul(%A: tensor<3x4x!Zp>, %B: tensor<4x5x!Zp>, %init: tensor<3x5x!Zp>) -> tensor<3x5x!Zp> {
  %res = linalg.matmul ins(%A, %B : tensor<3x4x!Zp>, tensor<4x5x!Zp>) outs(%init : tensor<3x5x!Zp>) -> tensor<3x5x!Zp>
  return %res : tensor<3x5x!Zp>
}
