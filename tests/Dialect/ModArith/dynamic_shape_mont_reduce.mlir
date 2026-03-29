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

// Verify that mod-arith-to-arith handles dynamic tensor shapes without
// SplatElementsAttr assertion failures.  Shape-parametric compilation
// (#411) produces dynamic tensor types (tensor<?x...>) that must pass
// through Montgomery reduction without crashing.

// RUN: prime-ir-opt -convert-elementwise-to-linalg -field-to-mod-arith -mod-arith-to-arith %s | FileCheck %s

!pf = !field.pf<2013265921 : i32, true>

// CHECK-LABEL: @test_dynamic_mont_mul
func.func @test_dynamic_mont_mul(%a: tensor<?x!pf>, %b: tensor<?x!pf>) -> tensor<?x!pf> {
  // elementwise-to-linalg converts this to linalg.generic with scalar
  // field.mul body, which mod-arith-to-arith then lowers without needing
  // SplatElementsAttr on dynamic tensor types.
  // CHECK: linalg.generic
  // CHECK: arith.mului_extended
  %res = field.mul %a, %b : tensor<?x!pf>
  return %res : tensor<?x!pf>
}

// CHECK-LABEL: @test_dynamic_mont_add
func.func @test_dynamic_mont_add(%a: tensor<?x!pf>, %b: tensor<?x!pf>) -> tensor<?x!pf> {
  // CHECK: linalg.generic
  // CHECK: arith.addi
  %res = field.add %a, %b : tensor<?x!pf>
  return %res : tensor<?x!pf>
}

// CHECK-LABEL: @test_dynamic_mont_sub
func.func @test_dynamic_mont_sub(%a: tensor<?x!pf>, %b: tensor<?x!pf>) -> tensor<?x!pf> {
  // CHECK: linalg.generic
  // CHECK: arith.subi
  %res = field.sub %a, %b : tensor<?x!pf>
  return %res : tensor<?x!pf>
}

// CHECK-LABEL: @test_dynamic_broadcast_mul
func.func @test_dynamic_broadcast_mul(
    %vec: tensor<?x!pf>, %scalar: tensor<!pf>) -> tensor<?x!pf> {
  // Broadcast scalar to match vec's dynamic shape, then multiply.
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %vec, %c0 : tensor<?x!pf>
  %empty = tensor.empty(%dim) : tensor<?x!pf>
  %bc = linalg.broadcast ins(%scalar : tensor<!pf>)
                          outs(%empty : tensor<?x!pf>)
                          dimensions = [0]
  // CHECK: linalg.generic
  // CHECK: arith.mului_extended
  %res = field.mul %vec, %bc : tensor<?x!pf>
  return %res : tensor<?x!pf>
}

// CHECK-LABEL: @test_dynamic_2d_mont_mul
func.func @test_dynamic_2d_mont_mul(
    %a: tensor<3x?x!pf>, %b: tensor<3x?x!pf>) -> tensor<3x?x!pf> {
  // 2D dynamic tensor (e.g., stacked polynomials with variable width).
  // CHECK: linalg.generic
  // CHECK: arith.mului_extended
  %res = field.mul %a, %b : tensor<3x?x!pf>
  return %res : tensor<3x?x!pf>
}
