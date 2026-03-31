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

// RUN: prime-ir-opt -field-to-mod-arith="use-elementwise-inverse=true" -split-input-file %s | FileCheck %s -enable-var-scope

!PF = !field.pf<97:i32, true>

// Scalar inverse is unchanged regardless of the option.
// CHECK-LABEL: @test_scalar_inverse
func.func @test_scalar_inverse(%arg0: !PF) -> !PF {
  // CHECK-NOT: linalg.generic
  // CHECK: mod_arith.inverse
  %inv = field.inverse %arg0 : !PF
  return %inv : !PF
}

// -----

!PF = !field.pf<97:i32, true>

// 1-D tensor: per-element inverse via linalg.generic (no scf.for).
// CHECK-LABEL: @test_1d_elementwise_inverse
func.func @test_1d_elementwise_inverse(%arg0: tensor<4x!PF>) -> tensor<4x!PF> {
  // CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel"]
  // CHECK:   mod_arith.inverse
  // CHECK:   linalg.yield
  // CHECK-NOT: scf.for
  %inv = field.inverse %arg0 : tensor<4x!PF>
  return %inv : tensor<4x!PF>
}

// -----

!PF = !field.pf<97:i32, true>

// Multi-dimensional tensor: linalg.generic handles all dims natively.
// CHECK-LABEL: @test_2d_elementwise_inverse
func.func @test_2d_elementwise_inverse(%arg0: tensor<2x3x!PF>) -> tensor<2x3x!PF> {
  // CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel", "parallel"]
  // CHECK:   mod_arith.inverse
  // CHECK:   linalg.yield
  // CHECK-NOT: scf.for
  // CHECK-NOT: tensor.collapse_shape
  %inv = field.inverse %arg0 : tensor<2x3x!PF>
  return %inv : tensor<2x3x!PF>
}

// -----

!PF = !field.pf<97:i32, true>

// Rank-0 tensor: extract, invert, wrap back.
// CHECK-LABEL: @test_rank0_elementwise_inverse
func.func @test_rank0_elementwise_inverse(%arg0: tensor<!PF>) -> tensor<!PF> {
  // CHECK: tensor.extract
  // CHECK: mod_arith.inverse
  // CHECK: tensor.from_elements
  // CHECK-NOT: linalg.generic
  %inv = field.inverse %arg0 : tensor<!PF>
  return %inv : tensor<!PF>
}

// -----

!PF = !field.pf<97:i32, true>

// Dynamic 1-D tensor: linalg.generic with tensor.dim for output shape.
// CHECK-LABEL: @test_dynamic_1d_elementwise_inverse
func.func @test_dynamic_1d_elementwise_inverse(%arg0: tensor<?x!PF>) -> tensor<?x!PF> {
  // CHECK: tensor.dim
  // CHECK: tensor.empty
  // CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel"]
  // CHECK:   mod_arith.inverse
  // CHECK:   linalg.yield
  // CHECK-NOT: scf.for
  %inv = field.inverse %arg0 : tensor<?x!PF>
  return %inv : tensor<?x!PF>
}

// -----

!PF = !field.pf<97:i32, true>

// Dynamic multi-dimensional tensor: linalg.generic handles it natively.
// CHECK-LABEL: @test_dynamic_2d_elementwise_inverse
func.func @test_dynamic_2d_elementwise_inverse(%arg0: tensor<?x3x!PF>) -> tensor<?x3x!PF> {
  // CHECK: tensor.dim
  // CHECK: tensor.empty
  // CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel", "parallel"]
  // CHECK:   mod_arith.inverse
  // CHECK:   linalg.yield
  %inv = field.inverse %arg0 : tensor<?x3x!PF>
  return %inv : tensor<?x3x!PF>
}
