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

// RUN: zkir-opt -canonicalize -field-to-mod-arith -mod-arith-to-arith -canonicalize %s | FileCheck %s -enable-var-scope

!PF17 = !field.pf<17:i32>

// CHECK-LABEL: @test_constant_folding_scalar_mul
// CHECK-SAME: (%arg0: [[T:.*]]) -> [[T]] {
func.func @test_constant_folding_scalar_mul(%arg0: tensor<8x!PF17>) -> tensor<8x!PF17> {
  // CHECK: arith.constant dense<[1, 5, 3, 7, 2, 6, 4, 8]> : [[C:.*]]
  // CHECK-NOT: tensor_ext.bit_reverse
  %const = arith.constant dense<[1, 2, 3, 4, 5, 6, 7, 8]> : tensor<8xi32>
  %twiddles = field.bitcast %const : tensor<8xi32> -> tensor<8x!PF17>
  %arg0_rev = tensor_ext.bit_reverse %arg0 into %arg0 : tensor<8x!PF17>
  %product_rev = field.mul %arg0_rev, %twiddles : tensor<8x!PF17>
  %product = tensor_ext.bit_reverse %product_rev into %product_rev : tensor<8x!PF17>
  return %product : tensor<8x!PF17>
}
