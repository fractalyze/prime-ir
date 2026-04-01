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

// RUN: cat %S/../../bn254_defs.mlir %s \
// RUN:   | prime-ir-opt -elliptic-curve-to-field='lowering-mode=aot_runtime' \
// RUN:   | FileCheck %s -enable-var-scope

// Rank-0 tensor affine → jacobian conversion must not crash in AOT mode.
// The AOT path must unwrap the tensor type before casting to PointTypeInterface.
// CHECK-LABEL: @test_r0_affine_to_jacobian_aot
func.func @test_r0_affine_to_jacobian_aot(%arg0: tensor<!affinem>) -> tensor<!jacobianm> {
  // CHECK: tensor.extract
  // CHECK: call @ec_affine_to_jacobian_bn254_g1_mont
  // CHECK: tensor.from_elements
  %result = elliptic_curve.convert_point_type %arg0 : tensor<!affinem> -> tensor<!jacobianm>
  return %result : tensor<!jacobianm>
}

// Rank-0 tensor jacobian → affine should also work in AOT mode.
// CHECK-LABEL: @test_r0_jacobian_to_affine_aot
func.func @test_r0_jacobian_to_affine_aot(%arg0: tensor<!jacobianm>) -> tensor<!affinem> {
  // CHECK: tensor.extract
  // CHECK: call @ec_jacobian_to_affine_bn254_g1_mont
  // CHECK: tensor.from_elements
  %result = elliptic_curve.convert_point_type %arg0 : tensor<!jacobianm> -> tensor<!affinem>
  return %result : tensor<!affinem>
}
