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

// RUN: cat %S/../../bn254_defs.mlir %S/../../bn254_ec_mont_helpers.mlir %s \
// RUN:   | prime-ir-opt -linalg-generalize-named-ops -split-input-file \
// RUN:   | FileCheck %s -enable-var-scope

// CHECK-LABEL: @test_g2_msm_by_dot_product
func.func @test_g2_msm_by_dot_product(%scalars: tensor<3x!PF>, %points: tensor<3x!g2jacobianm>) -> tensor<!g2jacobianm> {
  // CHECK-NOT: linalg.dot
  %result = tensor.empty() : tensor<!g2jacobianm>
  %msm_result = linalg.dot ins(%scalars, %points : tensor<3x!PF>, tensor<3x!g2jacobianm>) outs(%result: tensor<!g2jacobianm>) -> tensor<!g2jacobianm>
  return %msm_result : tensor<!g2jacobianm>
}

// CHECK-LABEL: @test_affine_tensor_reduce_to_jacobian
func.func @test_affine_tensor_reduce_to_jacobian(%points: tensor<4x!affinem>) -> tensor<!jacobianm> {
  // Create zero point for jacobian accumulation
  %zeroPF = field.constant 0 : !PFm
  %onePF = field.constant 1 : !PFm
  %zero_jacobian = elliptic_curve.from_coords %onePF, %onePF, %zeroPF : (!PFm, !PFm, !PFm) -> !jacobianm

  // Initialize output tensor with zero point
  %init = tensor.from_elements %zero_jacobian : tensor<!jacobianm>

  // Use linalg.reduce to sum all affine points into jacobian result
  // CHECK-NOT: linalg.reduce
  %reduced = linalg.reduce ins(%points : tensor<4x!affinem>) outs(%init : tensor<!jacobianm>) dimensions = [0]
    (%affine_point: !affinem, %jacobian_acc: !jacobianm) {
      %sum = elliptic_curve.add %affine_point, %jacobian_acc : !affinem, !jacobianm -> !jacobianm
      linalg.yield %sum : !jacobianm
    }

  return %reduced: tensor<!jacobianm>
}
