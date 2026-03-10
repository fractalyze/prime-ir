// Copyright 2026 The ZKX Authors.
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

// BN254 G1 MSM benchmark: 2²⁰ = 1,048,576 scalar multiplications.

!BN254_Fr = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256, true>
!BN254_Fp = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583 : i256, true>

#g1_curve = #elliptic_curve.sw<0 : i256, 3 : i256, (1 : i256, 2 : i256)> : !BN254_Fp

!g1_affine = !elliptic_curve.affine<#g1_curve>

module @msm_g1_bench {
  func.func public @main(%scalars: tensor<1048576x!BN254_Fr>, %bases: tensor<1048576x!g1_affine>) -> tensor<!g1_affine> {
    %0 = stablehlo.msm %scalars, %bases : (tensor<1048576x!BN254_Fr>, tensor<1048576x!g1_affine>) -> tensor<!g1_affine>
    return %0 : tensor<!g1_affine>
  }
}
