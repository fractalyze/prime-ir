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

// BN254 pairing check: verifies e(P, Q) * e(-P, Q) = 1.

!BN254_Fp = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583 : i256, true>
!BN254_Fq2 = !field.ef<2x!BN254_Fp, 21888242871839275222246405745257275088696311157297823662689037894645226208582 : i256>

#g1_curve = #elliptic_curve.sw<0 : i256, 3 : i256, (1 : i256, 2 : i256)> : !BN254_Fp

#g2_curve = #elliptic_curve.sw<
  dense<[0, 0]> : tensor<2xi256>,
  dense<[19485874751759354771024239261021720505790618469301721065564631296452457478373, 266929791119991161246907387137283842545076965332900288569378510910307636690]> : tensor<2xi256>,
  (dense<[10857046999023057135944570762232829481370756359578518086990519993285655852781, 11559732032986387107991004021392285783925812861821192530917403151452391805634]> : tensor<2xi256>,
   dense<[8495653923123431417604973247489272438418190587263600148770280649306958101930, 4082367875863433681332203403145435568316851327593401208105741076214120093531]> : tensor<2xi256>)
> : !BN254_Fq2

!g1_affine = !elliptic_curve.affine<#g1_curve>
!g2_affine = !elliptic_curve.affine<#g2_curve>

module @pairing_check_test {
  func.func public @main(%g1: tensor<2x!g1_affine>, %g2: tensor<2x!g2_affine>) -> tensor<i1> {
    %0 = stablehlo.pairing_check %g1, %g2 : (tensor<2x!g1_affine>, tensor<2x!g2_affine>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}
