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

// Shared BN254 type aliases for AOT runtime MLIR sources and tests.
// Naming follows existing test conventions (bn254_field_defs.mlir,
// bn254_ec_defs.mlir, bn254_ec_mont_defs.mlir).

// ===== Field types =====

!PF = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!SF = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256>
!PFm = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256, true>
!SFm = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256, true>
!QF = !field.ef<2x!PF, 21888242871839275222246405745257275088696311157297823662689037894645226208582:i256>
!QFm = !field.ef<2x!PFm, 21888242871839275222246405745257275088696311157297823662689037894645226208582:i256>

// zk_dtypes convention aliases for AOT runtime function naming.
!bn254_bfx2 = !QF
!bn254_bfx2m = !QFm

// ===== G1 EC types (standard) =====

#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PF
!affine = !elliptic_curve.affine<#curve>
!jacobian = !elliptic_curve.jacobian<#curve>
!xyzz = !elliptic_curve.xyzz<#curve>

// ===== G2 EC types (standard) =====

#g2a = dense<[0,0]> : tensor<2xi256>
#g2b = dense<[19485874751759354771024239261021720505790618469301721065564631296452457478373,266929791119991161246907387137283842545076965332900288569378510910307636690]> : tensor<2xi256>
#g2x = dense<[10857046999023057135944570762232829481370756359578518086990519993285655852781,11559732032986387107991004021392285783925812861821192530917403151452391805634]> : tensor<2xi256>
#g2y = dense<[8495653923123431417604973247489272438418190587263600148770280649306958101930,4082367875863433681332203403145435568316851327593401208105741076214120093531]> : tensor<2xi256>
#g2curve = #elliptic_curve.sw<#g2a, #g2b, (#g2x, #g2y)> : !QF
!g2affine = !elliptic_curve.affine<#g2curve>
!g2jacobian = !elliptic_curve.jacobian<#g2curve>
!g2xyzz = !elliptic_curve.xyzz<#g2curve>

// ===== G1 EC types (Montgomery) =====

#curvem = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PFm
!affinem = !elliptic_curve.affine<#curvem>
!jacobianm = !elliptic_curve.jacobian<#curvem>
!xyzzm = !elliptic_curve.xyzz<#curvem>

// ===== G2 EC types (Montgomery) =====

#g2curvem = #elliptic_curve.sw<#g2a, #g2b, (#g2x, #g2y)> : !QFm
!g2affinem = !elliptic_curve.affine<#g2curvem>
!g2jacobianm = !elliptic_curve.jacobian<#g2curvem>
!g2xyzzm = !elliptic_curve.xyzz<#g2curvem>
