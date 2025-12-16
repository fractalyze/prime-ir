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

#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PF
!affine = !elliptic_curve.affine<#curve>
!jacobian = !elliptic_curve.jacobian<#curve>
!xyzz = !elliptic_curve.xyzz<#curve>

#g2a = dense<[0,0]> : tensor<2xi256>
#g2b = dense<[19485874751759354771024239261021720505790618469301721065564631296452457478373,266929791119991161246907387137283842545076965332900288569378510910307636690]> : tensor<2xi256>
#g2x = dense<[10857046999023057135944570762232829481370756359578518086990519993285655852781,11559732032986387107991004021392285783925812861821192530917403151452391805634]> : tensor<2xi256>
#g2y = dense<[8495653923123431417604973247489272438418190587263600148770280649306958101930,4082367875863433681332203403145435568316851327593401208105741076214120093531]> : tensor<2xi256>
#g2curve = #elliptic_curve.sw<#g2a, #g2b, (#g2x, #g2y)> : !QF
!g2affine = !elliptic_curve.affine<#g2curve>
!g2jacobian = !elliptic_curve.jacobian<#g2curve>
!g2xyzz = !elliptic_curve.xyzz<#g2curve>

func.func private @printMemrefBn254G1AffineStd(memref<*x!affine>) attributes { llvm.emit_c_interface }

func.func private @printMemrefG1Affine(%affine: memref<*x!affine>) {
  func.call @printMemrefBn254G1AffineStd(%affine) : (memref<*x!affine>) -> ()
  return
}

func.func private @printMemrefBn254G1JacobianStd(memref<*x!jacobian>) attributes { llvm.emit_c_interface }

func.func private @printMemrefG1Jacobian(%jacobian: memref<*x!jacobian>) {
  func.call @printMemrefBn254G1JacobianStd(%jacobian) : (memref<*x!jacobian>) -> ()
  return
}

func.func private @printMemrefBn254G1XyzzStd(memref<*x!xyzz>) attributes { llvm.emit_c_interface }

func.func private @printMemrefG1Xyzz(%xyzz: memref<*x!xyzz>) {
  func.call @printMemrefBn254G1XyzzStd(%xyzz) : (memref<*x!xyzz>) -> ()
  return
}

// assumes standard form scalar input and outputs affine with standard form coordinates
func.func @getG1GeneratorMultiple(%k: !SF) -> !affine {
  %one = field.constant 1 : !PF
  %two = field.constant 2 : !PF
  %g = elliptic_curve.point %one, %two : (!PF, !PF) -> !affine
  %g_multiple = elliptic_curve.scalar_mul %k, %g : !SF, !affine -> !jacobian
  %g_multiple_affine = elliptic_curve.convert_point_type %g_multiple : !jacobian -> !affine
  return %g_multiple_affine : !affine
}
