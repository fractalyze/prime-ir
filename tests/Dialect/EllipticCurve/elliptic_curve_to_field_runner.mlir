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

// RUN: cat %S/../../default_print_utils.mlir %S/../../bn254_defs.mlir %S/../../bn254_ec_mont_helpers.mlir %S/../../bn254_ec_utils.mlir %s \
// RUN:   | prime-ir-opt -elliptic-curve-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_ops_in_order -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext,%S/../../libruntime_functions%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_OPS_IN_ORDER < %t

// CHECK-LABEL: @test_ops_in_order
func.func @test_ops_in_order() {
  %var1 = field.constant 1 : !PFm
  %var2 = field.constant 2 : !PFm
  %var7 = field.constant 7 : !SFm

  // (1,2)
  %affine1 = elliptic_curve.from_coords %var1, %var2 : (!PFm, !PFm) -> !affinem
  // (1,2,1)
  %jacobian1 = elliptic_curve.from_coords %var1, %var2, %var1 : (!PFm, !PFm, !PFm) -> !jacobianm

  %jacobian2 = elliptic_curve.add %affine1, %jacobian1 : !affinem, !jacobianm -> !jacobianm
  func.call @printG1JacobianMont(%jacobian2) : (!jacobianm) -> ()

  %jacobian3 = elliptic_curve.sub %affine1, %jacobian2 : !affinem, !jacobianm -> !jacobianm
  func.call @printG1JacobianMont(%jacobian3) : (!jacobianm) -> ()

  %jacobian4 = elliptic_curve.negate %jacobian3 : !jacobianm
  func.call @printG1JacobianMont(%jacobian4) : (!jacobianm) -> ()

  %jacobian5 = elliptic_curve.double %jacobian4 : !jacobianm -> !jacobianm
  func.call @printG1JacobianMont(%jacobian5) : (!jacobianm) -> ()

  %xyzz1 = elliptic_curve.convert_point_type %affine1 : !affinem -> !xyzzm
  func.call @printG1XyzzMont(%xyzz1) : (!xyzzm) -> ()

  %affine2 = elliptic_curve.convert_point_type %xyzz1 : !xyzzm -> !affinem
  func.call @printG1AffineMont(%affine2) : (!affinem) -> ()

  %jacobian6 = elliptic_curve.scalar_mul %var7, %affine2 : !SFm, !affinem -> !jacobianm
  %affine2_1 = elliptic_curve.convert_point_type %jacobian6 : !jacobianm -> !affinem
  %jacobian6_1 = elliptic_curve.convert_point_type %affine2_1 : !affinem -> !jacobianm
  func.call @printG1JacobianMont(%jacobian6_1) : (!jacobianm) -> ()

  %affine3 = elliptic_curve.convert_point_type %jacobian6 : !jacobianm -> !affinem
  func.call @printG1AffineMont(%affine3) : (!affinem) -> ()

  %xyzz2 = elliptic_curve.add %affine3, %xyzz1 : !affinem, !xyzzm -> !xyzzm
  %affine4 = elliptic_curve.convert_point_type %xyzz2 : !xyzzm -> !affinem
  %xyzz3 = elliptic_curve.convert_point_type %affine4 : !affinem -> !xyzzm
  func.call @printG1XyzzMont(%xyzz3) : (!xyzzm) -> ()

  return
}

// CHECK_TEST_OPS_IN_ORDER: [(21888242871839275222246405745257275088696311157297823662689037894645226208560, 21888242871839275222246405745257275088696311157297823662689037894645226208572, 4)]
// CHECK_TEST_OPS_IN_ORDER: [(97344, 21888242871839275222246405745257275088696311157297823662689037894645165465927, 312)]
// CHECK_TEST_OPS_IN_ORDER: [(97344, 60742656, 312)]
// CHECK_TEST_OPS_IN_ORDER: [(21888242871839275222246405745257275088696311157297823660623826140512156187975, 21888242871839275222246405745257275088696311147938427866742942580146565872967, 37903417344)]
// CHECK_TEST_OPS_IN_ORDER: [(1, 2, 1, 1)]
// CHECK_TEST_OPS_IN_ORDER: [(1, 2)]
// CHECK_TEST_OPS_IN_ORDER: [(10415861484417082502655338383609494480414113902179649885744799961447382638712, 10196215078179488638353184030336251401353352596818396260819493263908881608606, 1)]
// CHECK_TEST_OPS_IN_ORDER: [(10415861484417082502655338383609494480414113902179649885744799961447382638712, 10196215078179488638353184030336251401353352596818396260819493263908881608606)]
// CHECK_TEST_OPS_IN_ORDER: [(3932705576657793550893430333273221375907985235130430286685735064194643946083, 18813763293032256545937756946359266117037834559191913266454084342712532869153, 1, 1)]
