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

// RUN: cat %S/../../default_print_utils.mlir %S/../../bn254_field_defs.mlir %S/../../bn254_ec_mont_defs.mlir %S/../../bn254_ec_utils.mlir %s \
// RUN:   | zkir-opt -elliptic-curve-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_ops_in_order -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext,%S/../../libruntime_functions%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_OPS_IN_ORDER < %t

// CHECK-LABEL: @test_ops_in_order
func.func @test_ops_in_order() {
  %c1 = arith.constant 1 : i256
  %c2 = arith.constant 2 : i256

  %c7 = field.constant 7 : !SF

  %index1 = arith.constant 0 : index
  %index2 = arith.constant 1 : index

  %c_tensor = tensor.from_elements %c1, %c2: tensor<2xi256>
  %f_tensor = field.bitcast %c_tensor : tensor<2xi256> -> tensor<2x!PF>
  %c_mont_tensor = field.to_mont %f_tensor : tensor<2x!PFm>
  %var1 = tensor.extract %c_mont_tensor[%index1] : tensor<2x!PFm>
  %var2 = tensor.extract %c_mont_tensor[%index2] : tensor<2x!PFm>
  %var7 = field.to_mont %c7 : !SFm

  // (1,2)
  %affine1 = elliptic_curve.point %var1, %var2 : (!PFm, !PFm) -> !affine
  // (1,2,1)
  %jacobian1 = elliptic_curve.point %var1, %var2, %var1 : (!PFm, !PFm, !PFm) -> !jacobian

  %jacobian2 = elliptic_curve.add %affine1, %jacobian1 : !affine, !jacobian -> !jacobian
  func.call @printG1Jacobian(%jacobian2) : (!jacobian) -> ()

  %jacobian3 = elliptic_curve.sub %affine1, %jacobian2 : !affine, !jacobian -> !jacobian
  func.call @printG1Jacobian(%jacobian3) : (!jacobian) -> ()

  %jacobian4 = elliptic_curve.negate %jacobian3 : !jacobian
  func.call @printG1Jacobian(%jacobian4) : (!jacobian) -> ()

  %jacobian5 = elliptic_curve.double %jacobian4 : !jacobian -> !jacobian
  func.call @printG1Jacobian(%jacobian5) : (!jacobian) -> ()

  %xyzz1 = elliptic_curve.convert_point_type %affine1 : !affine -> !xyzz
  func.call @printG1Xyzz(%xyzz1) : (!xyzz) -> ()

  %affine2 = elliptic_curve.convert_point_type %xyzz1 : !xyzz -> !affine
  func.call @printG1Affine(%affine2) : (!affine) -> ()

  %jacobian6 = elliptic_curve.scalar_mul %var7, %affine2 : !SFm, !affine -> !jacobian
  %affine2_1 = elliptic_curve.convert_point_type %jacobian6 : !jacobian -> !affine
  %jacobian6_1 = elliptic_curve.convert_point_type %affine2_1 : !affine -> !jacobian
  func.call @printG1Jacobian(%jacobian6_1) : (!jacobian) -> ()

  %affine3 = elliptic_curve.convert_point_type %jacobian6 : !jacobian -> !affine
  func.call @printG1Affine(%affine3) : (!affine) -> ()

  %xyzz2 = elliptic_curve.add %affine3, %xyzz1 : !affine, !xyzz -> !xyzz
  %affine4 = elliptic_curve.convert_point_type %xyzz2 : !xyzz -> !affine
  %xyzz3 = elliptic_curve.convert_point_type %affine4 : !affine -> !xyzz
  func.call @printG1Xyzz(%xyzz3) : (!xyzz) -> ()

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
