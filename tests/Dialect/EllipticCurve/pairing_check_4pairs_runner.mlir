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

// RUN: cat %S/../../default_print_utils.mlir %S/../../bn254_defs.mlir %S/../../bn254_ec_mont_helpers.mlir %S/../../bn254_ec_utils.mlir %s \
// RUN:   | prime-ir-opt -elliptic-curve-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_pairing_check_4pairs -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext,%S/../../libruntime_functions%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_PAIRING_4 < %t

func.func @printI1(%val: i1) {
  %ext = arith.extui %val : i1 to i32
  %vec = vector.from_elements %ext : vector<1xi32>
  %mem = memref.alloc() : memref<1xi32>
  %c0 = arith.constant 0 : index
  vector.store %vec, %mem[%c0] : memref<1xi32>, vector<1xi32>
  %cast = memref.cast %mem : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%cast) : (memref<*xi32>) -> ()
  return
}

// 4-pair bilinearity check:
//   e(2·G1, 3·G2) * e(6·G1, -G2) * e(4·G1, 5·G2) * e(20·G1, -G2) = 1
// Since 2*3 + 6*(-1) + 4*5 + 20*(-1) = 6 - 6 + 20 - 20 = 0.
// CHECK-LABEL: @test_pairing_check_4pairs
func.func @test_pairing_check_4pairs() {
  // G1 generator: (1, 2)
  %g1x = field.constant 1 : !PFm
  %g1y = field.constant 2 : !PFm
  %g1 = elliptic_curve.from_coords %g1x, %g1y : (!PFm, !PFm) -> !affinem

  // G2 generator coordinates
  %g2x = field.constant [10857046999023057135944570762232829481370756359578518086990519993285655852781,
                          11559732032986387107991004021392285783925812861821192530917403151452391805634] : !QFm
  %g2y = field.constant [8495653923123431417604973247489272438418190587263600148770280649306958101930,
                          4082367875863433681332203403145435568316851327593401208105741076214120093531] : !QFm
  %g2 = elliptic_curve.from_coords %g2x, %g2y : (!QFm, !QFm) -> !g2affinem

  // Scalars
  %s2 = field.constant 2 : !SFm
  %s3 = field.constant 3 : !SFm
  %s4 = field.constant 4 : !SFm
  %s5 = field.constant 5 : !SFm
  %s6 = field.constant 6 : !SFm
  %s20 = field.constant 20 : !SFm

  // P1 = 2·G1 (affine)
  %p1_jac = elliptic_curve.scalar_mul %s2, %g1 : !SFm, !affinem -> !jacobianm
  %p1 = elliptic_curve.convert_point_type %p1_jac : !jacobianm -> !affinem

  // P2 = 6·G1 (affine)
  %p2_jac = elliptic_curve.scalar_mul %s6, %g1 : !SFm, !affinem -> !jacobianm
  %p2 = elliptic_curve.convert_point_type %p2_jac : !jacobianm -> !affinem

  // P3 = 4·G1 (affine)
  %p3_jac = elliptic_curve.scalar_mul %s4, %g1 : !SFm, !affinem -> !jacobianm
  %p3 = elliptic_curve.convert_point_type %p3_jac : !jacobianm -> !affinem

  // P4 = 20·G1 (affine)
  %p4_jac = elliptic_curve.scalar_mul %s20, %g1 : !SFm, !affinem -> !jacobianm
  %p4 = elliptic_curve.convert_point_type %p4_jac : !jacobianm -> !affinem

  // Q1 = 3·G2 (affine)
  %q1_jac = elliptic_curve.scalar_mul %s3, %g2 : !SFm, !g2affinem -> !g2jacobianm
  %q1 = elliptic_curve.convert_point_type %q1_jac : !g2jacobianm -> !g2affinem

  // Q2 = -G2 (affine)
  %neg_g2 = elliptic_curve.negate %g2 : !g2affinem

  // Q3 = 5·G2 (affine)
  %q3_jac = elliptic_curve.scalar_mul %s5, %g2 : !SFm, !g2affinem -> !g2jacobianm
  %q3 = elliptic_curve.convert_point_type %q3_jac : !g2jacobianm -> !g2affinem

  // Q4 = -G2 (affine) (same as Q2)
  // Use a fresh negate to ensure independent value
  %neg_g2_2 = elliptic_curve.negate %g2 : !g2affinem

  // Build tensors
  %g1_pts = tensor.from_elements %p1, %p2, %p3, %p4 : tensor<4x!affinem>
  %g2_pts = tensor.from_elements %q1, %neg_g2, %q3, %neg_g2_2 : tensor<4x!g2affinem>

  // Pairing check: should be true (product = 1)
  %result = elliptic_curve.pairing_check %g1_pts, %g2_pts
      : tensor<4x!affinem>, tensor<4x!g2affinem> -> i1

  func.call @printI1(%result) : (i1) -> ()
  return
}
// CHECK_PAIRING_4: [1]
