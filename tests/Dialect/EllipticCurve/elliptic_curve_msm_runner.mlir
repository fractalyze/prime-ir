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

// RUN: cat %S/../../default_print_utils.mlir %S/../../bn254_field_defs.mlir %S/../../bn254_ec_mont_defs.mlir %S/../../bn254_ec_utils.mlir %s \
// RUN:   | prime-ir-opt -elliptic-curve-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_msm -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext,%S/../../libruntime_functions%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_MSM < %t

// CHECK-LABEL: @test_msm
func.func @test_msm() {
  %c1 = field.constant 1 : !PF
  %c2 = field.constant 2 : !PF
  %var1 = field.to_mont %c1 : !PFm
  %var2 = field.to_mont %c2 : !PFm

  %c3 = field.constant 3 : !SF
  %c5 = field.constant 5 : !SF
  %c7 = field.constant 7 : !SF

  %scalar3 = field.to_mont %c3 : !SFm
  %scalar5 = field.to_mont %c5 : !SFm
  %scalar7 = field.to_mont %c7 : !SFm

  %jacobian1 = elliptic_curve.point %var1, %var2, %var1 : (!PFm, !PFm, !PFm) -> !jacobian
  %jacobian2 = elliptic_curve.double %jacobian1 : !jacobian -> !jacobian
  %jacobian3 = elliptic_curve.double %jacobian2 : !jacobian -> !jacobian
  %jacobian4 = elliptic_curve.double %jacobian3 : !jacobian -> !jacobian


  // CALCULATING TRUE VALUE OF MSM
  %scalar_mul1 = elliptic_curve.scalar_mul %scalar3, %jacobian1 : !SFm, !jacobian -> !jacobian
  %scalar_mul2 = elliptic_curve.scalar_mul %scalar3, %jacobian2 : !SFm, !jacobian -> !jacobian
  %scalar_mul3 = elliptic_curve.scalar_mul %scalar5, %jacobian3 : !SFm, !jacobian -> !jacobian
  %scalar_mul4 = elliptic_curve.scalar_mul %scalar7, %jacobian4 : !SFm, !jacobian -> !jacobian

  %add1 = elliptic_curve.add %scalar_mul1, %scalar_mul2 : !jacobian, !jacobian -> !jacobian
  %add2 = elliptic_curve.add %scalar_mul3, %scalar_mul4 : !jacobian, !jacobian -> !jacobian
  %msm_true = elliptic_curve.add %add1, %add2 : !jacobian, !jacobian -> !jacobian
  func.call @printG1AffineFromJacobian(%msm_true) : (!jacobian) -> ()

  // RUNNING MSM
  %scalars = tensor.from_elements %scalar3, %scalar3, %scalar5, %scalar7 : tensor<4x!SFm>
  %points = tensor.from_elements %jacobian1, %jacobian2, %jacobian3, %jacobian4 : tensor<4x!jacobian>
  %msm_test = elliptic_curve.msm %scalars, %points degree=2 parallel : tensor<4x!SFm>, tensor<4x!jacobian> -> !jacobian
  func.call @printG1AffineFromJacobian(%msm_test) : (!jacobian) -> ()

  return
}

// CHECK_TEST_MSM: [(17990338800136330219282030132013276894006083605897860662265286903319615807158, 3289917882057076627927906556860231277487594098124155054955738043890261042492)]
// CHECK_TEST_MSM: [(17990338800136330219282030132013276894006083605897860662265286903319615807158, 3289917882057076627927906556860231277487594098124155054955738043890261042492)]
