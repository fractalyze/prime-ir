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
// RUN:   | mlir-runner -e test_window_reduce -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext,%S/../../libruntime_functions%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_WINDOW_REDUCE < %t

// RUN: cat %S/../../default_print_utils.mlir %S/../../bn254_field_defs.mlir %S/../../bn254_ec_mont_defs.mlir %S/../../bn254_ec_utils.mlir %s \
// RUN:   | prime-ir-opt -elliptic-curve-to-field -field-to-gpu=parallelize-affine \
// RUN:   | mlir-runner -e test_window_reduce -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext,%S/../../libruntime_functions%shlibext,%mlir_lib_dir/libmlir_cuda_runtime%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_WINDOW_REDUCE < %t

func.func @test_window_reduce() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c128 = arith.constant 128 : index

  %k1 = field.constant 1 : !SF
  %k2 = field.constant 2 : !SF
  %k3 = field.constant 3 : !SF

  %pfm0 = field.constant 0 : !PFm

  %affine0 = elliptic_curve.point %pfm0, %pfm0 : (!PFm, !PFm) -> !affine
  %affine1 = func.call @getG1GeneratorMultiple(%k1) : (!SF) -> (!affine)
  %affine2 = func.call @getG1GeneratorMultiple(%k2) : (!SF) -> (!affine)
  %affine3 = func.call @getG1GeneratorMultiple(%k3) : (!SF) -> (!affine)

  %jacobian0 = elliptic_curve.convert_point_type %affine0 : !affine -> !jacobian
  %jacobian1 = elliptic_curve.convert_point_type %affine1 : !affine -> !jacobian
  %jacobian2 = elliptic_curve.convert_point_type %affine2 : !affine -> !jacobian
  %jacobian3 = elliptic_curve.convert_point_type %affine3 : !affine -> !jacobian

  // 1G 2G 3G 0G
  %windows0 = tensor.splat %jacobian0 : tensor<128x!jacobian>
  %windows1 = tensor.insert %jacobian1 into %windows0[%c0] : tensor<128x!jacobian>
  %windows2 = tensor.insert %jacobian2 into %windows1[%c1] : tensor<128x!jacobian>
  %windows3 = tensor.insert %jacobian3 into %windows2[%c2] : tensor<128x!jacobian>
  %windows = tensor.insert %jacobian0 into %windows3[%c3] : tensor<128x!jacobian>

  // Expected output:
  // 1G + 2^2(2G) + 2^4(3G) + 2^6(0G) = 57G

  // 57G
  %k57 = field.constant 57 : !SF
  %true_result = func.call @getG1GeneratorMultiple(%k57) : (!SF) -> (!affine)
  func.call @printG1Affine(%true_result) : (!affine) -> ()

  %result = elliptic_curve.window_reduce %windows {bitsPerWindow = 2 : i16, scalarType = !SF}: (tensor<128x!jacobian>) -> !jacobian
  func.call @printG1AffineFromJacobian(%result) : (!jacobian) -> ()
  return
}

// CHECK_TEST_WINDOW_REDUCE: [(5267322610033386327594727284085617807706598503218388887104616381227512437954, 201257782416518842482277204984225354519663728413732211137155795260901992108)]
// CHECK_TEST_WINDOW_REDUCE: [(5267322610033386327594727284085617807706598503218388887104616381227512437954, 201257782416518842482277204984225354519663728413732211137155795260901992108)]
