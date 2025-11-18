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

// RUN: cat %S/../../default_print_utils.mlir %S/../../bn254_field_defs.mlir %S/../../bn254_ec_mont_defs.mlir %S/../../bn254_ec_mont_utils.mlir %s \
// RUN:   | zkir-opt -elliptic-curve-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_bucket_reduce -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext,%S/../../printI256%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_BUCKET_REDUCE < %t

// RUN: cat %S/../../default_print_utils.mlir %S/../../bn254_field_defs.mlir %S/../../bn254_ec_mont_defs.mlir %S/../../bn254_ec_mont_utils.mlir %s \
// RUN:   | zkir-opt -elliptic-curve-to-field -field-to-gpu=parallelize-affine \
// RUN:   | mlir-runner -e test_bucket_reduce -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext,%S/../../printI256%shlibext,%mlir_lib_dir/libmlir_cuda_runtime%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_BUCKET_REDUCE < %t

func.func @test_bucket_reduce() {
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : index
  %i2 = arith.constant 2 : index
  %i3 = arith.constant 3 : index

  %k1 = field.constant 1 : !SF
  %k2 = field.constant 2 : !SF
  %k3 = field.constant 3 : !SF

  %pfm0 = field.constant 0 : !PFm

  %affine0 = elliptic_curve.point %pfm0, %pfm0 : (!PFm, !PFm) -> !affine
  %affine1 = func.call @getGeneratorMultiple(%k1) : (!SF) -> (!affine)
  %affine2 = func.call @getGeneratorMultiple(%k2) : (!SF) -> (!affine)
  %affine3 = func.call @getGeneratorMultiple(%k3) : (!SF) -> (!affine)

  %jacobian0 = elliptic_curve.convert_point_type %affine0 : !affine -> !jacobian
  %jacobian1 = elliptic_curve.convert_point_type %affine1 : !affine -> !jacobian
  %jacobian2 = elliptic_curve.convert_point_type %affine2 : !affine -> !jacobian
  %jacobian3 = elliptic_curve.convert_point_type %affine3 : !affine -> !jacobian

  // 1 2 3 0
  // 0 0 2 0
  // 1 3 0 0
  // 3 0 2 1
  %points = tensor.from_elements %jacobian1, %jacobian2, %jacobian3, %jacobian0, %jacobian0, %jacobian0, %jacobian2, %jacobian0, %jacobian1, %jacobian3, %jacobian0, %jacobian0, %jacobian3, %jacobian0, %jacobian2, %jacobian1 : tensor<4x4x!jacobian>

  // Expected output:
  // [(1*2 + 2*3)G, 2*2G, 1*3G, (2*2 + 3*1)G] = [8G, 4G, 3G, 7G]
  %result = elliptic_curve.bucket_reduce %points {scalarType = !SF}: (tensor<4x4x!jacobian>) -> tensor<4x!jacobian>

  // 8G
  %k8 = field.constant 8 : !SF
  %result0 = func.call @getGeneratorMultiple(%k8) : (!SF) -> (!affine)
  func.call @printAffine(%result0) : (!affine) -> ()
  %result_zero = tensor.extract %result[%i0] : tensor<4x!jacobian>
  func.call @printAffineFromJacobian(%result_zero) : (!jacobian) -> ()

  // 4G
  %k4 = field.constant 4 : !SF
  %result1 = func.call @getGeneratorMultiple(%k4) : (!SF) -> (!affine)
  func.call @printAffine(%result1) : (!affine) -> ()
  %result_one = tensor.extract %result[%i1] : tensor<4x!jacobian>
  func.call @printAffineFromJacobian(%result_one) : (!jacobian) -> ()

  // 3G
  %result2 = func.call @getGeneratorMultiple(%k3) : (!SF) -> (!affine)
  func.call @printAffine(%result2) : (!affine) -> ()
  %result_two = tensor.extract %result[%i2] : tensor<4x!jacobian>
  func.call @printAffineFromJacobian(%result_two) : (!jacobian) -> ()

  // 7G
  %k7 = field.constant 7 : !SF
  %result3 = func.call @getGeneratorMultiple(%k7) : (!SF) -> (!affine)
  func.call @printAffine(%result3) : (!affine) -> ()
  %result_three = tensor.extract %result[%i3] : tensor<4x!jacobian>
  func.call @printAffineFromJacobian(%result_three) : (!jacobian) -> ()
  return
}

// CHECK_TEST_BUCKET_REDUCE: [3932705576657793550893430333273221375907985235130430286685735064194643946083, 18813763293032256545937756946359266117037834559191913266454084342712532869153]
// CHECK_TEST_BUCKET_REDUCE: [3932705576657793550893430333273221375907985235130430286685735064194643946083, 18813763293032256545937756946359266117037834559191913266454084342712532869153]

// CHECK_TEST_BUCKET_REDUCE: [3010198690406615200373504922352659861758983907867017329644089018310584441462, 4027184618003122424972590350825261965929648733675738730716654005365300998076]
// CHECK_TEST_BUCKET_REDUCE: [3010198690406615200373504922352659861758983907867017329644089018310584441462, 4027184618003122424972590350825261965929648733675738730716654005365300998076]

// CHECK_TEST_BUCKET_REDUCE: [3353031288059533942658390886683067124040920775575537747144343083137631628272, 19321533766552368860946552437480515441416830039777911637913418824951667761761]
// CHECK_TEST_BUCKET_REDUCE: [3353031288059533942658390886683067124040920775575537747144343083137631628272, 19321533766552368860946552437480515441416830039777911637913418824951667761761]

// CHECK_TEST_BUCKET_REDUCE: [10415861484417082502655338383609494480414113902179649885744799961447382638712, 10196215078179488638353184030336251401353352596818396260819493263908881608606]
// CHECK_TEST_BUCKET_REDUCE: [10415861484417082502655338383609494480414113902179649885744799961447382638712, 10196215078179488638353184030336251401353352596818396260819493263908881608606]
