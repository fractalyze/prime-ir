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
// RUN:   | mlir-runner -e test_bucket_acc -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext,%S/../../libruntime_functions%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_BUCKET_ACC < %t

// RUN: cat %S/../../default_print_utils.mlir %S/../../bn254_field_defs.mlir %S/../../bn254_ec_mont_defs.mlir %S/../../bn254_ec_utils.mlir %s \
// RUN:   | zkir-opt -elliptic-curve-to-field -field-to-gpu=parallelize-affine \
// RUN:   | mlir-runner -e test_bucket_acc -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext,%S/../../libruntime_functions%shlibext,%mlir_lib_dir/libmlir_cuda_runtime%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_BUCKET_ACC < %t

func.func @test_bucket_acc() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index

  %sorted_unique_bucket_indices = tensor.from_elements %c1, %c2, %c5, %c7 : tensor<4xindex>
  %offsets = tensor.from_elements %c0, %c2, %c3, %c5, %c6 : tensor<5xindex>
  %sorted_point_indices = tensor.from_elements %c0, %c2, %c1, %c0, %c1, %c2 : tensor<6xindex>

  %k1 = field.constant 1 : !SF
  %k2 = field.constant 2 : !SF
  %k3 = field.constant 3 : !SF

  %affine1 = func.call @getG1GeneratorMultiple(%k1) : (!SF) -> (!affine)
  %affine2 = func.call @getG1GeneratorMultiple(%k2) : (!SF) -> (!affine)
  %affine3 = func.call @getG1GeneratorMultiple(%k3) : (!SF) -> (!affine)

  %points = tensor.from_elements %affine1, %affine2, %affine3 : tensor<3x!affine>

  %result = elliptic_curve.bucket_acc %points, %sorted_point_indices, %sorted_unique_bucket_indices, %offsets: (tensor<3x!affine>, tensor<6xindex>, tensor<4xindex>, tensor<5xindex>) -> tensor<8x!jacobian>

  %jacobian_sum_13 = elliptic_curve.add %affine1, %affine3 : !affine, !affine -> !jacobian
  func.call @printG1AffineFromJacobian(%jacobian_sum_13) : (!jacobian) -> ()

  %jacobian_sum_12 = elliptic_curve.add %affine1, %affine2 : !affine, !affine -> !jacobian
  func.call @printG1AffineFromJacobian(%jacobian_sum_12) : (!jacobian) -> ()

  func.call @printG1Affine(%affine2) : (!affine) -> ()
  func.call @printG1Affine(%affine3) : (!affine) -> ()

  scf.for %i = %c0 to %c8 step %c1 {
    %point = tensor.extract %result[%i] : tensor<8x!jacobian>
    func.call @printG1AffineFromJacobian(%point) : (!jacobian) -> ()
    scf.yield
  }
  return
}

// CHECK_TEST_BUCKET_ACC: [(3010198690406615200373504922352659861758983907867017329644089018310584441462, 4027184618003122424972590350825261965929648733675738730716654005365300998076)]
// CHECK_TEST_BUCKET_ACC: [(3353031288059533942658390886683067124040920775575537747144343083137631628272, 19321533766552368860946552437480515441416830039777911637913418824951667761761)]
// CHECK_TEST_BUCKET_ACC: [(1368015179489954701390400359078579693043519447331113978918064868415326638035, 9918110051302171585080402603319702774565515993150576347155970296011118125764)]
// CHECK_TEST_BUCKET_ACC: [(3353031288059533942658390886683067124040920775575537747144343083137631628272, 19321533766552368860946552437480515441416830039777911637913418824951667761761)]

// CHECK_TEST_BUCKET_ACC: [(0, 0)]
// CHECK_TEST_BUCKET_ACC: [(3010198690406615200373504922352659861758983907867017329644089018310584441462, 4027184618003122424972590350825261965929648733675738730716654005365300998076)]
// CHECK_TEST_BUCKET_ACC: [(1368015179489954701390400359078579693043519447331113978918064868415326638035, 9918110051302171585080402603319702774565515993150576347155970296011118125764)]
// CHECK_TEST_BUCKET_ACC: [(0, 0)]
// CHECK_TEST_BUCKET_ACC: [(0, 0)]
// CHECK_TEST_BUCKET_ACC: [(3353031288059533942658390886683067124040920775575537747144343083137631628272, 19321533766552368860946552437480515441416830039777911637913418824951667761761)]
// CHECK_TEST_BUCKET_ACC: [(0, 0)]
// CHECK_TEST_BUCKET_ACC: [(3353031288059533942658390886683067124040920775575537747144343083137631628272, 19321533766552368860946552437480515441416830039777911637913418824951667761761)]
