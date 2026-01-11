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
// RUN:   | prime-ir-opt -convert-elementwise-to-linalg -sparsification-and-bufferization -elliptic-curve-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_bucket_acc_csr -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext,%S/../../libruntime_functions%shlibext" > %t > %t
// RUN: FileCheck %s -check-prefix=CHECK_BUCKET_ACC_CSR < %t

// CSR encoding
#CSR = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : dense, d1 : compressed)}>

// Affine maps for linalg operations
#accesses = [
  affine_map<(i,j) -> (i, j)>,
  affine_map<(i, j) -> (i)>
]

#attrs = {
  indexing_maps = #accesses,
  iterator_types = ["parallel", "reduction"]
}

func.func @test_bucket_acc_csr() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index

  %k1 = field.constant 1 : !SF
  %k2 = field.constant 2 : !SF
  %k3 = field.constant 3 : !SF

  %affine1 = func.call @getG1GeneratorMultiple(%k1) : (!SF) -> (!affine)
  %affine2 = func.call @getG1GeneratorMultiple(%k2) : (!SF) -> (!affine)
  %affine3 = func.call @getG1GeneratorMultiple(%k3) : (!SF) -> (!affine)


  %jacobian_sum_13 = elliptic_curve.add %affine1, %affine3 : !affine, !affine -> !jacobian
  func.call @printG1AffineFromJacobian(%jacobian_sum_13) : (!jacobian) -> ()

  %jacobian_sum_12 = elliptic_curve.add %affine1, %affine2 : !affine, !affine -> !jacobian
  func.call @printG1AffineFromJacobian(%jacobian_sum_12) : (!jacobian) -> ()

  func.call @printG1Affine(%affine2) : (!affine) -> ()
  func.call @printG1Affine(%affine3) : (!affine) -> ()

  // Convert all affine points to jacobian before creating sparse matrix
  %jacobian1 = elliptic_curve.convert_point_type %affine1 : !affine -> !jacobian
  %jacobian2 = elliptic_curve.convert_point_type %affine2 : !affine -> !jacobian
  %jacobian3 = elliptic_curve.convert_point_type %affine3 : !affine -> !jacobian
  %points = tensor.from_elements %jacobian1, %jacobian2, %jacobian3 : tensor<3x!jacobian>

  // Create sparse CSR matrix representation
  // Bucket 1: points [0,2]
  // Bucket 2: points [1]
  // Bucket 5: points [0,1]
  // Bucket 7: points [2]

  // CSR values: points in each bucket
  %csr_values = tensor.from_elements %jacobian1, %jacobian3, %jacobian2, %jacobian1, %jacobian2, %jacobian3 : tensor<6x!jacobian>
  // CSR row pointers: 8 buckets + 1 = 9 elements
  %csr_pos = arith.constant dense<[0, 0, 2, 3, 3, 3, 5, 5, 6]> : tensor<9xindex>
  // CSR column indices: which points go into each bucket (doesn't matter)
  %csr_indices = arith.constant dense<[0, 2, 1, 0, 1, 2]> : tensor<6xindex>

  // Assemble sparse CSR matrix: buckets x points (now all jacobian)
  %sparse_matrix = sparse_tensor.assemble (%csr_pos, %csr_indices), %csr_values
    : (tensor<9xindex>, tensor<6xindex>), tensor<6x!jacobian> to tensor<8x3x!jacobian, #CSR>

  // Create zero point for jacobian accumulation
  %zeroPF = field.constant 0 : !PFm
  %onePF = field.constant 1 : !PFm
  %zero_jacobian = elliptic_curve.point %onePF, %onePF, %zeroPF : (!PFm, !PFm, !PFm) -> !jacobian

  // Fill tensor with zero points
  %bucket_results = tensor.empty() : tensor<8x!jacobian>
  %filled_buckets = linalg.fill ins(%zero_jacobian : !jacobian) outs(%bucket_results : tensor<8x!jacobian>) -> tensor<8x!jacobian>

  // Use linalg.generic with sparse_tensor.reduce for proper EC reduction
  %result = linalg.generic #attrs
    ins(%sparse_matrix : tensor<8x3x!jacobian, #CSR>)
    outs(%filled_buckets : tensor<8x!jacobian>) {
  ^bb0(%point: !jacobian, %accumulator: !jacobian):
    // Use sparse_tensor.reduce with same types (jacobian + jacobian -> jacobian)
    %result = sparse_tensor.reduce %point, %accumulator, %zero_jacobian
 : !jacobian {
      ^bb0(%p: !jacobian, %acc: !jacobian):
        %sum = elliptic_curve.add %acc, %p : !jacobian, !jacobian -> !jacobian
        sparse_tensor.yield %sum : !jacobian
    }
    linalg.yield %result : !jacobian
  } -> tensor<8x!jacobian>

  // Print results for comparison with bucket_acc
  scf.for %i = %c0 to %c8 step %c1 {
    %bucket_point = tensor.extract %result[%i] : tensor<8x!jacobian>
    func.call @printG1AffineFromJacobian(%bucket_point) : (!jacobian) -> ()
    scf.yield
  }
  return
}

// CHECK_BUCKET_ACC_CSR: [(3010198690406615200373504922352659861758983907867017329644089018310584441462, 4027184618003122424972590350825261965929648733675738730716654005365300998076)]
// CHECK_BUCKET_ACC_CSR: [(3353031288059533942658390886683067124040920775575537747144343083137631628272, 19321533766552368860946552437480515441416830039777911637913418824951667761761)]
// CHECK_BUCKET_ACC_CSR: [(1368015179489954701390400359078579693043519447331113978918064868415326638035, 9918110051302171585080402603319702774565515993150576347155970296011118125764)]
// CHECK_BUCKET_ACC_CSR: [(3353031288059533942658390886683067124040920775575537747144343083137631628272, 19321533766552368860946552437480515441416830039777911637913418824951667761761)]

// CHECK_BUCKET_ACC_CSR: [(0, 0)]
// CHECK_BUCKET_ACC_CSR: [(3010198690406615200373504922352659861758983907867017329644089018310584441462, 4027184618003122424972590350825261965929648733675738730716654005365300998076)]
// CHECK_BUCKET_ACC_CSR: [(1368015179489954701390400359078579693043519447331113978918064868415326638035, 9918110051302171585080402603319702774565515993150576347155970296011118125764)]
// CHECK_BUCKET_ACC_CSR: [(0, 0)]
// CHECK_BUCKET_ACC_CSR: [(0, 0)]
// CHECK_BUCKET_ACC_CSR: [(3353031288059533942658390886683067124040920775575537747144343083137631628272, 19321533766552368860946552437480515441416830039777911637913418824951667761761)]
// CHECK_BUCKET_ACC_CSR: [(0, 0)]
// CHECK_BUCKET_ACC_CSR: [(3353031288059533942658390886683067124040920775575537747144343083137631628272, 19321533766552368860946552437480515441416830039777911637913418824951667761761)]
