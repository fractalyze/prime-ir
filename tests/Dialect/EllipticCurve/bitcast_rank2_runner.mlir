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

// Point/field reinterpret carried all the way to a memref descriptor. The op
// always changes extent — N field elements against N/K points — so the output
// descriptor has to be rebuilt rather than forwarded.
//
// Reading the result back with tensor.extract, as bitcast_runner.mlir does,
// never forms a descriptor and so cannot observe any of this; printing the
// buffer does. A rank-2 result is what makes the sizes and strides both
// observable, since a rank-1 descriptor has no stride to get wrong.
//
// The bitcast sits in its own function so its operand arrives as a real memref
// instead of folding into the constant.

// RUN: cat %S/../../default_print_utils.mlir %S/../../bn254_defs.mlir %s \
// RUN:   | prime-ir-opt -elliptic-curve-to-field \
// RUN:       --field-to-llvm='bufferize-function-boundaries=true' \
// RUN:   | mlir-runner -e test_bitcast_rank2 -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext,%S/../../libruntime_functions%shlibext" > %t
// RUN: FileCheck %s < %t

// G1 jacobian: 3 coordinates over a prime field, so K = 3 and a row of 2 points
// is a row of 6 field elements.
func.func @bitcast_points_to_fields_rank2(
    %src: tensor<2x2x!jacobianm>) -> tensor<2x6x!PFm> {
  %f = elliptic_curve.bitcast %src : tensor<2x2x!jacobianm> -> tensor<2x6x!PFm>
  return %f : tensor<2x6x!PFm>
}

func.func @test_bitcast_rank2() {
  // Rows are reversals of each other, so a row read from the wrong base offset
  // prints a visibly different sequence rather than a plausible one.
  %pts = elliptic_curve.constant dense<[[[1, 2, 3], [4, 5, 6]],
                                        [[6, 5, 4], [3, 2, 1]]]>
      : tensor<2x2x!jacobianm>
  %fields = func.call @bitcast_points_to_fields_rank2(%pts)
      : (tensor<2x2x!jacobianm>) -> tensor<2x6x!PFm>
  %ints = field.bitcast %fields : tensor<2x6x!PFm> -> tensor<2x6xi256>
  %m = bufferization.to_buffer %ints : tensor<2x6xi256> to memref<2x6xi256>
  %u = memref.cast %m : memref<2x6xi256> to memref<*xi256>
  func.call @printMemrefI256(%u) : (memref<*xi256>) -> ()
  // CHECK: {{\[}}[1, 2, 3, 4, 5, 6]
  // CHECK: [6, 5, 4, 3, 2, 1]
  return
}
