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

// RUN: prime-ir-opt %s -poly-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_poly_ntt -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_POLY_NTT < %t

// RUN: prime-ir-opt %s -poly-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_poly_ntt_with_twiddles -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_POLY_NTT_WITH_TWIDDLES < %t

// RUN: prime-ir-opt %s -poly-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_poly_ntt_out_of_place -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_POLY_NTT_OUT_OF_PLACE < %t

// RUN: prime-ir-opt %s -poly-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_poly_ntt_out_of_place_no_bit_reversal -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_POLY_NTT_OUT_OF_PLACE_NO_BIT_REVERSAL < %t

// RUN: prime-ir-opt %s -poly-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_poly_negacyclic_ntt -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_POLY_NEGACYCLIC_NTT < %t

// RUN: prime-ir-opt %s -poly-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_poly_negacyclic_convolution -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_POLY_NEGACYCLIC_CONVOLUTION < %t

!coeff_ty = !field.pf<7681:i32>
!coeff_ty_mont = !field.pf<7681:i32, true>
#root_of_unity = #field.root_of_unity<3383:i32, 4:i32> : !coeff_ty_mont
// psi is a primitive 8th root with psi^2 = 3383, the omega above: a length-4
// negacyclic transform evaluates at the roots of X^4 + 1, which are the odd
// powers of a 2n-th root.
#psi = #field.root_of_unity<1925:i32, 8:i32> : !coeff_ty_mont
!poly_ty = !poly.polynomial<!coeff_ty_mont, 3>

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @test_poly_ntt() {
  %coeffs_mont = field.constant dense<[1, 2, 3, 4]> : tensor<4x!coeff_ty_mont>
  %res = poly.ntt %coeffs_mont into %coeffs_mont {root=#root_of_unity} : tensor<4x!coeff_ty_mont>

  %res_standard = field.from_mont %res : tensor<4x!coeff_ty>
  %extract = field.bitcast %res_standard : tensor<4x!coeff_ty> -> tensor<4xi32>
  %1 = bufferization.to_buffer %extract : tensor<4xi32> to memref<4xi32>
  %U = memref.cast %1 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()

  %intt = poly.ntt %res into %res {root=#root_of_unity} inverse=true : tensor<4x!coeff_ty_mont>
  %poly = poly.from_tensor %intt : tensor<4x!coeff_ty_mont> -> !poly_ty
  %res2 = poly.to_tensor %poly : !poly_ty -> tensor<4x!coeff_ty_mont>
  %res2_standard = field.from_mont %res2 : tensor<4x!coeff_ty>
  %extract2 = field.bitcast %res2_standard : tensor<4x!coeff_ty> -> tensor<4xi32>
  %2= bufferization.to_buffer %extract2 : tensor<4xi32> to memref<4xi32>
  %U2 = memref.cast %2 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%U2) : (memref<*xi32>) -> ()
  return
}
// CHECK_TEST_POLY_NTT: [10, 913, 7679, 6764]
// CHECK_TEST_POLY_NTT: [1, 2, 3, 4]

func.func @test_poly_ntt_with_twiddles() {
  %coeffs_mont = field.constant dense<[1, 2, 3, 4]> : tensor<4x!coeff_ty_mont>
  %twiddles_raw = arith.constant dense<[5569, 6115, 2112, 1566]> : tensor<4xi32>
  %twiddles = field.bitcast %twiddles_raw : tensor<4xi32> -> tensor<4x!coeff_ty_mont>
  %res = poly.ntt %coeffs_mont into %coeffs_mont with %twiddles : tensor<4x!coeff_ty_mont>

  %res_standard = field.from_mont %res : tensor<4x!coeff_ty>
  %extract = field.bitcast %res_standard : tensor<4x!coeff_ty> -> tensor<4xi32>
  %1 = bufferization.to_buffer %extract : tensor<4xi32> to memref<4xi32>
  %U = memref.cast %1 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()

  %inv_twiddles_raw = arith.constant dense<[5569, 1566, 2112, 6115]> : tensor<4xi32>
  %inv_twiddles = field.bitcast %inv_twiddles_raw : tensor<4xi32> -> tensor<4x!coeff_ty_mont>
  %intt = poly.ntt %res into %res with %inv_twiddles inverse=true : tensor<4x!coeff_ty_mont>
  %poly = poly.from_tensor %intt : tensor<4x!coeff_ty_mont> -> !poly_ty
  %res2 = poly.to_tensor %poly : !poly_ty -> tensor<4x!coeff_ty_mont>
  %res2_standard = field.from_mont %res2 : tensor<4x!coeff_ty>
  %extract2 = field.bitcast %res2_standard : tensor<4x!coeff_ty> -> tensor<4xi32>
  %2= bufferization.to_buffer %extract2 : tensor<4xi32> to memref<4xi32>
  %U2 = memref.cast %2 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%U2) : (memref<*xi32>) -> ()
  return
}

// CHECK_TEST_POLY_NTT_WITH_TWIDDLES: [10, 913, 7679, 6764]
// CHECK_TEST_POLY_NTT_WITH_TWIDDLES: [1, 2, 3, 4]

func.func @test_poly_ntt_out_of_place() {
  %coeffs_mont = field.constant dense<[1, 2, 3, 4]> : tensor<4x!coeff_ty_mont>
  %tmp = bufferization.alloc_tensor() : tensor<4x!coeff_ty_mont>
  %res = poly.ntt %coeffs_mont into %tmp {root=#root_of_unity} : tensor<4x!coeff_ty_mont>

  %res_standard = field.from_mont %res : tensor<4x!coeff_ty>
  %extract = field.bitcast %res_standard : tensor<4x!coeff_ty> -> tensor<4xi32>
  %1 = bufferization.to_buffer %extract : tensor<4xi32> to memref<4xi32>
  %U = memref.cast %1 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()

  %tmp1 = bufferization.alloc_tensor() : tensor<4x!coeff_ty_mont>
  %intt = poly.ntt %res into %tmp1 {root=#root_of_unity} inverse=true : tensor<4x!coeff_ty_mont>
  %poly = poly.from_tensor %intt : tensor<4x!coeff_ty_mont> -> !poly_ty
  %res2 = poly.to_tensor %poly : !poly_ty -> tensor<4x!coeff_ty_mont>
  %res2_standard = field.from_mont %res2 : tensor<4x!coeff_ty>
  %extract2 = field.bitcast %res2_standard : tensor<4x!coeff_ty> -> tensor<4xi32>
  %2= bufferization.to_buffer %extract2 : tensor<4xi32> to memref<4xi32>
  %U2 = memref.cast %2 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%U2) : (memref<*xi32>) -> ()
  return
}

// CHECK_TEST_POLY_NTT_OUT_OF_PLACE: [10, 913, 7679, 6764]
// CHECK_TEST_POLY_NTT_OUT_OF_PLACE: [1, 2, 3, 4]

func.func @test_poly_ntt_out_of_place_no_bit_reversal() {
  %coeffs_mont = field.constant dense<[1, 2, 3, 4]> : tensor<4x!coeff_ty_mont>
  %tmp = bufferization.alloc_tensor() : tensor<4x!coeff_ty_mont>
  %res = poly.ntt %coeffs_mont into %tmp {root=#root_of_unity} bit_reverse=false : tensor<4x!coeff_ty_mont>

  %res_standard = field.from_mont %res : tensor<4x!coeff_ty>
  %extract = field.bitcast %res_standard : tensor<4x!coeff_ty> -> tensor<4xi32>
  %1 = bufferization.to_buffer %extract : tensor<4xi32> to memref<4xi32>
  %U = memref.cast %1 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()

  %tmp1 = bufferization.alloc_tensor() : tensor<4x!coeff_ty_mont>
  %intt = poly.ntt %res into %tmp1 {root=#root_of_unity} inverse=true bit_reverse=false : tensor<4x!coeff_ty_mont>
  %poly = poly.from_tensor %intt : tensor<4x!coeff_ty_mont> -> !poly_ty
  %res2 = poly.to_tensor %poly : !poly_ty -> tensor<4x!coeff_ty_mont>
  %res2_standard = field.from_mont %res2 : tensor<4x!coeff_ty>
  %extract2 = field.bitcast %res2_standard : tensor<4x!coeff_ty> -> tensor<4xi32>
  %2= bufferization.to_buffer %extract2 : tensor<4xi32> to memref<4xi32>
  %U2 = memref.cast %2 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%U2) : (memref<*xi32>) -> ()
  return
}

// CHECK_TEST_POLY_NTT_OUT_OF_PLACE_NO_BIT_REVERSAL: [10, 4297, 7677, 3382]
// CHECK_TEST_POLY_NTT_OUT_OF_PLACE_NO_BIT_REVERSAL: [1, 2, 3, 4]

// A negacyclic transform is not the cyclic one: evaluating [1,2,3,4] at the odd
// powers of psi gives values the cyclic case above never produces, and the round
// trip returns the input.
func.func @test_poly_negacyclic_ntt() {
  %coeffs_mont = field.constant dense<[1, 2, 3, 4]> : tensor<4x!coeff_ty_mont>
  %tmp = bufferization.alloc_tensor() : tensor<4x!coeff_ty_mont>
  %res = poly.ntt %coeffs_mont into %tmp {root=#psi} negacyclic=true : tensor<4x!coeff_ty_mont>

  %res_standard = field.from_mont %res : tensor<4x!coeff_ty>
  %extract = field.bitcast %res_standard : tensor<4x!coeff_ty> -> tensor<4xi32>
  %1 = bufferization.to_buffer %extract : tensor<4xi32> to memref<4xi32>
  %U = memref.cast %1 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()

  %tmp1 = bufferization.alloc_tensor() : tensor<4x!coeff_ty_mont>
  %intt = poly.ntt %res into %tmp1 {root=#psi} inverse=true negacyclic=true : tensor<4x!coeff_ty_mont>
  %intt_standard = field.from_mont %intt : tensor<4x!coeff_ty>
  %extract2 = field.bitcast %intt_standard : tensor<4x!coeff_ty> -> tensor<4xi32>
  %2 = bufferization.to_buffer %extract2 : tensor<4xi32> to memref<4xi32>
  %U2 = memref.cast %2 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%U2) : (memref<*xi32>) -> ()
  return
}
// CHECK_TEST_POLY_NEGACYCLIC_NTT: [1467, 2807, 3471, 7621]
// CHECK_TEST_POLY_NEGACYCLIC_NTT: [1, 2, 3, 4]

// The property the transform exists for, and the one a round trip cannot see: a
// pointwise product in the evaluation domain is multiplication in
// Z_q[X]/(X^4 + 1), where the wrap-around comes back negated. [1,2,3,4]*[5,6,7,8]
// has full product [5,16,34,60,61,52,32]; cyclic folds it as c_i + c_{i+4} =
// [66,68,66,60], negacyclic as c_i - c_{i+4} = [-56,-36,2,60].
func.func @test_poly_negacyclic_convolution() {
  %a = field.constant dense<[1, 2, 3, 4]> : tensor<4x!coeff_ty_mont>
  %b = field.constant dense<[5, 6, 7, 8]> : tensor<4x!coeff_ty_mont>
  %ta = bufferization.alloc_tensor() : tensor<4x!coeff_ty_mont>
  %tb = bufferization.alloc_tensor() : tensor<4x!coeff_ty_mont>
  %fa = poly.ntt %a into %ta {root=#psi} negacyclic=true : tensor<4x!coeff_ty_mont>
  %fb = poly.ntt %b into %tb {root=#psi} negacyclic=true : tensor<4x!coeff_ty_mont>

  %prod = field.mul %fa, %fb : tensor<4x!coeff_ty_mont>

  %tc = bufferization.alloc_tensor() : tensor<4x!coeff_ty_mont>
  %conv = poly.ntt %prod into %tc {root=#psi} inverse=true negacyclic=true : tensor<4x!coeff_ty_mont>
  %conv_standard = field.from_mont %conv : tensor<4x!coeff_ty>
  %extract = field.bitcast %conv_standard : tensor<4x!coeff_ty> -> tensor<4xi32>
  %1 = bufferization.to_buffer %extract : tensor<4xi32> to memref<4xi32>
  %U = memref.cast %1 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}
// CHECK_TEST_POLY_NEGACYCLIC_CONVOLUTION: [7625, 7645, 2, 60]
