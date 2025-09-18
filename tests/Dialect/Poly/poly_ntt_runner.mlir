// RUN: zkir-opt %s -poly-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_poly_ntt -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_POLY_NTT < %t

// RUN: zkir-opt %s -poly-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_poly_ntt_with_twiddles -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_POLY_NTT_WITH_TWIDDLES < %t

// RUN: zkir-opt %s -poly-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_poly_ntt_out_of_place -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_POLY_NTT_OUT_OF_PLACE < %t

// RUN: zkir-opt %s -poly-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_poly_ntt_out_of_place_no_bit_reversal -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_POLY_NTT_OUT_OF_PLACE_NO_BIT_REVERSAL < %t

!coeff_ty = !field.pf<7681:i32>
!coeff_ty_mont = !field.pf<7681:i32, true>
#elem = #field.pf.elem<3383:i32>  : !coeff_ty
#root_of_unity = #field.root_of_unity<#elem, 4:i32>
!poly_ty = !poly.polynomial<!coeff_ty_mont, 3>

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @test_poly_ntt() {
  %coeffs_raw = arith.constant dense<[1,2,3,4]> : tensor<4xi32>
  %coeffs = field.bitcast %coeffs_raw : tensor<4xi32> -> tensor<4x!coeff_ty>
  %coeffs_mont = field.to_mont %coeffs : tensor<4x!coeff_ty_mont>
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
  %coeffs_raw = arith.constant dense<[1,2,3,4]> : tensor<4xi32>
  %coeffs = field.bitcast %coeffs_raw : tensor<4xi32> -> tensor<4x!coeff_ty>
  %coeffs_mont = field.to_mont %coeffs : tensor<4x!coeff_ty_mont>
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
  %coeffs_raw = arith.constant dense<[1,2,3,4]> : tensor<4xi32>
  %coeffs = field.bitcast %coeffs_raw : tensor<4xi32> -> tensor<4x!coeff_ty>
  %coeffs_mont = field.to_mont %coeffs : tensor<4x!coeff_ty_mont>
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
  %coeffs_raw = arith.constant dense<[1,2,3,4]> : tensor<4xi32>
  %coeffs = field.bitcast %coeffs_raw : tensor<4xi32> -> tensor<4x!coeff_ty>
  %coeffs_mont = field.to_mont %coeffs : tensor<4x!coeff_ty_mont>
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
