// RUN: zkir-opt %s -poly-to-llvm \
// RUN:   | mlir-runner -e test_poly_ntt -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_POLY_NTT < %t

!coeff_ty = !field.pf<7681:i32>
#elem = #field.pf_elem<3383:i32>  : !coeff_ty
#inv_elem = #field.pf_elem<4298:i32>  : !coeff_ty
#root = #poly.primitive_root<root=#elem, degree=4 :i32>
!poly_ty = !poly.polynomial<!coeff_ty, 3>

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @test_poly_ntt() {
  %coeffsRaw = arith.constant dense<[1,2,3,4]> : tensor<4xi32>
  %coeffs = field.pf.encapsulate %coeffsRaw : tensor<4xi32> -> tensor<4x!coeff_ty>
  %res = poly.ntt %coeffs {root=#root} : tensor<4x!coeff_ty>

  %extract = field.pf.extract %res : tensor<4x!coeff_ty> -> tensor<4xi32>
  %1 = bufferization.to_memref %extract : tensor<4xi32> to memref<4xi32>
  %U = memref.cast %1 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()

  %intt = poly.intt %res {root=#root} : tensor<4x!coeff_ty>
  %poly = poly.from_tensor %intt : tensor<4x!coeff_ty> -> !poly_ty
  %res2 = poly.to_tensor %poly : !poly_ty -> tensor<4x!coeff_ty>
  %extract2 = field.pf.extract %res2 : tensor<4x!coeff_ty> -> tensor<4xi32>
  %2= bufferization.to_memref %extract2 : tensor<4xi32> to memref<4xi32>
  %U2 = memref.cast %2 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%U2) : (memref<*xi32>) -> ()
  return
}
// CHECK_TEST_POLY_NTT: [10, 913, 7679, 6764]
// CHECK_TEST_POLY_NTT: [1, 2, 3, 4]
