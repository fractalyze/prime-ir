// RUN: zkir-opt %s --mod-arith-to-arith -convert-elementwise-to-linalg --one-shot-bufferize --convert-scf-to-cf --convert-cf-to-llvm --convert-to-llvm \
// RUN:   | mlir-runner -e test_lower_inverse -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_INVERSE < %t

!Fr = !mod_arith.int<2147483647:i32>

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @test_lower_inverse() {
  %p = mod_arith.constant 3723 : !Fr
  %1 = mod_arith.inverse %p : !Fr
  %2 = mod_arith.extract %1 : !Fr -> i32
  %3 = tensor.from_elements %2 : tensor<1xi32>

  %4 = bufferization.to_memref %3 : tensor<1xi32> to memref<1xi32>
  %U = memref.cast %4 : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

// CHECK_TEST_INVERSE: [1324944920]
