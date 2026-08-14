// Numeric exec: ring.automorphism sigma_3 (X -> X^3) in Z_17[X]/(X^4+1).
// a = 1 + 2X + 3X^2 + 4X^3.  sigma_3(a) = 1 + 2X^3 + 3X^6 + 4X^9.
//   X^6 = -X^2, X^9 = X  ->  1 + 4X - 3X^2 + 2X^3 = [1, 4, 14, 2] mod 17.

// RUN: prime-ir-opt %s -ring-to-mod-arith -field-to-llvm \
// RUN:   | mlir-runner -e test_auto -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

func.func private @printMemrefI64(memref<*xi64>) attributes { llvm.emit_c_interface }

func.func @test_auto() {
  %a = arith.constant dense<[[1, 2, 3, 4]]> : tensor<1x4xi64>
  %ra = ring.from_tensor %a : tensor<1x4xi64> to !ring.rq<[17], 4 : i32>
  %rc = ring.automorphism %ra {exponent = 3 : i64} : !ring.rq<[17], 4 : i32>
  %out = ring.to_tensor %rc : !ring.rq<[17], 4 : i32> to tensor<1x4xi64>
  %m = bufferization.to_buffer %out : tensor<1x4xi64> to memref<1x4xi64>
  %U = memref.cast %m : memref<1x4xi64> to memref<*xi64>
  // CHECK: [1, 4, 14, 2]
  func.call @printMemrefI64(%U) : (memref<*xi64>) -> ()
  return
}
