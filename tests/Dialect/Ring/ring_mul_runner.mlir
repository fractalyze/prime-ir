// Numeric exec: ring.mul = negacyclic polynomial product in Z_17[X]/(X^4+1).
// q = 17 admits an 8th (2N) root of unity (8 | 16), N = 4.

// RUN: prime-ir-opt %s -ring-to-mod-arith -poly-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_mul_wrap -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s --check-prefix=WRAP < %t
// RUN: prime-ir-opt %s -ring-to-mod-arith -poly-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_mul_simple -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t2
// RUN: FileCheck %s --check-prefix=SIMPLE < %t2

func.func private @printMemrefI64(memref<*xi64>) attributes { llvm.emit_c_interface }

// X^2 * X^2 = X^4 = -1 (mod X^4+1) = 16 mod 17. Negacyclic wrap -> [16,0,0,0]
// (a cyclic product would wrongly give X^0 = [1,0,0,0]).
func.func @test_mul_wrap() {
  %a = arith.constant dense<[[0, 0, 1, 0]]> : tensor<1x4xi64>
  %b = arith.constant dense<[[0, 0, 1, 0]]> : tensor<1x4xi64>
  %ra = ring.from_tensor %a : tensor<1x4xi64> to !ring.rq<[17], 4 : i32>
  %rb = ring.from_tensor %b : tensor<1x4xi64> to !ring.rq<[17], 4 : i32>
  %rc = ring.mul %ra, %rb : !ring.rq<[17], 4 : i32>
  %ot = ring.to_tensor %rc : !ring.rq<[17], 4 : i32> to tensor<1x4xi64>
  %m = bufferization.to_buffer %ot : tensor<1x4xi64> to memref<1x4xi64>
  %U = memref.cast %m : memref<1x4xi64> to memref<*xi64>
  // WRAP: [16, 0, 0, 0]
  func.call @printMemrefI64(%U) : (memref<*xi64>) -> ()
  return
}

// (1 + X)(1 + X) = 1 + 2X + X^2 (no wrap) -> [1, 2, 1, 0].
func.func @test_mul_simple() {
  %a = arith.constant dense<[[1, 1, 0, 0]]> : tensor<1x4xi64>
  %b = arith.constant dense<[[1, 1, 0, 0]]> : tensor<1x4xi64>
  %ra = ring.from_tensor %a : tensor<1x4xi64> to !ring.rq<[17], 4 : i32>
  %rb = ring.from_tensor %b : tensor<1x4xi64> to !ring.rq<[17], 4 : i32>
  %rc = ring.mul %ra, %rb : !ring.rq<[17], 4 : i32>
  %ot = ring.to_tensor %rc : !ring.rq<[17], 4 : i32> to tensor<1x4xi64>
  %m = bufferization.to_buffer %ot : tensor<1x4xi64> to memref<1x4xi64>
  %U = memref.cast %m : memref<1x4xi64> to memref<*xi64>
  // SIMPLE: [1, 2, 1, 0]
  func.call @printMemrefI64(%U) : (memref<*xi64>) -> ()
  return
}
