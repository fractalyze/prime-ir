// Numeric exec: ring.mul in the evaluation basis of Z_17[X]/(X^4+1) is the
// pointwise product of the evaluations, reduced mod 17. N = 4, and 2N = 8
// divides 16, so the ring does split and the eval basis exists.

// RUN: prime-ir-opt %s -ring-to-mod-arith -field-to-llvm \
// RUN:   | mlir-runner -e test_mul_reduces -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s --check-prefix=REDUCE < %t
// RUN: prime-ir-opt %s -ring-to-mod-arith -field-to-llvm \
// RUN:   | mlir-runner -e test_mul_in_range -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t2
// RUN: FileCheck %s --check-prefix=INRANGE < %t2

!Rq = !ring.rq<[17], 4 : i32, eval>

func.func private @printMemrefI64(memref<*xi64>) attributes { llvm.emit_c_interface }

// Every slot overflows the modulus, so this exercises the reducer:
// [5,7,13,16] * [4,9,2,16] = [20,63,26,256] = [3,12,9,1] mod 17.
func.func @test_mul_reduces() {
  %a = arith.constant dense<[[5, 7, 13, 16]]> : tensor<1x4xi64>
  %b = arith.constant dense<[[4, 9, 2, 16]]> : tensor<1x4xi64>
  %ra = ring.from_tensor %a : tensor<1x4xi64> to !Rq
  %rb = ring.from_tensor %b : tensor<1x4xi64> to !Rq
  %rc = ring.mul %ra, %rb : !Rq
  %out = ring.to_tensor %rc : !Rq to tensor<1x4xi64>
  %m = bufferization.to_buffer %out : tensor<1x4xi64> to memref<1x4xi64>
  %U = memref.cast %m : memref<1x4xi64> to memref<*xi64>
  // REDUCE: [3, 12, 9, 1]
  func.call @printMemrefI64(%U) : (memref<*xi64>) -> ()
  return
}

// Products below the modulus must come back untouched — a reducer that always
// subtracts would still pass the case above.
func.func @test_mul_in_range() {
  %a = arith.constant dense<[[1, 2, 3, 4]]> : tensor<1x4xi64>
  %b = arith.constant dense<[[1, 1, 2, 2]]> : tensor<1x4xi64>
  %ra = ring.from_tensor %a : tensor<1x4xi64> to !Rq
  %rb = ring.from_tensor %b : tensor<1x4xi64> to !Rq
  %rc = ring.mul %ra, %rb : !Rq
  %out = ring.to_tensor %rc : !Rq to tensor<1x4xi64>
  %m = bufferization.to_buffer %out : tensor<1x4xi64> to memref<1x4xi64>
  %U = memref.cast %m : memref<1x4xi64> to memref<*xi64>
  // INRANGE: [1, 2, 6, 8]
  func.call @printMemrefI64(%U) : (memref<*xi64>) -> ()
  return
}
