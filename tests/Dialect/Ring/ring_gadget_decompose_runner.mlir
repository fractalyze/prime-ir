// Numeric exec: ring.gadget_decompose, base 2^2 = 4, 3 levels. Basis [17], N=2.
//   11 = 3 + 2*4 + 0*16  -> digits (3, 2, 0)
//    6 = 2 + 1*4 + 0*16  -> digits (2, 1, 0)
// so digit_0 = [3,2], digit_1 = [2,1], digit_2 = [0,0].

// RUN: prime-ir-opt %s -ring-to-mod-arith -field-to-llvm \
// RUN:   | mlir-runner -e test_gadget -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

func.func private @printMemrefI64(memref<*xi64>) attributes { llvm.emit_c_interface }

func.func @test_gadget() {
  %x = arith.constant dense<[[11, 6]]> : tensor<1x2xi64>
  %rx = ring.from_tensor %x : tensor<1x2xi64> to !ring.rq<[17], 2 : i32>
  %d0, %d1, %d2 = ring.gadget_decompose %rx {baseBits = 2 : i64, levels = 3 : i64}
      : !ring.rq<[17], 2 : i32>
      -> !ring.rq<[17], 2 : i32>, !ring.rq<[17], 2 : i32>, !ring.rq<[17], 2 : i32>
  %t0 = ring.to_tensor %d0 : !ring.rq<[17], 2 : i32> to tensor<1x2xi64>
  %t1 = ring.to_tensor %d1 : !ring.rq<[17], 2 : i32> to tensor<1x2xi64>
  %t2 = ring.to_tensor %d2 : !ring.rq<[17], 2 : i32> to tensor<1x2xi64>
  %m0 = bufferization.to_buffer %t0 : tensor<1x2xi64> to memref<1x2xi64>
  %m1 = bufferization.to_buffer %t1 : tensor<1x2xi64> to memref<1x2xi64>
  %m2 = bufferization.to_buffer %t2 : tensor<1x2xi64> to memref<1x2xi64>
  %U0 = memref.cast %m0 : memref<1x2xi64> to memref<*xi64>
  %U1 = memref.cast %m1 : memref<1x2xi64> to memref<*xi64>
  %U2 = memref.cast %m2 : memref<1x2xi64> to memref<*xi64>
  // CHECK: [3, 2]
  func.call @printMemrefI64(%U0) : (memref<*xi64>) -> ()
  // CHECK: [2, 1]
  func.call @printMemrefI64(%U1) : (memref<*xi64>) -> ()
  // CHECK: [0, 0]
  func.call @printMemrefI64(%U2) : (memref<*xi64>) -> ()
  return
}
