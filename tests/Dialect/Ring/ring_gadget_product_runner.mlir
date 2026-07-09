// Numeric exec: ring.gadget_product = sum_j decompose(x)_j * keys[j].
// Basis [17], N=4, baseBits=2 (base 4). x = 5 (constant) decomposes to digits
// (1, 1): 5 = 1 + 1*4. keys k0 = X, k1 = X^2.
//   sum = 1*X + 1*X^2 = X + X^2 = [0, 1, 1, 0].

// RUN: prime-ir-opt %s -ring-to-mod-arith -poly-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_gp -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

func.func private @printMemrefI64(memref<*xi64>) attributes { llvm.emit_c_interface }

func.func @test_gp() {
  %x = arith.constant dense<[[5, 0, 0, 0]]> : tensor<1x4xi64>
  %k0 = arith.constant dense<[[0, 1, 0, 0]]> : tensor<1x4xi64>
  %k1 = arith.constant dense<[[0, 0, 1, 0]]> : tensor<1x4xi64>
  %rx = ring.from_tensor %x : tensor<1x4xi64> to !ring.rq<[17], 4 : i32>
  %rk0 = ring.from_tensor %k0 : tensor<1x4xi64> to !ring.rq<[17], 4 : i32>
  %rk1 = ring.from_tensor %k1 : tensor<1x4xi64> to !ring.rq<[17], 4 : i32>
  %rc = ring.gadget_product %rx, %rk0, %rk1 {baseBits = 2 : i64}
      : !ring.rq<[17], 4 : i32>, !ring.rq<[17], 4 : i32>, !ring.rq<[17], 4 : i32>
      -> !ring.rq<[17], 4 : i32>
  %ot = ring.to_tensor %rc : !ring.rq<[17], 4 : i32> to tensor<1x4xi64>
  %m = bufferization.to_buffer %ot : tensor<1x4xi64> to memref<1x4xi64>
  %U = memref.cast %m : memref<1x4xi64> to memref<*xi64>
  // CHECK: [0, 1, 1, 0]
  func.call @printMemrefI64(%U) : (memref<*xi64>) -> ()
  return
}
