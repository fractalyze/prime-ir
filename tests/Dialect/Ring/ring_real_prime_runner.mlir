// Numeric exec on a real word-size NTT prime (KoalaBear = 2130706433 = 2^31 -
// 2^24 + 1; 2N=8 | q-1). Exercises the generator-based root finder (an O(q) scan
// would be hopeless here). N = 4.
//   X^2 * X^2 = X^4 = -1 (mod X^4+1) = q-1 = 2130706432 mod q. -> [2130706432,0,0,0]

// RUN: prime-ir-opt %s -ring-to-mod-arith -poly-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_kb -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

func.func private @printMemrefI64(memref<*xi64>) attributes { llvm.emit_c_interface }

func.func @test_kb() {
  %a = arith.constant dense<[[0, 0, 1, 0]]> : tensor<1x4xi64>
  %b = arith.constant dense<[[0, 0, 1, 0]]> : tensor<1x4xi64>
  %ra = ring.from_tensor %a : tensor<1x4xi64> to !ring.rq<[2130706433], 4 : i32>
  %rb = ring.from_tensor %b : tensor<1x4xi64> to !ring.rq<[2130706433], 4 : i32>
  %rc = ring.mul %ra, %rb : !ring.rq<[2130706433], 4 : i32>
  %ot = ring.to_tensor %rc : !ring.rq<[2130706433], 4 : i32> to tensor<1x4xi64>
  %m = bufferization.to_buffer %ot : tensor<1x4xi64> to memref<1x4xi64>
  %U = memref.cast %m : memref<1x4xi64> to memref<*xi64>
  // CHECK: [2130706432, 0, 0, 0]
  func.call @printMemrefI64(%U) : (memref<*xi64>) -> ()
  return
}
