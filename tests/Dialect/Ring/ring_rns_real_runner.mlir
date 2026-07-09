// Numeric exec: 2-limb RNS negacyclic multiply on two real word-size NTT primes
// (KoalaBear 2130706433 and BabyBear 2013265921; both have 8 | q-1). N = 4.
// X^2 * X^2 = -1 in each limb -> (q0-1, q1-1) = (2130706432, 2013265920).
// Input limb-major tensor<2x4xi64>: both limbs hold X^2 = [0,0,1,0].

// RUN: prime-ir-opt %s -ring-to-mod-arith -poly-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_rns -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

func.func private @printMemrefI64(memref<*xi64>) attributes { llvm.emit_c_interface }

func.func @test_rns() {
  %a = arith.constant dense<[[0, 0, 1, 0], [0, 0, 1, 0]]> : tensor<2x4xi64>
  %ra = ring.from_tensor %a
      : tensor<2x4xi64> to !ring.rq<[2130706433, 2013265921], 4 : i32>
  %rc = ring.mul %ra, %ra : !ring.rq<[2130706433, 2013265921], 4 : i32>
  %ot = ring.to_tensor %rc
      : !ring.rq<[2130706433, 2013265921], 4 : i32> to tensor<2x4xi64>
  %m = bufferization.to_buffer %ot : tensor<2x4xi64> to memref<2x4xi64>
  %U = memref.cast %m : memref<2x4xi64> to memref<*xi64>
  // CHECK: {{\[}}[2130706432, 0, 0, 0]
  // CHECK: [2013265920, 0, 0, 0]
  func.call @printMemrefI64(%U) : (memref<*xi64>) -> ()
  return
}
