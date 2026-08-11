// Numeric exec: 2-limb RNS pointwise multiply on two real word-size NTT primes
// (KoalaBear 2130706433 and BabyBear 2013265921; 2N = 8 divides q-1 for both,
// so the eval basis exists at N = 4). Each limb carries its own modulus, so the
// same input slot must reduce differently per limb.

// RUN: prime-ir-opt %s -ring-to-mod-arith -field-to-llvm \
// RUN:   | mlir-runner -e test_rns -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

!Rq = !ring.rq<[2130706433, 2013265921], 4 : i32, eval>

func.func private @printMemrefI64(memref<*xi64>) attributes { llvm.emit_c_interface }

// Slot 0 holds q_i - 1 in each limb (squares to 1); slot 1 holds 100000 in both,
// and 10^10 reduces to different residues under the two moduli.
func.func @test_rns() {
  %a = arith.constant dense<[[2130706432, 100000, 3, 4],
                             [2013265920, 100000, 3, 4]]> : tensor<2x4xi64>
  %ra = ring.from_tensor %a : tensor<2x4xi64> to !Rq
  %rc = ring.mul %ra, %ra : !Rq
  %out = ring.to_tensor %rc : !Rq to tensor<2x4xi64>
  %m = bufferization.to_buffer %out : tensor<2x4xi64> to memref<2x4xi64>
  %U = memref.cast %m : memref<2x4xi64> to memref<*xi64>
  // CHECK: {{\[}}[1, 1477174268, 9, 16]
  // CHECK: [1, 1946936316, 9, 16]
  func.call @printMemrefI64(%U) : (memref<*xi64>) -> ()
  return
}
