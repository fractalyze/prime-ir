// Numeric exec: ring.ntt round-trip (forward then inverse == identity).
// Basis [5, 13], N = 4 (both admit a 4th root of unity: 4 | 5-1 and 4 | 13-1).
// Residues must be canonical: limb 0 mod 5 in {0..4}, limb 1 mod 13 in {0..12}.
// intt(ntt(x)) must return x unchanged.

// RUN: prime-ir-opt %s -ring-to-mod-arith -poly-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_ntt -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

func.func @test_ntt() {
  %x = arith.constant dense<[[1, 2, 3, 4], [7, 8, 9, 10]]> : tensor<2x4xi64>
  %rx = ring.from_tensor %x : tensor<2x4xi64> to !ring.rq<[5, 13], 4 : i32>
  %f = ring.ntt %rx : !ring.rq<[5, 13], 4 : i32>
  %r = ring.ntt %f {inverse = true} : !ring.rq<[5, 13], 4 : i32>
  %ot = ring.to_tensor %r : !ring.rq<[5, 13], 4 : i32> to tensor<2x4xi64>

  %m = bufferization.to_buffer %ot : tensor<2x4xi64> to memref<2x4xi64>
  %U = memref.cast %m : memref<2x4xi64> to memref<*xi64>
  // CHECK: [1, 2, 3, 4]
  // CHECK: [7, 8, 9, 10]
  func.call @printMemrefI64(%U) : (memref<*xi64>) -> ()
  return
}
func.func private @printMemrefI64(memref<*xi64>) attributes { llvm.emit_c_interface }
