// Numeric exec: ring.negate is the per-limb modular negation, so each residue
// becomes q_i - x_i (and 0 stays 0, which is where a plain subtract-from-q
// would instead produce q). Two limbs, so the same input slot negates against a
// different modulus in each.

// RUN: prime-ir-opt %s -ring-to-mod-arith -field-to-llvm \
// RUN:   | mlir-runner -e test_negate -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

!Rq = !ring.rq<[17, 41], 4 : i32>

func.func private @printMemrefI64(memref<*xi64>) attributes { llvm.emit_c_interface }

func.func @test_negate() {
  %a = arith.constant dense<[[0, 1, 5, 16], [0, 1, 5, 40]]> : tensor<2x4xi64>
  %ra = ring.from_tensor %a : tensor<2x4xi64> to !Rq
  %rc = ring.negate %ra : !Rq
  %out = ring.to_tensor %rc : !Rq to tensor<2x4xi64>
  %m = bufferization.to_buffer %out : tensor<2x4xi64> to memref<2x4xi64>
  %U = memref.cast %m : memref<2x4xi64> to memref<*xi64>
  // CHECK: {{\[}}[0, 16, 12, 1]
  // CHECK: [0, 40, 36, 1]
  func.call @printMemrefI64(%U) : (memref<*xi64>) -> ()
  return
}
