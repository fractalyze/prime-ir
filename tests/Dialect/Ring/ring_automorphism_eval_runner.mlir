// Numeric exec: sigma_g on the evaluation basis is a pure permutation of the
// slots, with none of the sign flips the coefficient form needs.
//
// At q = 17, N = 4 the primitive 2N-th root is psi = 9, so the slots evaluate at
// psi^1, psi^3, psi^5, psi^7 = 9, 15, 8, 2. Under X -> X^3 slot j takes the
// value at psi^(3(2j+1)): psi^3, psi^9 = psi^1, psi^15 = psi^7, psi^21 = psi^5,
// i.e. slots 1, 0, 3, 2.

// RUN: prime-ir-opt %s -ring-to-mod-arith -field-to-llvm \
// RUN:   | mlir-runner -e test_automorphism_eval -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t
// RUN: prime-ir-opt %s -ring-to-mod-arith -field-to-llvm \
// RUN:   | mlir-runner -e test_automorphism_eval_involution -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t2
// RUN: FileCheck %s --check-prefix=TWICE < %t2

!Rq = !ring.rq<[17], 4 : i32, eval>

func.func private @printMemrefI64(memref<*xi64>) attributes { llvm.emit_c_interface }

func.func @test_automorphism_eval() {
  %a = arith.constant dense<[[10, 11, 12, 13]]> : tensor<1x4xi64>
  %ra = ring.from_tensor %a : tensor<1x4xi64> to !Rq
  %rc = ring.automorphism %ra {exponent = 3 : i64} : !Rq
  %out = ring.to_tensor %rc : !Rq to tensor<1x4xi64>
  %m = bufferization.to_buffer %out : tensor<1x4xi64> to memref<1x4xi64>
  %U = memref.cast %m : memref<1x4xi64> to memref<*xi64>
  // CHECK: [11, 10, 13, 12]
  func.call @printMemrefI64(%U) : (memref<*xi64>) -> ()
  return
}

// 3^2 = 9 = 1 mod 2N, so sigma_3 is its own inverse and applying it twice must
// return the input untouched.
func.func @test_automorphism_eval_involution() {
  %a = arith.constant dense<[[10, 11, 12, 13]]> : tensor<1x4xi64>
  %ra = ring.from_tensor %a : tensor<1x4xi64> to !Rq
  %r1 = ring.automorphism %ra {exponent = 3 : i64} : !Rq
  %r2 = ring.automorphism %r1 {exponent = 3 : i64} : !Rq
  %out = ring.to_tensor %r2 : !Rq to tensor<1x4xi64>
  %m = bufferization.to_buffer %out : tensor<1x4xi64> to memref<1x4xi64>
  %U = memref.cast %m : memref<1x4xi64> to memref<*xi64>
  // TWICE: [10, 11, 12, 13]
  func.call @printMemrefI64(%U) : (memref<*xi64>) -> ()
  return
}
