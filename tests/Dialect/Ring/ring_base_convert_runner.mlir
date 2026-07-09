// Numeric exec-vs-oracle for ring.base_convert (CRT fast basis extension).
//
// Input integers (per coefficient) are 8 and 11, in RNS basis [3, 5]:
//   8  = (8 mod 3, 8 mod 5)  = (2, 3)
//   11 = (11 mod 3, 11 mod 5) = (2, 1)
// so limb-major tensor<2x2xi64> = [[2, 2], [3, 1]].
// base_convert to basis [7] must give (8 mod 7, 11 mod 7) = (1, 4).

// RUN: prime-ir-opt %s -ring-to-mod-arith -field-to-llvm \
// RUN:   | mlir-runner -e test_bconv -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

func.func @test_bconv() {
  %in = arith.constant dense<[[2, 2], [3, 1]]> : tensor<2x2xi64>
  %r = ring.from_tensor %in : tensor<2x2xi64> to !ring.rq<[3, 5], 2 : i32>
  %o = ring.base_convert %r : !ring.rq<[3, 5], 2 : i32> to !ring.rq<[7], 2 : i32>
  %ot = ring.to_tensor %o : !ring.rq<[7], 2 : i32> to tensor<1x2xi64>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %v0 = tensor.extract %ot[%c0, %c0] : tensor<1x2xi64>
  %v1 = tensor.extract %ot[%c0, %c1] : tensor<1x2xi64>
  // CHECK: 1
  vector.print %v0 : i64
  // CHECK: 4
  vector.print %v1 : i64
  return
}
