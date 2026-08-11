// Numeric exec: homomorphic addition in RNS basis [7, 11] (Q = 77), N = 2.
//   a = (20, 3):  20 = (6, 9) in [7,11],  3 = (3, 3)   -> [[6, 3], [9, 3]]
//   b = (15, 8):  15 = (1, 4),            8 = (1, 8)   -> [[1, 1], [4, 8]]
// a + b = (35, 11) -> residues (35 mod 7, 35 mod 11) = (0, 2),
//                               (11 mod 7, 11 mod 11) = (4, 0)
// so the output tensor<2x2xi64> (limb-major) is [[0, 4], [2, 0]].

// RUN: prime-ir-opt %s -ring-to-mod-arith -field-to-llvm \
// RUN:   | mlir-runner -e test_add -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

func.func @test_add() {
  %a = arith.constant dense<[[6, 3], [9, 3]]> : tensor<2x2xi64>
  %b = arith.constant dense<[[1, 1], [4, 8]]> : tensor<2x2xi64>
  %ra = ring.from_tensor %a : tensor<2x2xi64> to !ring.rq<[7, 11], 2 : i32>
  %rb = ring.from_tensor %b : tensor<2x2xi64> to !ring.rq<[7, 11], 2 : i32>
  %rc = ring.add %ra, %rb : !ring.rq<[7, 11], 2 : i32>
  %out = ring.to_tensor %rc : !ring.rq<[7, 11], 2 : i32> to tensor<2x2xi64>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %v00 = tensor.extract %out[%c0, %c0] : tensor<2x2xi64>
  %v01 = tensor.extract %out[%c0, %c1] : tensor<2x2xi64>
  %v10 = tensor.extract %out[%c1, %c0] : tensor<2x2xi64>
  %v11 = tensor.extract %out[%c1, %c1] : tensor<2x2xi64>
  // CHECK: 0
  vector.print %v00 : i64
  // CHECK: 4
  vector.print %v01 : i64
  // CHECK: 2
  vector.print %v10 : i64
  // CHECK: 0
  vector.print %v11 : i64
  return
}
