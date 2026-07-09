// Numeric exec: rescale (exact division by the dropped modulus) in RNS.
// Basis [7, 11, 13] -> drop 13 -> [7, 11], N = 2. Values 27 and 40:
//   27 = (6, 5, 1) in [7,11,13];  40 = (5, 7, 1)  -> input [[6,5],[5,7],[1,1]]
//   (27 - 27 mod 13)/13 = 26/13 = 2;  (40 - 40 mod 13)/13 = 39/13 = 3
//   2 = (2, 2) in [7,11];  3 = (3, 3)  -> output [[2,3],[2,3]].

// RUN: prime-ir-opt %s -ring-to-mod-arith -field-to-llvm \
// RUN:   | mlir-runner -e test_rescale -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

func.func @test_rescale() {
  %x = arith.constant dense<[[6, 5], [5, 7], [1, 1]]> : tensor<3x2xi64>
  %rx = ring.from_tensor %x : tensor<3x2xi64> to !ring.rq<[7, 11, 13], 2 : i32>
  %ro = ring.rescale %rx : !ring.rq<[7, 11, 13], 2 : i32> to !ring.rq<[7, 11], 2 : i32>
  %ot = ring.to_tensor %ro : !ring.rq<[7, 11], 2 : i32> to tensor<2x2xi64>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %v00 = tensor.extract %ot[%c0, %c0] : tensor<2x2xi64>
  %v01 = tensor.extract %ot[%c0, %c1] : tensor<2x2xi64>
  %v10 = tensor.extract %ot[%c1, %c0] : tensor<2x2xi64>
  %v11 = tensor.extract %ot[%c1, %c1] : tensor<2x2xi64>
  // CHECK: 2
  vector.print %v00 : i64
  // CHECK: 3
  vector.print %v01 : i64
  // CHECK: 2
  vector.print %v10 : i64
  // CHECK: 3
  vector.print %v11 : i64
  return
}
