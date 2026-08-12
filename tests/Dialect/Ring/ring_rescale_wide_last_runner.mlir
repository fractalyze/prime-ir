// Numeric exec: rescale when the dropped modulus exceeds a surviving one, so
// its residue is not already a residue of that limb. Basis [7, 11, 13] -> drop
// 13 -> [7, 11], N = 2. Values 24 and 12:
//   24 = (3, 2, 11) in [7,11,13];  12 = (5, 1, 12) -> input [[3,5],[2,1],[11,12]]
//   (24 - 11)/13 = 1;  (12 - 12)/13 = 0
//   1 = (1, 1) in [7,11];  0 = (0, 0)  -> output [[1,0],[1,0]].
// Both trailing residues (11, 12) are >= 7 and 11 <= 11, which is what makes
// this distinct from the small-residue case: reinterpreting them into the
// surviving limbs without reducing first names the wrong value.

// RUN: prime-ir-opt %s -ring-to-mod-arith -field-to-llvm \
// RUN:   | mlir-runner -e test_rescale_wide_last -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

func.func @test_rescale_wide_last() {
  %x = arith.constant dense<[[3, 5], [2, 1], [11, 12]]> : tensor<3x2xi64>
  %rx = ring.from_tensor %x : tensor<3x2xi64> to !ring.rq<[7, 11, 13], 2 : i32>
  %ro = ring.rescale %rx : !ring.rq<[7, 11, 13], 2 : i32> to !ring.rq<[7, 11], 2 : i32>
  %out = ring.to_tensor %ro : !ring.rq<[7, 11], 2 : i32> to tensor<2x2xi64>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %v00 = tensor.extract %out[%c0, %c0] : tensor<2x2xi64>
  %v01 = tensor.extract %out[%c0, %c1] : tensor<2x2xi64>
  %v10 = tensor.extract %out[%c1, %c0] : tensor<2x2xi64>
  %v11 = tensor.extract %out[%c1, %c1] : tensor<2x2xi64>
  // CHECK: 1
  vector.print %v00 : i64
  // CHECK: 0
  vector.print %v01 : i64
  // CHECK: 1
  vector.print %v10 : i64
  // CHECK: 0
  vector.print %v11 : i64
  return
}
