// Numeric exec: base conversion out of a modulus wider than the target one.
// A single input limb makes the conversion a pure reduction -- Q/q_0 = 1, so
// y_0 = x_0 and out = x_0 mod p_0 -- which isolates the reinterpret.
// q_0 = 2^61-1, p_0 = 2^31-1, x = 2^61-2, N = 2:
//   (2^61 - 2) mod (2^31 - 1) = 1073741822
// The target is Mersenne, so mod_arith takes its shift-and-add-back path,
// which subtracts p at most once and needs an operand already below 2p.

// RUN: prime-ir-opt %s -ring-to-mod-arith -field-to-llvm \
// RUN:   | mlir-runner -e test_base_convert_wide_source -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

func.func @test_base_convert_wide_source() {
  %x = arith.constant dense<[[2305843009213693950, 2305843009213693950]]>
      : tensor<1x2xi64>
  %rx = ring.from_tensor %x
      : tensor<1x2xi64> to !ring.rq<[2305843009213693951], 2 : i32>
  %ro = ring.base_convert %rx
      : !ring.rq<[2305843009213693951], 2 : i32> to !ring.rq<[2147483647], 2 : i32>
  %out = ring.to_tensor %ro
      : !ring.rq<[2147483647], 2 : i32> to tensor<1x2xi64>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %v0 = tensor.extract %out[%c0, %c0] : tensor<1x2xi64>
  %v1 = tensor.extract %out[%c0, %c1] : tensor<1x2xi64>
  // CHECK: 1073741822
  vector.print %v0 : i64
  // CHECK: 1073741822
  vector.print %v1 : i64
  return
}
