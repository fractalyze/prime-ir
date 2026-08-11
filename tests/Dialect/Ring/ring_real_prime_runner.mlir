// Numeric exec on a real word-size NTT prime (KoalaBear = 2130706433 = 2^31 -
// 2^24 + 1; 2N = 8 divides q-1, so the eval basis exists at N = 4). Operands
// near q square to ~2^62, which is where a reducer that stays in 32 bits or
// forgets the high half of the product falls over.

// RUN: prime-ir-opt %s -ring-to-mod-arith -field-to-llvm \
// RUN:   | mlir-runner -e test_kb -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

!Rq = !ring.rq<[2130706433], 4 : i32, eval>

func.func private @printMemrefI64(memref<*xi64>) attributes { llvm.emit_c_interface }

// Squaring [q-1, q-1, 12345, 2]: (q-1)^2 = 1, 12345^2 = 152399025 (< q), 2^2 = 4.
func.func @test_kb() {
  %a = arith.constant dense<[[2130706432, 2130706432, 12345, 2]]> : tensor<1x4xi64>
  %ra = ring.from_tensor %a : tensor<1x4xi64> to !Rq
  %rc = ring.mul %ra, %ra : !Rq
  %out = ring.to_tensor %rc : !Rq to tensor<1x4xi64>
  %m = bufferization.to_buffer %out : tensor<1x4xi64> to memref<1x4xi64>
  %U = memref.cast %m : memref<1x4xi64> to memref<*xi64>
  // CHECK: [1, 1, 152399025, 4]
  func.call @printMemrefI64(%U) : (memref<*xi64>) -> ()
  return
}
