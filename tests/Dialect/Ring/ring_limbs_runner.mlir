// Numeric exec through the limb bridge: two field-typed limbs go in, one ring
// value comes out, and the eval-basis product is the per-limb residue product.
// The same input slot reduces against a different modulus in each limb, so a
// bridge that mixed the rows up would show as a wrong residue rather than a
// wrong shape.

// RUN: prime-ir-opt %s -ring-to-mod-arith -field-to-llvm \
// RUN:   | mlir-runner -e test_limb_mul -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

!PF17 = !field.pf<17:i64>
!PF41 = !field.pf<41:i64>
!Rq = !ring.rq<[17, 41], 4 : i32, eval>

func.func private @printMemrefI64(memref<*xi64>) attributes { llvm.emit_c_interface }

func.func @test_limb_mul() {
  // 3*5=15, 4*6=24, 7*8=56, 9*2=18 before reduction.
  %a0i = arith.constant dense<[3, 4, 7, 9]> : tensor<4xi64>
  %b0i = arith.constant dense<[5, 6, 8, 2]> : tensor<4xi64>
  %a1i = arith.constant dense<[3, 4, 7, 9]> : tensor<4xi64>
  %b1i = arith.constant dense<[5, 6, 8, 2]> : tensor<4xi64>

  %a0 = field.bitcast %a0i : tensor<4xi64> -> tensor<4x!PF17>
  %b0 = field.bitcast %b0i : tensor<4xi64> -> tensor<4x!PF17>
  %a1 = field.bitcast %a1i : tensor<4xi64> -> tensor<4x!PF41>
  %b1 = field.bitcast %b1i : tensor<4xi64> -> tensor<4x!PF41>

  %ra = ring.from_limbs %a0, %a1 : tensor<4x!PF17>, tensor<4x!PF41> to !Rq
  %rb = ring.from_limbs %b0, %b1 : tensor<4x!PF17>, tensor<4x!PF41> to !Rq
  %rc = ring.mul %ra, %rb : !Rq

  %out = ring.to_tensor %rc : !Rq to tensor<2x4xi64>
  %m = bufferization.to_buffer %out : tensor<2x4xi64> to memref<2x4xi64>
  %U = memref.cast %m : memref<2x4xi64> to memref<*xi64>
  call @printMemrefI64(%U) : (memref<*xi64>) -> ()
  // mod 17: 15, 7, 5, 1   mod 41: 15, 24, 15, 18
  // CHECK: [15, 7, 5, 1]
  // CHECK: [15, 24, 15, 18]
  return
}
