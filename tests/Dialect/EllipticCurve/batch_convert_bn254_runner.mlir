// RUN: prime-ir-opt %s -elliptic-curve-to-field --field-to-llvm \
// RUN:   | mlir-runner -e main -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext,%mlir_lib_dir/libmlir_c_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

// BN254 Fq Montgomery.
!PFm = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256, true>
#curvem = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PFm
!affinem = !elliptic_curve.affine<#curvem>
!jacobianm = !elliptic_curve.jacobian<#curvem>

func.func private @printMemrefI64(memref<*xi64>) attributes { llvm.emit_c_interface }

// Print low 64 bits of a field element (enough to detect mismatches).
func.func @print_low64(%v: !PFm) {
  %i256 = field.bitcast %v : !PFm -> i256
  %i64 = arith.trunci %i256 : i256 to i64
  %t = tensor.from_elements %i64 : tensor<1xi64>
  %m = bufferization.to_buffer %t : tensor<1xi64> to memref<1xi64>
  %u = memref.cast %m : memref<1xi64> to memref<*xi64>
  func.call @printMemrefI64(%u) : (memref<*xi64>) -> ()
  return
}

func.func @run(%x: !PFm, %y: !PFm, %z: !PFm) {
  // Scalar.
  %j = elliptic_curve.from_coords %x, %y, %z : (!PFm, !PFm, !PFm) -> !jacobianm
  %a_s = elliptic_curve.convert_point_type %j : !jacobianm -> !affinem
  %sx, %sy = elliptic_curve.to_coords %a_s : (!affinem) -> (!PFm, !PFm)
  func.call @print_low64(%sx) : (!PFm) -> ()
  func.call @print_low64(%sy) : (!PFm) -> ()

  // Batch N=8 (same point repeated).
  %jt = tensor.from_elements %j, %j, %j, %j, %j, %j, %j, %j
      : tensor<8x!jacobianm>
  %at = elliptic_curve.convert_point_type %jt
      : tensor<8x!jacobianm> -> tensor<8x!affinem>
  %c0 = arith.constant 0 : index
  %c7 = arith.constant 7 : index
  // Check first and last elements.
  %a0 = tensor.extract %at[%c0] : tensor<8x!affinem>
  %bx0, %by0 = elliptic_curve.to_coords %a0 : (!affinem) -> (!PFm, !PFm)
  func.call @print_low64(%bx0) : (!PFm) -> ()
  func.call @print_low64(%by0) : (!PFm) -> ()
  %a7 = tensor.extract %at[%c7] : tensor<8x!affinem>
  %bx7, %by7 = elliptic_curve.to_coords %a7 : (!affinem) -> (!PFm, !PFm)
  func.call @print_low64(%bx7) : (!PFm) -> ()
  func.call @print_low64(%by7) : (!PFm) -> ()
  return
}

func.func @main() {
  %x = field.constant 3 : !PFm
  %y = field.constant 5 : !PFm
  %z = field.constant 2 : !PFm
  func.call @run(%x, %y, %z) : (!PFm, !PFm, !PFm) -> ()
  return
}

// Scalar x, y (low64):
// CHECK: [[X:\[[-0-9]+\]]]
// CHECK: [[Y:\[[-0-9]+\]]]
// Batch[0] should match scalar:
// CHECK: [[X]]
// CHECK: [[Y]]
// Batch[7] should match scalar:
// CHECK: [[X]]
// CHECK: [[Y]]
