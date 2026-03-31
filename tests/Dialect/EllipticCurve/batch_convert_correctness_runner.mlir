// RUN: prime-ir-opt %s -elliptic-curve-to-field --field-to-llvm \
// RUN:   | mlir-runner -e main -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext,%mlir_lib_dir/libmlir_c_runner_utils%shlibext" > %t
// RUN: FileCheck %s < %t

!F = !field.pf<31:i32>
#curve = #elliptic_curve.sw<0:i32, 7:i32, (1:i32, 15:i32)> : !F
!Affine = !elliptic_curve.affine<#curve>
!Jacobian = !elliptic_curve.jacobian<#curve>

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @print_field(%v: !F) {
  %i = field.bitcast %v : !F -> i32
  %t = tensor.from_elements %i : tensor<1xi32>
  %m = bufferization.to_buffer %t : tensor<1xi32> to memref<1xi32>
  %u = memref.cast %m : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%u) : (memref<*xi32>) -> ()
  return
}

// Use function args to prevent constant folding.
func.func @run(%x: !F, %y: !F, %z: !F) {
  // Scalar convert_point_type.
  %j = elliptic_curve.from_coords %x, %y, %z : (!F, !F, !F) -> !Jacobian
  %a_s = elliptic_curve.convert_point_type %j : !Jacobian -> !Affine
  %sx, %sy = elliptic_curve.to_coords %a_s : (!Affine) -> (!F, !F)
  func.call @print_field(%sx) : (!F) -> ()
  func.call @print_field(%sy) : (!F) -> ()

  // Batch (N=2) convert_point_type — same point twice.
  %jt = tensor.from_elements %j, %j : tensor<2x!Jacobian>
  %at = elliptic_curve.convert_point_type %jt
      : tensor<2x!Jacobian> -> tensor<2x!Affine>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %a0 = tensor.extract %at[%c0] : tensor<2x!Affine>
  %bx0, %by0 = elliptic_curve.to_coords %a0 : (!Affine) -> (!F, !F)
  func.call @print_field(%bx0) : (!F) -> ()
  func.call @print_field(%by0) : (!F) -> ()
  %a1 = tensor.extract %at[%c1] : tensor<2x!Affine>
  %bx1, %by1 = elliptic_curve.to_coords %a1 : (!Affine) -> (!F, !F)
  func.call @print_field(%bx1) : (!F) -> ()
  func.call @print_field(%by1) : (!F) -> ()
  return
}

func.func @main() {
  %x = field.constant 3 : !F
  %y = field.constant 5 : !F
  %z = field.constant 2 : !F
  func.call @run(%x, %y, %z) : (!F, !F, !F) -> ()
  return
}

// Scalar results:
// CHECK: [24]
// CHECK: [20]
// Batch[0] (should match scalar):
// CHECK: [24]
// CHECK: [20]
// Batch[1] (should match scalar):
// CHECK: [24]
// CHECK: [20]
