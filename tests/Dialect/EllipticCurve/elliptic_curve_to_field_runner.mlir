// RUN: zkir-opt %s -elliptic-curve-to-llvm \
// RUN:   | mlir-runner -e test_ops_in_order -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_OPS_IN_ORDER < %t

!PF = !field.pf<11:i32>

#1 = #field.pf_elem<1:i32> : !PF
#2 = #field.pf_elem<2:i32> : !PF
#3 = #field.pf_elem<3:i32> : !PF
#4 = #field.pf_elem<4:i32> : !PF

#curve = #elliptic_curve.sw<#1, #2, (#3, #4)>
!affine = !elliptic_curve.affine<#curve>
!jacobian = !elliptic_curve.jacobian<#curve>
!xyzz = !elliptic_curve.xyzz<#curve>

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

// CHECK-LABEL: @test_ops_in_order
func.func @test_ops_in_order() {
  %var1 = field.pf.constant 1 : !PF
  %var2 = field.pf.constant 2 : !PF
  %var3 = field.pf.constant 3 : !PF
  %var5 = field.pf.constant 5 : !PF
  %var7 = field.pf.constant 7 : !PF

  %affine1 = elliptic_curve.point %var1, %var2 : !PF -> !affine
  %jacobian1 = elliptic_curve.point %var5, %var3, %var2 : !PF -> !jacobian

  %jacobian2 = elliptic_curve.add %affine1, %jacobian1 : !affine, !jacobian -> !jacobian
  %extract_point1 = elliptic_curve.extract %jacobian2 : !jacobian -> tensor<3x!PF>
  %extract1 = field.pf.extract %extract_point1 : tensor<3x!PF> -> tensor<3xi32>
  %1 = bufferization.to_memref %extract1 : tensor<3xi32> to memref<3xi32>
  %U1 = memref.cast %1 : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%U1) : (memref<*xi32>) -> ()

  %jacobian3 = elliptic_curve.sub %affine1, %jacobian2 : !affine, !jacobian -> !jacobian
  %extract_point2 = elliptic_curve.extract %jacobian3 : !jacobian -> tensor<3x!PF>
  %extract2 = field.pf.extract %extract_point2 : tensor<3x!PF> -> tensor<3xi32>
  %2 = bufferization.to_memref %extract2 : tensor<3xi32> to memref<3xi32>
  %U2 = memref.cast %2 : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%U2) : (memref<*xi32>) -> ()

  %jacobian4 = elliptic_curve.negate %jacobian3 : !jacobian
  %extract_point3 = elliptic_curve.extract %jacobian4 : !jacobian -> tensor<3x!PF>
  %extract3 = field.pf.extract %extract_point3 : tensor<3x!PF> -> tensor<3xi32>
  %3 = bufferization.to_memref %extract3 : tensor<3xi32> to memref<3xi32>
  %U3 = memref.cast %3 : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%U3) : (memref<*xi32>) -> ()

  %jacobian5 = elliptic_curve.double %jacobian4 : !jacobian -> !jacobian
  %extract_point4 = elliptic_curve.extract %jacobian5 : !jacobian -> tensor<3x!PF>
  %extract4 = field.pf.extract %extract_point4 : tensor<3x!PF> -> tensor<3xi32>
  %4 = bufferization.to_memref %extract4 : tensor<3xi32> to memref<3xi32>
  %U4 = memref.cast %4 : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%U4) : (memref<*xi32>) -> ()

  %xyzz1 = elliptic_curve.convert_point_type %jacobian5 : !jacobian -> !xyzz
  %extract_point5 = elliptic_curve.extract %xyzz1 : !xyzz -> tensor<4x!PF>
  %extract5 = field.pf.extract %extract_point5 : tensor<4x!PF> -> tensor<4xi32>
  %5 = bufferization.to_memref %extract5 : tensor<4xi32> to memref<4xi32>
  %U5 = memref.cast %5 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%U5) : (memref<*xi32>) -> ()

  %affine2 = elliptic_curve.convert_point_type %xyzz1 : !xyzz -> !affine
  %extract_point6 = elliptic_curve.extract %affine2 : !affine -> tensor<2x!PF>
  %extract6 = field.pf.extract %extract_point6 : tensor<2x!PF> -> tensor<2xi32>
  %6 = bufferization.to_memref %extract6 : tensor<2xi32> to memref<2xi32>
  %U6 = memref.cast %6 : memref<2xi32> to memref<*xi32>
  func.call @printMemrefI32(%U6) : (memref<*xi32>) -> ()

  %jacobian6 = elliptic_curve.scalar_mul %var7, %affine2 : !PF, !affine -> !jacobian
  %extract_point7 = elliptic_curve.extract %jacobian6 : !jacobian -> tensor<3x!PF>
  %extract7 = field.pf.extract %extract_point7 : tensor<3x!PF> -> tensor<3xi32>
  %7 = bufferization.to_memref %extract7 : tensor<3xi32> to memref<3xi32>
  %U7 = memref.cast %7 : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%U7) : (memref<*xi32>) -> ()

  %affine3 = elliptic_curve.convert_point_type %jacobian6 : !jacobian -> !affine
  %extract_point8 = elliptic_curve.extract %affine3 : !affine -> tensor<2x!PF>
  %extract8 = field.pf.extract %extract_point8 : tensor<2x!PF> -> tensor<2xi32>
  %8 = bufferization.to_memref %extract8 : tensor<2xi32> to memref<2xi32>
  %U8 = memref.cast %8 : memref<2xi32> to memref<*xi32>
  func.call @printMemrefI32(%U8) : (memref<*xi32>) -> ()

  %xyzz2 = elliptic_curve.add %affine3, %xyzz1 : !affine, !xyzz -> !xyzz
  %extract_point9 = elliptic_curve.extract %xyzz2 : !xyzz -> tensor<4x!PF>
  %extract9 = field.pf.extract %extract_point9 : tensor<4x!PF> -> tensor<4xi32>
  %9 = bufferization.to_memref %extract9 : tensor<4xi32> to memref<4xi32>
  %U9 = memref.cast %9 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%U9) : (memref<*xi32>) -> ()
  return
}

// CHECK_TEST_OPS_IN_ORDER: [2, 8, 7]
// CHECK_TEST_OPS_IN_ORDER: [5, 3, 9]
// CHECK_TEST_OPS_IN_ORDER: [5, 8, 9]
// CHECK_TEST_OPS_IN_ORDER: [1, 10, 1]
// CHECK_TEST_OPS_IN_ORDER: [1, 10, 1, 1]
// CHECK_TEST_OPS_IN_ORDER: [1, 10]
// CHECK_TEST_OPS_IN_ORDER: [0, 0, 0]
// CHECK_TEST_OPS_IN_ORDER: [1, 1]
// CHECK_TEST_OPS_IN_ORDER: [4, 3, 0, 0]
