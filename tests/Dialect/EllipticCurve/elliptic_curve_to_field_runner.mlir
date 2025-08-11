// RUN: zkir-opt %s -elliptic-curve-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_ops_in_order -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_OPS_IN_ORDER < %t

//BN254
!PF = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!SF = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256>
!PFm = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256, true>
!SFm = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256, true>

#a = #field.pf.elem<0:i256> : !PFm
#b = #field.pf.elem<3:i256> : !PFm
#1 = #field.pf.elem<1:i256> : !PFm
#2 = #field.pf.elem<2:i256> : !PFm

#curve = #elliptic_curve.sw<#a, #b, (#1, #2)>
!affine = !elliptic_curve.affine<#curve>
!jacobian = !elliptic_curve.jacobian<#curve>
!xyzz = !elliptic_curve.xyzz<#curve>

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

// CHECK-LABEL: @test_ops_in_order
func.func @test_ops_in_order() {
  %c1 = arith.constant 1 : i256
  %c2 = arith.constant 2 : i256

  %c7 = field.constant 7 : !SF

  %index1 = arith.constant 0 : index
  %index2 = arith.constant 1 : index

  %c_tensor = tensor.from_elements %c1, %c2: tensor<2xi256>
  %f_tensor = field.encapsulate %c_tensor : tensor<2xi256> -> tensor<2x!PF>
  %c_mont_tensor = field.to_mont %f_tensor : tensor<2x!PFm>
  %var1 = tensor.extract %c_mont_tensor[%index1] : tensor<2x!PFm>
  %var2 = tensor.extract %c_mont_tensor[%index2] : tensor<2x!PFm>
  %var7 = field.to_mont %c7 : !SFm

  // (1,2)
  %affine1 = elliptic_curve.point %var1, %var2 : !affine
  // (1,2,1)
  %jacobian1 = elliptic_curve.point %var1, %var2, %var1 : !jacobian

  %jacobian2 = elliptic_curve.add %affine1, %jacobian1 : !affine, !jacobian -> !jacobian
  %extract_point1x, %extract_point1y, %extract_point1z = elliptic_curve.extract %jacobian2 : !jacobian -> !PFm, !PFm, !PFm
  %extract_point1 = tensor.from_elements %extract_point1x, %extract_point1y, %extract_point1z : tensor<3x!PFm>
  %from_mont1 = field.from_mont %extract_point1 : tensor<3x!PF>
  %extract1 = field.extract %from_mont1 : tensor<3x!PF> -> tensor<3xi256>
  %trunc1 = arith.trunci %extract1 : tensor<3xi256> to tensor<3xi32>
  %1 = bufferization.to_buffer %trunc1 : tensor<3xi32> to memref<3xi32>
  %U1 = memref.cast %1 : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%U1) : (memref<*xi32>) -> ()

  %jacobian3 = elliptic_curve.sub %affine1, %jacobian2 : !affine, !jacobian -> !jacobian
  %extract_point2x, %extract_point2y, %extract_point2z = elliptic_curve.extract %jacobian3 : !jacobian -> !PFm, !PFm, !PFm
  %extract_point2 = tensor.from_elements %extract_point2x, %extract_point2y, %extract_point2z : tensor<3x!PFm>
  %from_mont2 = field.from_mont %extract_point2 : tensor<3x!PF>
  %extract2 = field.extract %from_mont2 : tensor<3x!PF> -> tensor<3xi256>
  %trunc2 = arith.trunci %extract2 : tensor<3xi256> to tensor<3xi32>
  %2 = bufferization.to_buffer %trunc2 : tensor<3xi32> to memref<3xi32>
  %U2 = memref.cast %2 : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%U2) : (memref<*xi32>) -> ()

  %jacobian4 = elliptic_curve.negate %jacobian3 : !jacobian
  %extract_point3x, %extract_point3y, %extract_point3z = elliptic_curve.extract %jacobian4 : !jacobian -> !PFm, !PFm, !PFm
  %extract_point3 = tensor.from_elements %extract_point3x, %extract_point3y, %extract_point3z : tensor<3x!PFm>
  %from_mont3 = field.from_mont %extract_point3 : tensor<3x!PF>
  %extract3 = field.extract %from_mont3 : tensor<3x!PF> -> tensor<3xi256>
  %trunc3 = arith.trunci %extract3 : tensor<3xi256> to tensor<3xi32>
  %3 = bufferization.to_buffer %trunc3 : tensor<3xi32> to memref<3xi32>
  %U3 = memref.cast %3 : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%U3) : (memref<*xi32>) -> ()

  %jacobian5 = elliptic_curve.double %jacobian4 : !jacobian -> !jacobian
  %extract_point4x, %extract_point4y, %extract_point4z = elliptic_curve.extract %jacobian5 : !jacobian -> !PFm, !PFm, !PFm
  %extract_point4 = tensor.from_elements %extract_point4x, %extract_point4y, %extract_point4z : tensor<3x!PFm>
  %from_mont4 = field.from_mont %extract_point4 : tensor<3x!PF>
  %extract4 = field.extract %from_mont4 : tensor<3x!PF> -> tensor<3xi256>
  %trunc4 = arith.trunci %extract4 : tensor<3xi256> to tensor<3xi32>
  %4 = bufferization.to_buffer %trunc4 : tensor<3xi32> to memref<3xi32>
  %U4 = memref.cast %4 : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%U4) : (memref<*xi32>) -> ()

  %xyzz1 = elliptic_curve.convert_point_type %affine1 : !affine -> !xyzz
  %extract_point5x, %extract_point5y, %extract_point5zz, %extract_point5zzz = elliptic_curve.extract %xyzz1 : !xyzz -> !PFm, !PFm, !PFm, !PFm
  %extract_point5 = tensor.from_elements %extract_point5x, %extract_point5y, %extract_point5zz, %extract_point5zzz : tensor<4x!PFm>
  %from_mont5 = field.from_mont %extract_point5 : tensor<4x!PF>
  %extract5 = field.extract %from_mont5 : tensor<4x!PF> -> tensor<4xi256>
  %trunc5 = arith.trunci %extract5 : tensor<4xi256> to tensor<4xi32>
  %5 = bufferization.to_buffer %trunc5 : tensor<4xi32> to memref<4xi32>
  %U5 = memref.cast %5 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%U5) : (memref<*xi32>) -> ()

  %affine2 = elliptic_curve.convert_point_type %xyzz1 : !xyzz -> !affine
  %extract_point6x, %extract_point6y = elliptic_curve.extract %affine2 : !affine -> !PFm, !PFm
  %extract_point6 = tensor.from_elements %extract_point6x, %extract_point6y : tensor<2x!PFm>
  %from_mont6 = field.from_mont %extract_point6 : tensor<2x!PF>
  %extract6 = field.extract %from_mont6 : tensor<2x!PF> -> tensor<2xi256>
  %trunc6 = arith.trunci %extract6 : tensor<2xi256> to tensor<2xi32>
  %6 = bufferization.to_buffer %trunc6 : tensor<2xi32> to memref<2xi32>
  %U6 = memref.cast %6 : memref<2xi32> to memref<*xi32>
  func.call @printMemrefI32(%U6) : (memref<*xi32>) -> ()

  %jacobian6 = elliptic_curve.scalar_mul %var7, %affine2 : !SFm, !affine -> !jacobian
  %affine2_1 = elliptic_curve.convert_point_type %jacobian6 : !jacobian -> !affine
  %jacobian6_1 = elliptic_curve.convert_point_type %affine2_1 : !affine -> !jacobian
  %extract_point7x, %extract_point7y, %extract_point7z = elliptic_curve.extract %jacobian6_1 : !jacobian -> !PFm, !PFm, !PFm
  %extract_point7 = tensor.from_elements %extract_point7x, %extract_point7y, %extract_point7z : tensor<3x!PFm>
  %from_mont7 = field.from_mont %extract_point7 : tensor<3x!PF>
  %extract7 = field.extract %from_mont7 : tensor<3x!PF> -> tensor<3xi256>
  %trunc7 = arith.trunci %extract7 : tensor<3xi256> to tensor<3xi32>
  %7 = bufferization.to_buffer %trunc7 : tensor<3xi32> to memref<3xi32>
  %U7 = memref.cast %7 : memref<3xi32> to memref<*xi32>
  func.call @printMemrefI32(%U7) : (memref<*xi32>) -> ()

  %affine3 = elliptic_curve.convert_point_type %jacobian6 : !jacobian -> !affine
  %extract_point8x, %extract_point8y = elliptic_curve.extract %affine3 : !affine -> !PFm, !PFm
  %extract_point8 = tensor.from_elements %extract_point8x, %extract_point8y : tensor<2x!PFm>
  %from_mont8 = field.from_mont %extract_point8 : tensor<2x!PF>
  %extract8 = field.extract %from_mont8 : tensor<2x!PF> -> tensor<2xi256>
  %trunc8 = arith.trunci %extract8 : tensor<2xi256> to tensor<2xi32>
  %8 = bufferization.to_buffer %trunc8 : tensor<2xi32> to memref<2xi32>
  %U8 = memref.cast %8 : memref<2xi32> to memref<*xi32>
  func.call @printMemrefI32(%U8) : (memref<*xi32>) -> ()

  %xyzz2 = elliptic_curve.add %affine3, %xyzz1 : !affine, !xyzz -> !xyzz
  %affine4 = elliptic_curve.convert_point_type %xyzz2 : !xyzz -> !affine
  %xyzz3 = elliptic_curve.convert_point_type %affine4 : !affine -> !xyzz
  %extract_point9x, %extract_point9y, %extract_point9zz, %extract_point9zzz = elliptic_curve.extract %xyzz3 : !xyzz -> !PFm, !PFm, !PFm, !PFm
  %extract_point9 = tensor.from_elements %extract_point9x, %extract_point9y, %extract_point9zz, %extract_point9zzz : tensor<4x!PFm>
  %from_mont9 = field.from_mont %extract_point9 : tensor<4x!PF>
  %extract9 = field.extract %from_mont9 : tensor<4x!PF> -> tensor<4xi256>
  %trunc9 = arith.trunci %extract9 : tensor<4xi256> to tensor<4xi32>
  %9 = bufferization.to_buffer %trunc9 : tensor<4xi32> to memref<4xi32>
  %U9 = memref.cast %9 : memref<4xi32> to memref<*xi32>
  func.call @printMemrefI32(%U9) : (memref<*xi32>) -> ()

  return
}

// CHECK_TEST_OPS_IN_ORDER: [-662897360, -662897348, 4]
// CHECK_TEST_OPS_IN_ORDER: [97344, -723639993, 312]
// CHECK_TEST_OPS_IN_ORDER: [97344, 60742656, 312]
// CHECK_TEST_OPS_IN_ORDER: [-2122515129, -662897337, -751288320]
// CHECK_TEST_OPS_IN_ORDER: [1, 2, 1, 1]
// CHECK_TEST_OPS_IN_ORDER: [1, 2]
// CHECK_TEST_OPS_IN_ORDER: [-1409294216, 1624551326, 1]
// CHECK_TEST_OPS_IN_ORDER: [-1409294216, 1624551326]
// CHECK_TEST_OPS_IN_ORDER: [741370467, 1115824161, 1, 1]
