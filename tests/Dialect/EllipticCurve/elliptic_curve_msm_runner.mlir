// RUN: zkir-opt %s -elliptic-curve-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_msm -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_MSM < %t

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

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

// CHECK-LABEL: @test_msm
func.func @test_msm() {
  %c1 = field.constant 1 : !PF
  %c2 = field.constant 2 : !PF
  %var1 = field.to_mont %c1 : !PFm
  %var2 = field.to_mont %c2 : !PFm

  %c3 = field.constant 3 : !SF
  %c5 = field.constant 5 : !SF
  %c7 = field.constant 7 : !SF

  %scalar3 = field.to_mont %c3 : !SFm
  %scalar5 = field.to_mont %c5 : !SFm
  %scalar7 = field.to_mont %c7 : !SFm

  %jacobian1 = elliptic_curve.point %var1, %var2, %var1 : !jacobian
  %jacobian2 = elliptic_curve.double %jacobian1 : !jacobian -> !jacobian
  %jacobian3 = elliptic_curve.double %jacobian2 : !jacobian -> !jacobian
  %jacobian4 = elliptic_curve.double %jacobian3 : !jacobian -> !jacobian


  // CALCULATING TRUE VALUE OF MSM
  %scalar_mul1 = elliptic_curve.scalar_mul %scalar3, %jacobian1 : !SFm, !jacobian -> !jacobian
  %scalar_mul2 = elliptic_curve.scalar_mul %scalar3, %jacobian2 : !SFm, !jacobian -> !jacobian
  %scalar_mul3 = elliptic_curve.scalar_mul %scalar5, %jacobian3 : !SFm, !jacobian -> !jacobian
  %scalar_mul4 = elliptic_curve.scalar_mul %scalar7, %jacobian4 : !SFm, !jacobian -> !jacobian

  %add1 = elliptic_curve.add %scalar_mul1, %scalar_mul2 : !jacobian, !jacobian -> !jacobian
  %add2 = elliptic_curve.add %scalar_mul3, %scalar_mul4 : !jacobian, !jacobian -> !jacobian
  %msm_true = elliptic_curve.add %add1, %add2 : !jacobian, !jacobian -> !jacobian
  %msm_true_affine = elliptic_curve.convert_point_type %msm_true : !jacobian -> !affine

  %extract_point_x, %extract_point_y = elliptic_curve.extract %msm_true_affine : !affine -> !PFm, !PFm
  %extract_point = tensor.from_elements %extract_point_x, %extract_point_y : tensor<2x!PFm>
  %extract_point_reduced = field.from_mont %extract_point : tensor<2x!PF>
  %extract = field.extract %extract_point_reduced : tensor<2x!PF> -> tensor<2xi256>
  %trunc = arith.trunci %extract : tensor<2xi256> to tensor<2xi32>
  %mem = bufferization.to_buffer %trunc : tensor<2xi32> to memref<2xi32>
  %U = memref.cast %mem : memref<2xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()

  // RUNNING MSM
  %scalars = tensor.from_elements %scalar3, %scalar3, %scalar5, %scalar7 : tensor<4x!SFm>
  %points = tensor.from_elements %jacobian1, %jacobian2, %jacobian3, %jacobian4 : tensor<4x!jacobian>
  %msm_test = elliptic_curve.msm %scalars, %points degree=2 parallel : tensor<4x!SFm>, tensor<4x!jacobian> -> !jacobian
  %msm_test_affine = elliptic_curve.convert_point_type %msm_test : !jacobian -> !affine

  %extract_point1x, %extract_point1y = elliptic_curve.extract %msm_test_affine : !affine -> !PFm, !PFm
  %extract_point1 = tensor.from_elements %extract_point1x, %extract_point1y : tensor<2x!PFm>
  %extract_point1_reduced = field.from_mont %extract_point1 : tensor<2x!PF>
  %extract1 = field.extract %extract_point1_reduced : tensor<2x!PF> -> tensor<2xi256>
  %trunc1 = arith.trunci %extract1 : tensor<2xi256> to tensor<2xi32>
  %mem1 = bufferization.to_buffer %trunc1 : tensor<2xi32> to memref<2xi32>
  %U1 = memref.cast %mem1 : memref<2xi32> to memref<*xi32>
  func.call @printMemrefI32(%U1) : (memref<*xi32>) -> ()
  return
}

// CHECK_TEST_MSM: [-1898406218, -965466820]
// CHECK_TEST_MSM: [-1898406218, -965466820]
