// RUN: zkir-opt -elliptic-curve-to-field --split-input-file %s | FileCheck %s --enable-var-scope

!PF = !field.pf<35:i32>

#1 = #field.pf_elem<1:i32> : !PF
#2 = #field.pf_elem<2:i32> : !PF
#3 = #field.pf_elem<3:i32> : !PF
#4 = #field.pf_elem<4:i32> : !PF

#curve = #elliptic_curve.sw<#1, #2, (#3, #4)>
!affine = !elliptic_curve.affine<#curve>
!jacobian = !elliptic_curve.jacobian<#curve>
!xyzz = !elliptic_curve.xyzz<#curve>

// CHECK-LABEL: @test_intialization_and_conversion
func.func @test_intialization_and_conversion() {
  // CHECK: %[[VAR1:.*]] = field.pf.constant 1 : ![[PF:.*]]
  %var1 = field.pf.constant 1 : !PF
  // CHECK: %[[VAR2:.*]] = field.pf.constant 2 : ![[PF]]
  %var2 = field.pf.constant 2 : !PF
  // CHECK: %[[VAR4:.*]] = field.pf.constant 4 : ![[PF]]
  %var4 = field.pf.constant 4 : !PF
  // CHECK: %[[VAR5:.*]] = field.pf.constant 5 : ![[PF]]
  %var5 = field.pf.constant 5 : !PF
  // CHECK: %[[VAR8:.*]] = field.pf.constant 8 : ![[PF]]
  %var8 = field.pf.constant 8 : !PF

  // CHECK-NOT: elliptic_curve.point
  // CHECK: %[[AFFINE1:.*]] = tensor.from_elements %[[VAR1]], %[[VAR5]] : tensor<2x![[PF]]>
  %affine1 = elliptic_curve.point %var1, %var5 : !PF -> !affine
  // CHECK-NOT: elliptic_curve.point
  // CHECK: %[[JACOBIAN1:.*]] = tensor.from_elements %[[VAR1]], %[[VAR5]], %[[VAR2]] : tensor<3x![[PF]]>
  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : !PF -> !jacobian
  // CHECK-NOT: elliptic_curve.point
  // CHECK: %[[XYZZ1:.*]] = tensor.from_elements %[[VAR1]], %[[VAR5]], %[[VAR4]], %[[VAR8]] : tensor<4x![[PF]]>
  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : !PF -> !xyzz

  // CHECK-NOT: elliptic_curve.convert_point_type
  %jacobian2 = elliptic_curve.convert_point_type %affine1 : !affine -> !jacobian
  %xyzz2 = elliptic_curve.convert_point_type %affine1 : !affine -> !xyzz
  %affine2 = elliptic_curve.convert_point_type %jacobian1 : !jacobian -> !affine
  %xyzz3 = elliptic_curve.convert_point_type %jacobian1 : !jacobian -> !xyzz
  %affine3 = elliptic_curve.convert_point_type %xyzz1 : !xyzz -> !affine
  %jacobian3 = elliptic_curve.convert_point_type %xyzz1 : !xyzz -> !jacobian
  return
}

// CHECK-LABEL: @test_addition
func.func @test_addition() {
  %var1 = field.pf.constant 1 : !PF
  %var2 = field.pf.constant 2 : !PF
  %var3 = field.pf.constant 3 : !PF
  %var4 = field.pf.constant 4 : !PF
  %var5 = field.pf.constant 5 : !PF
  %var6 = field.pf.constant 6 : !PF
  %var8 = field.pf.constant 8 : !PF

  %affine1 = elliptic_curve.point %var1, %var5 : !PF -> !affine
  %affine2 = elliptic_curve.point %var3, %var6 : !PF -> !affine

  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : !PF -> !jacobian
  %jacobian2 = elliptic_curve.point %var3, %var6, %var1 : !PF -> !jacobian

  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : !PF -> !xyzz
  %xyzz2 = elliptic_curve.point %var3, %var6, %var1, %var1 : !PF -> !xyzz

  // CHECK-NOT: elliptic_curve.add
  // affine, affine -> jacobian
  %affine3 = elliptic_curve.add %affine1, %affine2 : !affine, !affine -> !jacobian
  // affine, jacobian -> jacobian
  %jacobian3 = elliptic_curve.add %affine1, %jacobian1 : !affine, !jacobian -> !jacobian
  %jacobian4 = elliptic_curve.add %jacobian1, %affine1 : !jacobian, !affine -> !jacobian
  // affine, xyzz -> xyzz
  %xyzz3 = elliptic_curve.add %affine1, %xyzz1 : !affine, !xyzz -> !xyzz
  %xyzz4 = elliptic_curve.add %xyzz1, %affine1 : !xyzz, !affine -> !xyzz
  // jacobian, jacobian -> jacobian
  %jacobian5 = elliptic_curve.add %jacobian1, %jacobian2 : !jacobian, !jacobian -> !jacobian
  // xyzz, xyzz -> xyzz
  %xyzz5 = elliptic_curve.add %xyzz1, %xyzz2 : !xyzz, !xyzz -> !xyzz
  return
}

// CHECK-LABEL: @test_double
func.func @test_double() {
  %var1 = field.pf.constant 1 : !PF
  %var2 = field.pf.constant 2 : !PF
  %var4 = field.pf.constant 4 : !PF
  %var5 = field.pf.constant 5 : !PF
  %var8 = field.pf.constant 8 : !PF

  %affine1 = elliptic_curve.point %var1, %var5 : !PF -> !affine
  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : !PF -> !jacobian
  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : !PF -> !xyzz

  // CHECK-NOT: elliptic_curve.double
  %affine2 = elliptic_curve.double %affine1 : !affine -> !jacobian
  %jacobian2 = elliptic_curve.double %jacobian1 : !jacobian -> !jacobian
  %xyzz2 = elliptic_curve.double %xyzz1 : !xyzz -> !xyzz
  return
}

// CHECK-LABEL: @test_negation
func.func @test_negation() {
  %var1 = field.pf.constant 1 : !PF
  %var2 = field.pf.constant 2 : !PF
  %var4 = field.pf.constant 4 : !PF
  %var5 = field.pf.constant 5 : !PF
  %var8 = field.pf.constant 8 : !PF

  %affine1 = elliptic_curve.point %var1, %var5 : !PF -> !affine
  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : !PF -> !jacobian
  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : !PF -> !xyzz

  // CHECK-NOT: elliptic_curve.negate
  %affine2 = elliptic_curve.negate %affine1 : !affine
  %jacobian2 = elliptic_curve.negate %jacobian1 : !jacobian
  %xyzz2 = elliptic_curve.negate %xyzz1 : !xyzz
  return
}
