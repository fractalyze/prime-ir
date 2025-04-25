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
  return
}
