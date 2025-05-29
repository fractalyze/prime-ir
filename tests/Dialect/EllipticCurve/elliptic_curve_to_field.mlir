// RUN: zkir-opt -elliptic-curve-to-field --split-input-file %s | FileCheck %s --enable-var-scope

!PF = !field.pf<97:i32>

#1 = #field.pf.elem<1:i32> : !PF
#2 = #field.pf.elem<2:i32> : !PF
#3 = #field.pf.elem<3:i32> : !PF
#4 = #field.pf.elem<4:i32> : !PF

#curve = #elliptic_curve.sw<#1, #2, (#3, #4)>
!affine = !elliptic_curve.affine<#curve>
!jacobian = !elliptic_curve.jacobian<#curve>
!xyzz = !elliptic_curve.xyzz<#curve>

#beta = #field.pf.elem<96:i32> : !PF
!QF = !field.f2<!PF, #beta>
#f2_elem = #field.f2.elem<#1, #2> : !QF
#g2curve = #elliptic_curve.sw<#f2_elem, #f2_elem, (#f2_elem, #f2_elem)>
!g2affine = !elliptic_curve.affine<#g2curve>
!g2jacobian = !elliptic_curve.jacobian<#g2curve>
!g2xyzz = !elliptic_curve.xyzz<#g2curve>

// CHECK-LABEL: @test_intialization_and_conversion
func.func @test_intialization_and_conversion() {
  // CHECK: %[[VAR1:.*]] = field.constant #[[ATTR1:.*]] : ![[PF:.*]]
  %var1 = field.constant 1 : !PF
  // CHECK: %[[VAR2:.*]] = field.constant #[[ATTR2:.*]] : ![[PF]]
  %var2 = field.constant 2 : !PF
  // CHECK: %[[VAR4:.*]] = field.constant #[[ATTR4:.*]] : ![[PF]]
  %var4 = field.constant 4 : !PF
  // CHECK: %[[VAR5:.*]] = field.constant #[[ATTR5:.*]] : ![[PF]]
  %var5 = field.constant 5 : !PF
  // CHECK: %[[VAR8:.*]] = field.constant #[[ATTR8:.*]] : ![[PF]]
  %var8 = field.constant 8 : !PF

  %g2_var1 = field.constant 1, 1 : !QF
  %affine_g2 = elliptic_curve.point %g2_var1, %g2_var1 : !g2affine

  // CHECK-NOT: elliptic_curve.point
  %affine1 = elliptic_curve.point %var1, %var5 : !affine
  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : !jacobian
  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : !xyzz

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
  %var1 = field.constant 1 : !PF
  %var2 = field.constant 2 : !PF
  %var3 = field.constant 3 : !PF
  %var4 = field.constant 4 : !PF
  %var5 = field.constant 5 : !PF
  %var6 = field.constant 6 : !PF
  %var8 = field.constant 8 : !PF

  %affine1 = elliptic_curve.point %var1, %var5 : !affine
  %affine2 = elliptic_curve.point %var3, %var6 : !affine

  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : !jacobian
  %jacobian2 = elliptic_curve.point %var3, %var6, %var1 : !jacobian

  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : !xyzz
  %xyzz2 = elliptic_curve.point %var3, %var6, %var1, %var1 : !xyzz

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
  %var1 = field.constant 1 : !PF
  %var2 = field.constant 2 : !PF
  %var4 = field.constant 4 : !PF
  %var5 = field.constant 5 : !PF
  %var8 = field.constant 8 : !PF

  %affine1 = elliptic_curve.point %var1, %var5 : !affine
  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : !jacobian
  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : !xyzz

  // CHECK-NOT: elliptic_curve.double
  %affine2 = elliptic_curve.double %affine1 : !affine -> !jacobian
  %jacobian2 = elliptic_curve.double %jacobian1 : !jacobian -> !jacobian
  %xyzz2 = elliptic_curve.double %xyzz1 : !xyzz -> !xyzz
  return
}

// CHECK-LABEL: @test_negation
func.func @test_negation() {
  %var1 = field.constant 1 : !PF
  %var2 = field.constant 2 : !PF
  %var4 = field.constant 4 : !PF
  %var5 = field.constant 5 : !PF
  %var8 = field.constant 8 : !PF

  %affine1 = elliptic_curve.point %var1, %var5 : !affine
  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : !jacobian
  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : !xyzz

  // CHECK-NOT: elliptic_curve.negate
  %affine2 = elliptic_curve.negate %affine1 : !affine
  %jacobian2 = elliptic_curve.negate %jacobian1 : !jacobian
  %xyzz2 = elliptic_curve.negate %xyzz1 : !xyzz
  return
}

// CHECK-LABEL: @test_subtraction
func.func @test_subtraction() {
  %var1 = field.constant 1 : !PF
  %var2 = field.constant 2 : !PF
  %var3 = field.constant 3 : !PF
  %var4 = field.constant 4 : !PF
  %var5 = field.constant 5 : !PF
  %var6 = field.constant 6 : !PF
  %var8 = field.constant 8 : !PF

  %affine1 = elliptic_curve.point %var1, %var5 : !affine
  %affine2 = elliptic_curve.point %var3, %var6 : !affine

  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : !jacobian
  %jacobian2 = elliptic_curve.point %var3, %var6, %var1 : !jacobian

  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : !xyzz
  %xyzz2 = elliptic_curve.point %var3, %var6, %var1, %var1 : !xyzz

  // CHECK-NOT: elliptic_curve.sub
  // affine, affine -> jacobian
  %affine3 = elliptic_curve.sub %affine1, %affine2 : !affine, !affine -> !jacobian
  // affine, jacobian -> jacobian
  %jacobian3 = elliptic_curve.sub %affine1, %jacobian1 : !affine, !jacobian -> !jacobian
  %jacobian4 = elliptic_curve.sub %jacobian1, %affine1 : !jacobian, !affine -> !jacobian
  // affine, xyzz -> xyzz
  %xyzz3 = elliptic_curve.sub %affine1, %xyzz1 : !affine, !xyzz -> !xyzz
  %xyzz4 = elliptic_curve.sub %xyzz1, %affine1 : !xyzz, !affine -> !xyzz
  // jacobian, jacobian -> jacobian
  %jacobian5 = elliptic_curve.sub %jacobian1, %jacobian2 : !jacobian, !jacobian -> !jacobian
  // xyzz, xyzz -> xyzz
  %xyzz5 = elliptic_curve.sub %xyzz1, %xyzz2 : !xyzz, !xyzz -> !xyzz
  return
}

// CHECK-LABEL: @test_scalar_mul
func.func @test_scalar_mul() {
  %var1 = field.constant 1 : !PF
  %var2 = field.constant 2 : !PF
  %var4 = field.constant 4 : !PF
  %var5 = field.constant 5 : !PF
  %var8 = field.constant 8 : !PF

  %affine1 = elliptic_curve.point %var1, %var5 : !affine
  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : !jacobian
  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : !xyzz

  // CHECK-NOT: elliptic_curve.scalar_mul
  %jacobian2 = elliptic_curve.scalar_mul %var1, %affine1 : !PF, !affine -> !jacobian
  %jacobian3 = elliptic_curve.scalar_mul %var8, %affine1 : !PF, !affine -> !jacobian

  %jacobian4 = elliptic_curve.scalar_mul %var1, %jacobian1 : !PF, !jacobian -> !jacobian
  %jacobian5 = elliptic_curve.scalar_mul %var8, %jacobian1 : !PF, !jacobian -> !jacobian

  %xyzz2 = elliptic_curve.scalar_mul %var1, %xyzz1 : !PF, !xyzz -> !xyzz
  %xyzz3 = elliptic_curve.scalar_mul %var8, %xyzz1 : !PF, !xyzz -> !xyzz
  return
}

func.func @test_msm() {
  %var1 = field.constant 1 : !PF
  %var5 = field.constant 5 : !PF

  %scalars = tensor.from_elements %var1, %var5, %var5 : tensor<3x!PF>
  %affine1 = elliptic_curve.point %var1, %var5 : !affine
  %points = tensor.from_elements %affine1, %affine1, %affine1 : tensor<3x!affine>
  %msm_result = elliptic_curve.msm %scalars, %points : tensor<3x!PF>, tensor<3x!affine> -> !jacobian
  return
}

func.func @test_g2_msm(%scalars: tensor<3x!PF>, %points: tensor<3x!g2affine>) {
  %msm_result = elliptic_curve.msm %scalars, %points : tensor<3x!PF>, tensor<3x!g2affine> -> !g2jacobian
  return
}

func.func @test_memref(%arg0: memref<3x!affine>, %arg1: memref<3x!affine>) {
  %c0 = arith.constant 0 : index
  %p0 = memref.load %arg0[%c0] : memref<3x!affine>
  %p1 = memref.load %arg1[%c0] : memref<3x!affine>
  %sum = elliptic_curve.add %p0, %p1 : !affine, !affine -> !jacobian
  %affine = elliptic_curve.convert_point_type %sum : !jacobian -> !affine
  memref.store %affine, %arg0[%c0] : memref<3x!affine>
  return
}
