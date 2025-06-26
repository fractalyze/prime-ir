// RUN: zkir-opt -convert-linalg-to-parallel-loops -elliptic-curve-to-field -split-input-file %s | FileCheck %s -enable-var-scope

!PF = !field.pf<97:i32>
!PFm = !field.pf<97:i32, true>

#1 = #field.pf.elem<1:i32> : !PFm
#2 = #field.pf.elem<2:i32> : !PFm
#3 = #field.pf.elem<3:i32> : !PFm
#4 = #field.pf.elem<4:i32> : !PFm

#curve = #elliptic_curve.sw<#1, #2, (#3, #4)>
!affine = !elliptic_curve.affine<#curve>
!jacobian = !elliptic_curve.jacobian<#curve>
!xyzz = !elliptic_curve.xyzz<#curve>

#beta = #field.pf.elem<96:i32> : !PF
#beta_mont = #field.pf.elem<96:i32> : !PFm
!QF = !field.f2<!PF, #beta>
!QFm = !field.f2<!PFm, #beta_mont>

#f2_elem = #field.f2.elem<#1, #2> : !QFm
#g2curve = #elliptic_curve.sw<#f2_elem, #f2_elem, (#f2_elem, #f2_elem)>
!g2affine = !elliptic_curve.affine<#g2curve>
!g2jacobian = !elliptic_curve.jacobian<#g2curve>
!g2xyzz = !elliptic_curve.xyzz<#g2curve>

// CHECK-LABEL: @test_intialization_and_conversion
func.func @test_intialization_and_conversion() {
  // CHECK: %[[VAR1:.*]] = field.constant #[[ATTR1:.*]] : ![[PF:.*]]
  %var1 = field.constant 1 : !PFm
  // CHECK: %[[VAR2:.*]] = field.constant #[[ATTR2:.*]] : ![[PF]]
  %var2 = field.constant 2 : !PFm
  // CHECK: %[[VAR4:.*]] = field.constant #[[ATTR4:.*]] : ![[PF]]
  %var4 = field.constant 4 : !PFm
  // CHECK: %[[VAR5:.*]] = field.constant #[[ATTR5:.*]] : ![[PF]]
  %var5 = field.constant 5 : !PFm
  // CHECK: %[[VAR8:.*]] = field.constant #[[ATTR8:.*]] : ![[PF]]
  %var8 = field.constant 8 : !PFm

  %g2_var1 = field.constant 1, 1 : !QFm
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
  %var1 = field.constant 1 : !PFm
  %var2 = field.constant 2 : !PFm
  %var3 = field.constant 3 : !PFm
  %var4 = field.constant 4 : !PFm
  %var5 = field.constant 5 : !PFm
  %var6 = field.constant 6 : !PFm
  %var8 = field.constant 8 : !PFm

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
  %var1 = field.constant 1 : !PFm
  %var2 = field.constant 2 : !PFm
  %var4 = field.constant 4 : !PFm
  %var5 = field.constant 5 : !PFm
  %var8 = field.constant 8 : !PFm

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
  %var1 = field.constant 1 : !PFm
  %var2 = field.constant 2 : !PFm
  %var4 = field.constant 4 : !PFm
  %var5 = field.constant 5 : !PFm
  %var8 = field.constant 8 : !PFm

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
  %var1 = field.constant 1 : !PFm
  %var2 = field.constant 2 : !PFm
  %var3 = field.constant 3 : !PFm
  %var4 = field.constant 4 : !PFm
  %var5 = field.constant 5 : !PFm
  %var6 = field.constant 6 : !PFm
  %var8 = field.constant 8 : !PFm

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
  %var1 = field.constant 1 : !PFm
  %var2 = field.constant 2 : !PFm
  %var4 = field.constant 4 : !PFm
  %var5 = field.constant 5 : !PFm
  %var8 = field.constant 8 : !PFm

  %affine1 = elliptic_curve.point %var1, %var5 : !affine
  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : !jacobian
  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : !xyzz

  // CHECK-NOT: elliptic_curve.scalar_mul
  %jacobian2 = elliptic_curve.scalar_mul %var1, %affine1 : !PFm, !affine -> !jacobian
  %jacobian3 = elliptic_curve.scalar_mul %var8, %affine1 : !PFm, !affine -> !jacobian

  %jacobian4 = elliptic_curve.scalar_mul %var1, %jacobian1 : !PFm, !jacobian -> !jacobian
  %jacobian5 = elliptic_curve.scalar_mul %var8, %jacobian1 : !PFm, !jacobian -> !jacobian

  %xyzz2 = elliptic_curve.scalar_mul %var1, %xyzz1 : !PFm, !xyzz -> !xyzz
  %xyzz3 = elliptic_curve.scalar_mul %var8, %xyzz1 : !PFm, !xyzz -> !xyzz
  return
}

func.func @test_msm() {
  %c_var1 = arith.constant 1 : i32
  %c_var5 = arith.constant 5 : i32
  %var1 = field.constant 1 : !PFm
  %var5 = field.constant 5 : !PFm

  %c_scalars = tensor.from_elements %c_var1, %c_var5, %c_var5 : tensor<3xi32>
  %scalars = field.pf.encapsulate %c_scalars : tensor<3xi32> -> tensor<3x!PFm>
  %affine1 = elliptic_curve.point %var1, %var5 : !affine
  %points = tensor.from_elements %affine1, %affine1, %affine1 : tensor<3x!affine>
  %msm_result = elliptic_curve.msm %scalars, %points degree=2 : tensor<3x!PFm>, tensor<3x!affine> -> !jacobian
  return
}

func.func @test_g2_msm(%scalars: tensor<3x!PFm>, %points: tensor<3x!g2affine>) {
  %msm_result = elliptic_curve.msm %scalars, %points degree=2 : tensor<3x!PFm>, tensor<3x!g2affine> -> !g2jacobian
  return
}

func.func @test_msm_by_dot_product(%scalars: tensor<3x!PF>, %points: tensor<3x!g2jacobian>) {
  %result = tensor.empty() : tensor<!g2jacobian>
  %msm_result = linalg.dot ins(%scalars, %points : tensor<3x!PF>, tensor<3x!g2jacobian>) outs(%result: tensor<!g2jacobian>) -> tensor<!g2jacobian>
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
