// RUN: zkir-opt --split-input-file %s | FileCheck %s --enable-var-scope

!PF = !field.pf<97:i32>

#1 = #field.pf.elem<1:i32> : !PF
#2 = #field.pf.elem<2:i32> : !PF
#3 = #field.pf.elem<3:i32> : !PF
#4 = #field.pf.elem<4:i32> : !PF

#curve = #elliptic_curve.sw<#1, #2, (#3, #4)>
!affine = !elliptic_curve.affine<#curve>
!jacobian = !elliptic_curve.jacobian<#curve>
!xyzz = !elliptic_curve.xyzz<#curve>

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

  // CHECK: %[[AFFINE1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]] : ![[AF:.*]]
  %affine1 = elliptic_curve.point %var1, %var5 : !affine
  // CHECK: %[[JACOBIAN1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR2]] : ![[JA:.*]]
  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : !jacobian
  // CHECK: %[[XYZZ1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR4]], %[[VAR8]] : ![[XY:.*]]
  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : !xyzz

  // CHECK: %[[JACOBIAN2:.*]] = elliptic_curve.convert_point_type %[[AFFINE1]] : ![[AF]] -> ![[JA]]
  %jacobian2 = elliptic_curve.convert_point_type %affine1 : !affine -> !jacobian
  // CHECK: %[[XYZZ2:.*]] = elliptic_curve.convert_point_type %[[AFFINE1]] : ![[AF]] -> ![[XY]]
  %xyzz2 = elliptic_curve.convert_point_type %affine1 : !affine -> !xyzz
  // CHECK: %[[AFFINE2:.*]] = elliptic_curve.convert_point_type %[[JACOBIAN1]] : ![[JA]] -> ![[AF]]
  %affine2 = elliptic_curve.convert_point_type %jacobian1 : !jacobian -> !affine
  // CHECK: %[[XYZZ3:.*]] = elliptic_curve.convert_point_type  %[[JACOBIAN1]] : ![[JA]] -> ![[XY]]
  %xyzz3 = elliptic_curve.convert_point_type %jacobian1 : !jacobian -> !xyzz
  // CHECK: %[[AFFINE3:.*]] = elliptic_curve.convert_point_type %[[XYZZ1]] : ![[XY]] -> ![[AF]]
  %affine3 = elliptic_curve.convert_point_type %xyzz1 : !xyzz -> !affine
  // CHECK: %[[JACOBIAN3:.*]] = elliptic_curve.convert_point_type %[[XYZZ1]] : ![[XY]] -> ![[JA]]
  %jacobian3 = elliptic_curve.convert_point_type %xyzz1 : !xyzz -> !jacobian
  return
}

// CHECK-LABEL: @test_add
func.func @test_add() {
  // CHECK: %[[VAR1:.*]] = field.constant #[[ATTR1:.*]] : ![[PF:.*]]
  %var1 = field.constant 1 : !PF
  // CHECK: %[[VAR2:.*]] = field.constant #[[ATTR2:.*]] : ![[PF]]
  %var2 = field.constant 2 : !PF
  // CHECK: %[[VAR3:.*]] = field.constant #[[ATTR3:.*]] : ![[PF]]
  %var3 = field.constant 3 : !PF
  // CHECK: %[[VAR4:.*]] = field.constant #[[ATTR4:.*]] : ![[PF]]
  %var4 = field.constant 4 : !PF
  // CHECK: %[[VAR5:.*]] = field.constant #[[ATTR5:.*]] : ![[PF]]
  %var5 = field.constant 5 : !PF
  // CHECK: %[[VAR6:.*]] = field.constant #[[ATTR6:.*]] : ![[PF]]
  %var6 = field.constant 6 : !PF
  // CHECK: %[[VAR8:.*]] = field.constant #[[ATTR8:.*]] : ![[PF]]
  %var8 = field.constant 8 : !PF

  // CHECK: %[[AFFINE1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]] : ![[AF:.*]]
  %affine1 = elliptic_curve.point %var1, %var5 : !affine
  // CHECK: %[[AFFINE2:.*]] = elliptic_curve.point %[[VAR3]], %[[VAR6]] : ![[AF]]
  %affine2 = elliptic_curve.point %var3, %var6 : !affine

  // CHECK: %[[JACOBIAN1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR2]] : ![[JA:.*]]
  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : !jacobian
  // CHECK: %[[JACOBIAN2:.*]] = elliptic_curve.point %[[VAR3]], %[[VAR6]], %[[VAR1]] : ![[JA]]
  %jacobian2 = elliptic_curve.point %var3, %var6, %var1 : !jacobian

  // CHECK: %[[XYZZ1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR4]], %[[VAR8]] : ![[XY:.*]]
  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : !xyzz
  // CHECK: %[[XYZZ2:.*]] = elliptic_curve.point %[[VAR3]], %[[VAR6]], %[[VAR1]], %[[VAR1]] : ![[XY]]
  %xyzz2 = elliptic_curve.point %var3, %var6, %var1, %var1 : !xyzz

  // affine, affine -> jacobian
  // CHECK: %[[AFFINE3:.*]] = elliptic_curve.add %[[AFFINE1]], %[[AFFINE2]] : ![[AF]], ![[AF]] -> ![[JA]]
  %affine3 = elliptic_curve.add %affine1, %affine2 : !affine, !affine -> !jacobian
  // affine, jacobian -> jacobian
  // CHECK: %[[JACOBIAN3:.*]] = elliptic_curve.add %[[AFFINE1]], %[[JACOBIAN1]] : ![[AF]], ![[JA]] -> ![[JA]]
  %jacobian3 = elliptic_curve.add %affine1, %jacobian1 : !affine, !jacobian -> !jacobian
  // CHECK: %[[JACOBIAN4:.*]] = elliptic_curve.add %[[JACOBIAN1]], %[[AFFINE1]] : ![[JA]], ![[AF]] -> ![[JA]]
  %jacobian4 = elliptic_curve.add %jacobian1, %affine1 : !jacobian, !affine -> !jacobian
  // affine, xyzz -> xyzz
  // CHECK: %[[XYZZ3:.*]] = elliptic_curve.add %[[AFFINE1]], %[[XYZZ1]] : ![[AF]], ![[XY]] -> ![[XY]]
  %xyzz3 = elliptic_curve.add %affine1, %xyzz1 : !affine, !xyzz -> !xyzz
  // CHECK: %[[XYZZ4:.*]] = elliptic_curve.add %[[XYZZ1]], %[[AFFINE1]] : ![[XY]], ![[AF]] -> ![[XY]]
  %xyzz4 = elliptic_curve.add %xyzz1, %affine1 : !xyzz, !affine -> !xyzz
  // jacobian, jacobian -> jacobian
  // CHECK: %[[JACOBIAN5:.*]] = elliptic_curve.add %[[JACOBIAN1]], %[[JACOBIAN2]] : ![[JA]], ![[JA]] -> ![[JA]]
  %jacobian5 = elliptic_curve.add %jacobian1, %jacobian2 : !jacobian, !jacobian -> !jacobian
  // xyzz, xyzz -> xyzz
  // CHECK: %[[XYZZ5:.*]] = elliptic_curve.add %[[XYZZ1]], %[[XYZZ2]] : ![[XY]], ![[XY]] -> ![[XY]]
  %xyzz5 = elliptic_curve.add %xyzz1, %xyzz2 : !xyzz, !xyzz -> !xyzz
  return
}

// CHECK-LABEL: @test_sub
func.func @test_sub() {
  // CHECK: %[[VAR1:.*]] = field.constant #[[ATTR1:.*]] : ![[PF:.*]]
  %var1 = field.constant 1 : !PF
  // CHECK: %[[VAR2:.*]] = field.constant #[[ATTR2:.*]] : ![[PF]]
  %var2 = field.constant 2 : !PF
  // CHECK: %[[VAR3:.*]] = field.constant #[[ATTR3:.*]] : ![[PF]]
  %var3 = field.constant 3 : !PF
  // CHECK: %[[VAR4:.*]] = field.constant #[[ATTR4:.*]] : ![[PF]]
  %var4 = field.constant 4 : !PF
  // CHECK: %[[VAR5:.*]] = field.constant #[[ATTR5:.*]] : ![[PF]]
  %var5 = field.constant 5 : !PF
  // CHECK: %[[VAR6:.*]] = field.constant #[[ATTR6:.*]] : ![[PF]]
  %var6 = field.constant 6 : !PF
  // CHECK: %[[VAR8:.*]] = field.constant #[[ATTR8:.*]] : ![[PF]]
  %var8 = field.constant 8 : !PF

  // CHECK: %[[AFFINE1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]] : ![[AF:.*]]
  %affine1 = elliptic_curve.point %var1, %var5 : !affine
  // CHECK: %[[AFFINE2:.*]] = elliptic_curve.point %[[VAR3]], %[[VAR6]] : ![[AF]]
  %affine2 = elliptic_curve.point %var3, %var6 : !affine

  // CHECK: %[[JACOBIAN1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR2]] : ![[JA:.*]]
  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : !jacobian
  // CHECK: %[[JACOBIAN2:.*]] = elliptic_curve.point %[[VAR3]], %[[VAR6]], %[[VAR1]] : ![[JA]]
  %jacobian2 = elliptic_curve.point %var3, %var6, %var1 : !jacobian

  // CHECK: %[[XYZZ1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR4]], %[[VAR8]] : ![[XY:.*]]
  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : !xyzz
  // CHECK: %[[XYZZ2:.*]] = elliptic_curve.point %[[VAR3]], %[[VAR6]], %[[VAR1]], %[[VAR1]] : ![[XY]]
  %xyzz2 = elliptic_curve.point %var3, %var6, %var1, %var1 : !xyzz

  // affine, affine -> jacobian
  // CHECK: %[[AFFINE3:.*]] = elliptic_curve.sub %[[AFFINE1]], %[[AFFINE2]] : ![[AF]], ![[AF]] -> ![[JA]]
  %affine3 = elliptic_curve.sub %affine1, %affine2 : !affine, !affine -> !jacobian
  // affine, jacobian -> jacobian
  // CHECK: %[[JACOBIAN3:.*]] = elliptic_curve.sub %[[AFFINE1]], %[[JACOBIAN1]] : ![[AF]], ![[JA]] -> ![[JA]]
  %jacobian3 = elliptic_curve.sub %affine1, %jacobian1 : !affine, !jacobian -> !jacobian
  // CHECK: %[[JACOBIAN4:.*]] = elliptic_curve.sub %[[JACOBIAN1]], %[[AFFINE1]] : ![[JA]], ![[AF]] -> ![[JA]]
  %jacobian4 = elliptic_curve.sub %jacobian1, %affine1 : !jacobian, !affine -> !jacobian
  // affine, xyzz -> xyzz
  // CHECK: %[[XYZZ3:.*]] = elliptic_curve.sub %[[AFFINE1]], %[[XYZZ1]] : ![[AF]], ![[XY]] -> ![[XY]]
  %xyzz3 = elliptic_curve.sub %affine1, %xyzz1 : !affine, !xyzz -> !xyzz
  // CHECK: %[[XYZZ4:.*]] = elliptic_curve.sub %[[XYZZ1]], %[[AFFINE1]] : ![[XY]], ![[AF]] -> ![[XY]]
  %xyzz4 = elliptic_curve.sub %xyzz1, %affine1 : !xyzz, !affine -> !xyzz
  // jacobian, jacobian -> jacobian
  // CHECK: %[[JACOBIAN5:.*]] = elliptic_curve.sub %[[JACOBIAN1]], %[[JACOBIAN2]] : ![[JA]], ![[JA]] -> ![[JA]]
  %jacobian5 = elliptic_curve.sub %jacobian1, %jacobian2 : !jacobian, !jacobian -> !jacobian
  // xyzz, xyzz -> xyzz
  // CHECK: %[[XYZZ5:.*]] = elliptic_curve.sub %[[XYZZ1]], %[[XYZZ2]] : ![[XY]], ![[XY]] -> ![[XY]]
  %xyzz5 = elliptic_curve.sub %xyzz1, %xyzz2 : !xyzz, !xyzz -> !xyzz
  return
}

// CHECK-LABEL: @test_negation
func.func @test_negation() {
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

  // CHECK: %[[AFFINE1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]] : ![[AF:.*]]
  %affine1 = elliptic_curve.point %var1, %var5 : !affine
  // CHECK: %[[AFFINE2:.*]] = elliptic_curve.negate %[[AFFINE1]] : ![[AF]]
  %affine2 = elliptic_curve.negate %affine1 : !affine

  // CHECK: %[[JACOBIAN1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR2]] : ![[JA:.*]]
  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : !jacobian
  // CHECK: %[[JACOBIAN2:.*]] = elliptic_curve.negate %[[JACOBIAN1]] : ![[JA]]
  %jacobian2 = elliptic_curve.negate %jacobian1 : !jacobian

  // CHECK: %[[XYZZ1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR4]], %[[VAR8]] : ![[XY:.*]]
  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : !xyzz
  // CHECK: %[[XYZZ2:.*]] = elliptic_curve.negate %[[XYZZ1]] : ![[XY]]
  %xyzz2 = elliptic_curve.negate %xyzz1 : !xyzz
  return
}

// CHECK-LABEL: @test_double
func.func @test_double() {
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

  // CHECK: %[[AFFINE1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]] : ![[AF:.*]]
  %affine1 = elliptic_curve.point %var1, %var5 : !affine
  // CHECK: %[[AFFINE2:.*]] = elliptic_curve.double %[[AFFINE1]] : ![[AF]] -> ![[JA:.*]]
  %affine2 = elliptic_curve.double %affine1 : !affine -> !jacobian

  // CHECK: %[[JACOBIAN1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR2]] : ![[JA:.*]]
  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : !jacobian
  // CHECK: %[[JACOBIAN2:.*]] = elliptic_curve.double %[[JACOBIAN1]] : ![[JA]] -> ![[JA]]
  %jacobian2 = elliptic_curve.double %jacobian1 : !jacobian -> !jacobian

  // CHECK: %[[XYZZ1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR4]], %[[VAR8]] : ![[XY:.*]]
  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : !xyzz
  // CHECK: %[[XYZZ2:.*]] = elliptic_curve.double %[[XYZZ1]] : ![[XY]] -> ![[XY]]
  %xyzz2 = elliptic_curve.double %xyzz1 : !xyzz -> !xyzz
  return
}

// CHECK-LABEL: @test_scalar_mul
func.func @test_scalar_mul() {
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

  // CHECK: %[[AFFINE1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]] : ![[AF:.*]]
  %affine1 = elliptic_curve.point %var1, %var5 : !affine
  // CHECK: %[[AFFINE2:.*]] = elliptic_curve.scalar_mul %[[VAR1]], %[[AFFINE1]] : ![[PF]], ![[AF]] -> ![[JA:.*]]
  %affine2 = elliptic_curve.scalar_mul %var1, %affine1 : !PF, !affine -> !jacobian
  // CHECK: %[[JACOBIAN4:.*]] = elliptic_curve.scalar_mul %[[VAR8]], %[[AFFINE1]] : ![[PF]], ![[AF]] -> ![[JA]]
  %jacobian4 = elliptic_curve.scalar_mul %var8, %affine1 : !PF, !affine -> !jacobian

  // CHECK: %[[JACOBIAN1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR2]] : ![[JA:.*]]
  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : !jacobian
  // CHECK: %[[JACOBIAN2:.*]] = elliptic_curve.scalar_mul %[[VAR1]], %[[JACOBIAN1]] : ![[PF]], ![[JA]] -> ![[JA]]
  %jacobian2 = elliptic_curve.scalar_mul %var1, %jacobian1 : !PF, !jacobian -> !jacobian
  // CHECK: %[[JACOBIAN3:.*]] = elliptic_curve.scalar_mul %[[VAR8]], %[[JACOBIAN1]] : ![[PF]], ![[JA]] -> ![[JA]]
  %jacobian3 = elliptic_curve.scalar_mul %var8, %jacobian1 : !PF, !jacobian -> !jacobian

  // CHECK: %[[XYZZ1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR4]], %[[VAR8]] : ![[XY:.*]]
  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : !xyzz
  // CHECK: %[[XYZZ2:.*]] = elliptic_curve.scalar_mul %[[VAR1]], %[[XYZZ1]] : ![[PF]], ![[XY]] -> ![[XY]]
  %xyzz2 = elliptic_curve.scalar_mul %var1, %xyzz1 : !PF, !xyzz -> !xyzz
  // CHECK: %[[XYZZ3:.*]] = elliptic_curve.scalar_mul %[[VAR8]], %[[XYZZ1]] : ![[PF]], ![[XY]] -> ![[XY]]
  %xyzz3 = elliptic_curve.scalar_mul %var8, %xyzz1 : !PF, !xyzz -> !xyzz
  return
}

// CHECK-LABEL: @test_msm
func.func @test_msm() {
  // CHECK: %[[VAR1:.*]] = field.constant #[[ATTR1:.*]] : ![[PF:.*]]
  %var1 = field.constant 1 : !PF
  // CHECK: %[[VAR5:.*]] = field.constant #[[ATTR5:.*]] : ![[PF]]
  %var5 = field.constant 5 : !PF

  // CHECK: %[[SCALARS:.*]] = tensor.from_elements %[[VAR1]], %[[VAR5]], %[[VAR5]] : [[TPF:.*]]
  %scalars = tensor.from_elements %var1, %var5, %var5 : tensor<3x!PF>
  // CHECK: %[[AFFINE1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]] : ![[AF:.*]]
  %affine1 = elliptic_curve.point %var1, %var5 : !affine
  // CHECK: %[[POINTS:.*]] = tensor.from_elements %[[AFFINE1]], %[[AFFINE1]], %[[AFFINE1]] : [[TAF:.*]]
  %points = tensor.from_elements %affine1, %affine1, %affine1 : tensor<3x!affine>
  // CHECK: %[[MSM_RESULT:.*]] = elliptic_curve.msm %[[SCALARS]], %[[POINTS]] : [[TPF]], [[TAF]] -> ![[JA:.*]]
  %msm_result = elliptic_curve.msm %scalars, %points : tensor<3x!PF>, tensor<3x!affine> -> !jacobian
  return
}
