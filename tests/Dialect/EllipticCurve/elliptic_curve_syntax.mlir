// RUN: zkir-opt --split-input-file %s | FileCheck %s --enable-var-scope

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

  // CHECK: %[[AFFINE1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]] : ![[PF]] -> ![[AF:.*]]
  %affine1 = elliptic_curve.point %var1, %var5 : !PF -> !affine
  // CHECK: %[[JACOBIAN1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR2]] : ![[PF]] -> ![[JA:.*]]
  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : !PF -> !jacobian
  // CHECK: %[[XYZZ1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR4]], %[[VAR8]] : ![[PF]] -> ![[XY:.*]]
  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : !PF -> !xyzz

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
  // CHECK: %[[VAR1:.*]] = field.pf.constant 1 : ![[PF:.*]]
  %var1 = field.pf.constant 1 : !PF
  // CHECK: %[[VAR2:.*]] = field.pf.constant 2 : ![[PF]]
  %var2 = field.pf.constant 2 : !PF
  // CHECK: %[[VAR3:.*]] = field.pf.constant 3 : ![[PF]]
  %var3 = field.pf.constant 3 : !PF
  // CHECK: %[[VAR4:.*]] = field.pf.constant 4 : ![[PF]]
  %var4 = field.pf.constant 4 : !PF
  // CHECK: %[[VAR5:.*]] = field.pf.constant 5 : ![[PF]]
  %var5 = field.pf.constant 5 : !PF
  // CHECK: %[[VAR6:.*]] = field.pf.constant 6 : ![[PF]]
  %var6 = field.pf.constant 6 : !PF
  // CHECK: %[[VAR8:.*]] = field.pf.constant 8 : ![[PF]]
  %var8 = field.pf.constant 8 : !PF

  // CHECK: %[[AFFINE1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]] : ![[PF]] -> ![[AF:.*]]
  %affine1 = elliptic_curve.point %var1, %var5 : !PF -> !affine
  // CHECK: %[[AFFINE2:.*]] = elliptic_curve.point %[[VAR3]], %[[VAR6]] : ![[PF]] -> ![[AF]]
  %affine2 = elliptic_curve.point %var3, %var6 : !PF -> !affine

  // CHECK: %[[JACOBIAN1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR2]] : ![[PF]] -> ![[JA:.*]]
  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : !PF -> !jacobian
  // CHECK: %[[JACOBIAN2:.*]] = elliptic_curve.point %[[VAR3]], %[[VAR6]], %[[VAR1]] : ![[PF]] -> ![[JA]]
  %jacobian2 = elliptic_curve.point %var3, %var6, %var1 : !PF -> !jacobian

  // CHECK: %[[XYZZ1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR4]], %[[VAR8]] : ![[PF]] -> ![[XY:.*]]
  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : !PF -> !xyzz
  // CHECK: %[[XYZZ2:.*]] = elliptic_curve.point %[[VAR3]], %[[VAR6]], %[[VAR1]], %[[VAR1]] : ![[PF]] -> ![[XY]]
  %xyzz2 = elliptic_curve.point %var3, %var6, %var1, %var1 : !PF -> !xyzz

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
  // CHECK: %[[VAR1:.*]] = field.pf.constant 1 : ![[PF:.*]]
  %var1 = field.pf.constant 1 : !PF
  // CHECK: %[[VAR2:.*]] = field.pf.constant 2 : ![[PF]]
  %var2 = field.pf.constant 2 : !PF
  // CHECK: %[[VAR3:.*]] = field.pf.constant 3 : ![[PF]]
  %var3 = field.pf.constant 3 : !PF
  // CHECK: %[[VAR4:.*]] = field.pf.constant 4 : ![[PF]]
  %var4 = field.pf.constant 4 : !PF
  // CHECK: %[[VAR5:.*]] = field.pf.constant 5 : ![[PF]]
  %var5 = field.pf.constant 5 : !PF
  // CHECK: %[[VAR6:.*]] = field.pf.constant 6 : ![[PF]]
  %var6 = field.pf.constant 6 : !PF
  // CHECK: %[[VAR8:.*]] = field.pf.constant 8 : ![[PF]]
  %var8 = field.pf.constant 8 : !PF

  // CHECK: %[[AFFINE1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]] : ![[PF]] -> ![[AF:.*]]
  %affine1 = elliptic_curve.point %var1, %var5 : !PF -> !affine
  // CHECK: %[[AFFINE2:.*]] = elliptic_curve.point %[[VAR3]], %[[VAR6]] : ![[PF]] -> ![[AF]]
  %affine2 = elliptic_curve.point %var3, %var6 : !PF -> !affine

  // CHECK: %[[JACOBIAN1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR2]] : ![[PF]] -> ![[JA:.*]]
  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : !PF -> !jacobian
  // CHECK: %[[JACOBIAN2:.*]] = elliptic_curve.point %[[VAR3]], %[[VAR6]], %[[VAR1]] : ![[PF]] -> ![[JA]]
  %jacobian2 = elliptic_curve.point %var3, %var6, %var1 : !PF -> !jacobian

  // CHECK: %[[XYZZ1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR4]], %[[VAR8]] : ![[PF]] -> ![[XY:.*]]
  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : !PF -> !xyzz
  // CHECK: %[[XYZZ2:.*]] = elliptic_curve.point %[[VAR3]], %[[VAR6]], %[[VAR1]], %[[VAR1]] : ![[PF]] -> ![[XY]]
  %xyzz2 = elliptic_curve.point %var3, %var6, %var1, %var1 : !PF -> !xyzz

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

  // CHECK: %[[AFFINE1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]] : ![[PF]] -> ![[AF:.*]]
  %affine1 = elliptic_curve.point %var1, %var5 : !PF -> !affine
  // CHECK: %[[AFFINE2:.*]] = elliptic_curve.neg %[[AFFINE1]] : ![[AF]]
  %affine2 = elliptic_curve.neg %affine1 : !affine

  // CHECK: %[[JACOBIAN1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR2]] : ![[PF]] -> ![[JA:.*]]
  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : !PF -> !jacobian
  // CHECK: %[[JACOBIAN2:.*]] = elliptic_curve.neg %[[JACOBIAN1]] : ![[JA]]
  %jacobian2 = elliptic_curve.neg %jacobian1 : !jacobian

  // CHECK: %[[XYZZ1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR4]], %[[VAR8]] : ![[PF]] -> ![[XY:.*]]
  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : !PF -> !xyzz
  // CHECK: %[[XYZZ2:.*]] = elliptic_curve.neg %[[XYZZ1]] : ![[XY]]
  %xyzz2 = elliptic_curve.neg %xyzz1 : !xyzz
  return
}

// CHECK-LABEL: @test_double
func.func @test_double() {
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

  // CHECK: %[[AFFINE1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]] : ![[PF]] -> ![[AF:.*]]
  %affine1 = elliptic_curve.point %var1, %var5 : !PF -> !affine
  // CHECK: %[[AFFINE2:.*]] = elliptic_curve.dbl %[[AFFINE1]] : ![[AF]] -> ![[JA:.*]]
  %affine2 = elliptic_curve.dbl %affine1 : !affine -> !jacobian

  // CHECK: %[[JACOBIAN1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR2]] : ![[PF]] -> ![[JA:.*]]
  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : !PF -> !jacobian
  // CHECK: %[[JACOBIAN2:.*]] = elliptic_curve.dbl %[[JACOBIAN1]] : ![[JA]] -> ![[JA]]
  %jacobian2 = elliptic_curve.dbl %jacobian1 : !jacobian -> !jacobian

  // CHECK: %[[XYZZ1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR4]], %[[VAR8]] : ![[PF]] -> ![[XY:.*]]
  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : !PF -> !xyzz
  // CHECK: %[[XYZZ2:.*]] = elliptic_curve.dbl %[[XYZZ1]] : ![[XY]] -> ![[XY]]
  %xyzz2 = elliptic_curve.dbl %xyzz1 : !xyzz -> !xyzz
  return
}

// CHECK-LABEL: @test_scalar_mul
func.func @test_scalar_mul() {
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

  // CHECK: %[[AFFINE1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]] : ![[PF]] -> ![[AF:.*]]
  %affine1 = elliptic_curve.point %var1, %var5 : !PF -> !affine
  // CHECK: %[[AFFINE2:.*]] = elliptic_curve.scalar_mul %[[AFFINE1]], %[[VAR1]] : ![[AF]], ![[PF]] -> ![[JA:.*]]
  %affine2 = elliptic_curve.scalar_mul %affine1, %var1 : !affine, !PF -> !jacobian
  // CHECK: %[[JACOBIAN4:.*]] = elliptic_curve.scalar_mul %[[AFFINE1]], %[[VAR8]] : ![[AF]], ![[PF]] -> ![[JA]]
  %jacobian4 = elliptic_curve.scalar_mul %affine1, %var8 : !affine, !PF -> !jacobian

  // CHECK: %[[JACOBIAN1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR2]] : ![[PF]] -> ![[JA:.*]]
  %jacobian1 = elliptic_curve.point %var1, %var5, %var2 : !PF -> !jacobian
  // CHECK: %[[JACOBIAN2:.*]] = elliptic_curve.scalar_mul %[[JACOBIAN1]], %[[VAR1]] : ![[JA]], ![[PF]] -> ![[JA]]
  %jacobian2 = elliptic_curve.scalar_mul %jacobian1, %var1 : !jacobian, !PF -> !jacobian
  // CHECK: %[[JACOBIAN3:.*]] = elliptic_curve.scalar_mul %[[JACOBIAN1]], %[[VAR8]] : ![[JA]], ![[PF]] -> ![[JA]]
  %jacobian3 = elliptic_curve.scalar_mul %jacobian1, %var8 : !jacobian, !PF -> !jacobian

  // CHECK: %[[XYZZ1:.*]] = elliptic_curve.point %[[VAR1]], %[[VAR5]], %[[VAR4]], %[[VAR8]] : ![[PF]] -> ![[XY:.*]]
  %xyzz1 = elliptic_curve.point %var1, %var5, %var4, %var8 : !PF -> !xyzz
  // CHECK: %[[XYZZ2:.*]] = elliptic_curve.scalar_mul %[[XYZZ1]], %[[VAR1]] : ![[XY]], ![[PF]] -> ![[XY]]
  %xyzz2 = elliptic_curve.scalar_mul %xyzz1, %var1 : !xyzz, !PF -> !xyzz
  // CHECK: %[[XYZZ3:.*]] = elliptic_curve.scalar_mul %[[XYZZ1]], %[[VAR8]] : ![[XY]], ![[PF]] -> ![[XY]]
  %xyzz3 = elliptic_curve.scalar_mul %xyzz1, %var8 : !xyzz, !PF -> !xyzz
  return
}
