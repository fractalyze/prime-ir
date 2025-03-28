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
  // CHECK: %[[AFFINE1:.*]] = elliptic_curve.point [[TMP1:.*]] : ![[AF:.*]]
  %affine1 = elliptic_curve.point 1, 5 : !affine
  // CHECK: %[[JACOBIAN1:.*]] = elliptic_curve.point [[TMP2:.*]] : ![[JA:.*]]
  %jacobian1 = elliptic_curve.point 1, 5, 2 : !jacobian
  // CHECK: %[[XYZZ1:.*]] = elliptic_curve.point [[TMP3:.*]] : ![[XY:.*]]
  %xyzz1 = elliptic_curve.point 1, 5, 4, 3 : !xyzz

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
  // CHECK: %[[AFFINE1:.*]] = elliptic_curve.point [[TMP1:.*]] : ![[AF:.*]]
  %affine1 = elliptic_curve.point 1, 5 : !affine
  // CHECK: %[[AFFINE2:.*]] = elliptic_curve.point
  %affine2 = elliptic_curve.point 3, 6 : !affine

  // CHECK: %[[JACOBIAN1:.*]] = elliptic_curve.point [[TMP2:.*]] : ![[JA:.*]]
  %jacobian1 = elliptic_curve.point 1, 5, 2 : !jacobian
  // CHECK: %[[JACOBIAN2:.*]] = elliptic_curve.point
  %jacobian2 = elliptic_curve.point 3, 6, 1 : !jacobian

  // CHECK: %[[XYZZ1:.*]] = elliptic_curve.point [[TMP3:.*]] : ![[XY:.*]]
  %xyzz1 = elliptic_curve.point 1, 5, 4, 3 : !xyzz
  // CHECK: %[[XYZZ2:.*]] = elliptic_curve.point
  %xyzz2 = elliptic_curve.point 3, 6, 1, 2 : !xyzz

  // affine, affine -> jacobian
  // CHECK: %[[AFFINE3:.*]] = elliptic_curve.add %[[AFFINE1]], %[[AFFINE2]] : ![[AF]], ![[AF]] -> ![[JA]]
  %affine3 = elliptic_curve.add %affine1, %affine2 : !affine, !affine -> !jacobian
  // affine, jacobian -> jacobian
  // CHECK: %[[JACOBIAN3:.*]] = elliptic_curve.add %[[AFFINE1]], %[[JACOBIAN1]] : ![[AF]], ![[JA]] -> ![[JA]]
  %jacobian3 = elliptic_curve.add %affine1, %jacobian1 : !affine, !jacobian -> !jacobian
  // CHECK: %[[JACOBIAN10:.*]] = elliptic_curve.add %[[JACOBIAN1]], %[[AFFINE1]] : ![[JA]], ![[AF]] -> ![[JA]]
  %jacobian10 = elliptic_curve.add %jacobian1, %affine1 : !jacobian, !affine -> !jacobian
  // affine, xyzz -> xyzz
  // CHECK: %[[XYZZ3:.*]] = elliptic_curve.add %[[AFFINE1]], %[[XYZZ1]] : ![[AF]], ![[XY]] -> ![[XY]]
  %xyzz3 = elliptic_curve.add %affine1, %xyzz1 : !affine, !xyzz -> !xyzz
  // jacobian, jacobian -> jacobian
  // CHECK: %[[JACOBIAN4:.*]] = elliptic_curve.add %[[JACOBIAN1]], %[[JACOBIAN2]] : ![[JA]], ![[JA]] -> ![[JA]]
  %jacobian4 = elliptic_curve.add %jacobian1, %jacobian2 : !jacobian, !jacobian -> !jacobian
  // xyzz, xyzz -> xyzz
  // CHECK: %[[XYZZ4:.*]] = elliptic_curve.add %[[XYZZ1]], %[[XYZZ2]] : ![[XY]], ![[XY]] -> ![[XY]]
  %xyzz4 = elliptic_curve.add %xyzz1, %xyzz2 : !xyzz, !xyzz -> !xyzz
  return
}

// CHECK-LABEL: @test_sub
func.func @test_sub() {
  // CHECK: %[[AFFINE1:.*]] = elliptic_curve.point [[TMP1:.*]] : ![[AF:.*]]
  %affine1 = elliptic_curve.point 1, 5 : !affine
  // CHECK: %[[AFFINE2:.*]] = elliptic_curve.point
  %affine2 = elliptic_curve.point 3, 6 : !affine

  // CHECK: %[[JACOBIAN1:.*]] = elliptic_curve.point [[TMP2:.*]] : ![[JA:.*]]
  %jacobian1 = elliptic_curve.point 1, 5, 2 : !jacobian
  // CHECK: %[[JACOBIAN2:.*]] = elliptic_curve.point
  %jacobian2 = elliptic_curve.point 3, 6, 1 : !jacobian

  // CHECK: %[[XYZZ1:.*]] = elliptic_curve.point [[TMP3:.*]] : ![[XY:.*]]
  %xyzz1 = elliptic_curve.point 1, 5, 4, 3 : !xyzz
  // CHECK: %[[XYZZ2:.*]] = elliptic_curve.point
  %xyzz2 = elliptic_curve.point 3, 6, 1, 2 : !xyzz

  // affine, affine -> jacobian
  // CHECK: %[[AFFINE3:.*]] = elliptic_curve.sub %[[AFFINE1]], %[[AFFINE2]] : ![[AF]], ![[AF]] -> ![[JA]]
  %affine3 = elliptic_curve.sub %affine1, %affine2 : !affine, !affine -> !jacobian
  // affine, jacobian -> jacobian
  // CHECK: %[[JACOBIAN3:.*]] = elliptic_curve.sub %[[AFFINE1]], %[[JACOBIAN1]] : ![[AF]], ![[JA]] -> ![[JA]]
  %jacobian3 = elliptic_curve.sub %affine1, %jacobian1 : !affine, !jacobian -> !jacobian
  // CHECK: %[[JACOBIAN10:.*]] = elliptic_curve.sub %[[JACOBIAN1]], %[[AFFINE1]] : ![[JA]], ![[AF]] -> ![[JA]]
  %jacobian10 = elliptic_curve.sub %jacobian1, %affine1 : !jacobian, !affine -> !jacobian
  // affine, xyzz -> xyzz
  // CHECK: %[[XYZZ3:.*]] = elliptic_curve.sub %[[AFFINE1]], %[[XYZZ1]] : ![[AF]], ![[XY]] -> ![[XY]]
  %xyzz3 = elliptic_curve.sub %affine1, %xyzz1 : !affine, !xyzz -> !xyzz
  // jacobian, jacobian -> jacobian
  // CHECK: %[[JACOBIAN4:.*]] = elliptic_curve.sub %[[JACOBIAN1]], %[[JACOBIAN2]] : ![[JA]], ![[JA]] -> ![[JA]]
  %jacobian4 = elliptic_curve.sub %jacobian1, %jacobian2 : !jacobian, !jacobian -> !jacobian
  // xyzz, xyzz -> xyzz
  // CHECK: %[[XYZZ4:.*]] = elliptic_curve.sub %[[XYZZ1]], %[[XYZZ2]] : ![[XY]], ![[XY]] -> ![[XY]]
  %xyzz4 = elliptic_curve.sub %xyzz1, %xyzz2 : !xyzz, !xyzz -> !xyzz
  return
}

// CHECK-LABEL: @test_negation
func.func @test_negation() {
  // CHECK: %[[AFFINE1:.*]] = elliptic_curve.point [[TMP1:.*]] : ![[AF:.*]]
  %affine1 = elliptic_curve.point 1, 5 : !affine
  // CHECK: %[[AFFINE2:.*]] = elliptic_curve.neg %[[AFFINE1]] : ![[AF]]
  %affine2 = elliptic_curve.neg %affine1 : !affine

  // CHECK: %[[JACOBIAN1:.*]] = elliptic_curve.point [[TMP2:.*]] : ![[JA:.*]]
  %jacobian1 = elliptic_curve.point 1, 5, 2 : !jacobian
  // CHECK: %[[JACOBIAN2:.*]] = elliptic_curve.neg %[[JACOBIAN1]] : ![[JA]]
  %jacobian2 = elliptic_curve.neg %jacobian1 : !jacobian

  // CHECK: %[[XYZZ1:.*]] = elliptic_curve.point [[TMP3:.*]] : ![[XY:.*]]
  %xyzz1 = elliptic_curve.point 1, 5, 4, 3 : !xyzz
  // CHECK: %[[XYZZ2:.*]] = elliptic_curve.neg %[[XYZZ1]] : ![[XY]]
  %xyzz2 = elliptic_curve.neg %xyzz1 : !xyzz
  return
}

// CHECK-LABEL: @test_double
func.func @test_double() {
  // CHECK: %[[AFFINE1:.*]] = elliptic_curve.point [[TMP1:.*]] : ![[AF:.*]]
  %affine1 = elliptic_curve.point 1, 5 : !affine
  // CHECK: %[[AFFINE2:.*]] = elliptic_curve.dbl %[[AFFINE1]] : ![[AF]] -> ![[JA:.*]]
  %affine2 = elliptic_curve.dbl %affine1 : !affine -> !jacobian

  // CHECK: %[[JACOBIAN1:.*]] = elliptic_curve.point [[TMP2:.*]] : ![[JA]]
  %jacobian1 = elliptic_curve.point 1, 5, 2 : !jacobian
  // CHECK: %[[JACOBIAN2:.*]] = elliptic_curve.dbl %[[JACOBIAN1]] : ![[JA]] -> ![[JA]]
  %jacobian2 = elliptic_curve.dbl %jacobian1 : !jacobian -> !jacobian

  // CHECK: %[[XYZZ1:.*]] = elliptic_curve.point [[TMP3:.*]] : ![[XY:.*]]
  %xyzz1 = elliptic_curve.point 1, 5, 4, 3 : !xyzz
  // CHECK: %[[XYZZ2:.*]] = elliptic_curve.dbl %[[XYZZ1]] : ![[XY]] -> ![[XY]]
  %xyzz2 = elliptic_curve.dbl %xyzz1 : !xyzz -> !xyzz
  return
}

// CHECK-LABEL: @test_scalar_mul
func.func @test_scalar_mul() {
  // CHECK: %[[SCALAR1:.*]] = field.pf.constant 1 : ![[PF:.*]]
  %scalar1 = field.pf.constant 1 : !PF
  // CHECK: %[[SCALAR2:.*]] = field.pf.constant 7 : ![[PF]]
  %scalar2 = field.pf.constant 7 : !PF

  // CHECK: %[[AFFINE1:.*]] = elliptic_curve.point [[TMP1:.*]] : ![[AF:.*]]
  %affine1 = elliptic_curve.point 1, 5 : !affine
  // CHECK: %[[AFFINE2:.*]] = elliptic_curve.scalar_mul %[[AFFINE1]], %[[SCALAR1]] : ![[AF]], ![[PF]] -> ![[JA:.*]]
  %affine2 = elliptic_curve.scalar_mul %affine1, %scalar1 : !affine, !PF -> !jacobian
  // CHECK: %[[JACOBIAN4:.*]] = elliptic_curve.scalar_mul %[[AFFINE1]], %[[SCALAR2]] : ![[AF]], ![[PF]] -> ![[JA]]
  %jacobian4 = elliptic_curve.scalar_mul %affine1, %scalar2 : !affine, !PF -> !jacobian

  // CHECK: %[[JACOBIAN1:.*]] = elliptic_curve.point [[TMP2:.*]] : ![[JA]]
  %jacobian1 = elliptic_curve.point 1, 5, 2 : !jacobian
  // CHECK: %[[JACOBIAN2:.*]] = elliptic_curve.scalar_mul %[[JACOBIAN1]], %[[SCALAR1]] : ![[JA]], ![[PF]] -> ![[JA]]
  %jacobian2 = elliptic_curve.scalar_mul %jacobian1, %scalar1 : !jacobian, !PF -> !jacobian
  // CHECK: %[[JACOBIAN3:.*]] = elliptic_curve.scalar_mul %[[JACOBIAN1]], %[[SCALAR2]] : ![[JA]], ![[PF]] -> ![[JA]]
  %jacobian3 = elliptic_curve.scalar_mul %jacobian1, %scalar2 : !jacobian, !PF -> !jacobian

  // CHECK: %[[XYZZ1:.*]] = elliptic_curve.point [[TMP3:.*]] : ![[XY:.*]]
  %xyzz1 = elliptic_curve.point 1, 5, 4, 3 : !xyzz
  // CHECK: %[[XYZZ2:.*]] = elliptic_curve.scalar_mul %[[XYZZ1]], %[[SCALAR1]] : ![[XY]], ![[PF]] -> ![[XY]]
  %xyzz2 = elliptic_curve.scalar_mul %xyzz1, %scalar1 : !xyzz, !PF -> !xyzz
  // CHECK: %[[XYZZ3:.*]] = elliptic_curve.scalar_mul %[[XYZZ1]], %[[SCALAR2]] : ![[XY]], ![[PF]] -> ![[XY]]
  %xyzz3 = elliptic_curve.scalar_mul %xyzz1, %scalar2 : !xyzz, !PF -> !xyzz
  return
}
