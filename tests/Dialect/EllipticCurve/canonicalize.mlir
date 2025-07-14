// RUN: zkir-opt -elliptic-curve-to-field -field-to-mod-arith -canonicalize %s | FileCheck %s -enable-var-scope

!PF = !field.pf<97:i32>

// a is 0
#1 = #field.pf.elem<0:i32> : !PF
#2 = #field.pf.elem<2:i32> : !PF
#3 = #field.pf.elem<3:i32> : !PF
#4 = #field.pf.elem<4:i32> : !PF

#curve = #elliptic_curve.sw<#1, #2, (#3, #4)>
!affine = !elliptic_curve.affine<#curve>
!jacobian = !elliptic_curve.jacobian<#curve>
!xyzz = !elliptic_curve.xyzz<#curve>

// CHECK-LABEL: @test_affine_to_jacobian_double
// CHECK-SAME: (%[[ARG0:.*]]: !z97_i32, %[[ARG1:.*]]: !z97_i32) -> (!z97_i32, !z97_i32, !z97_i32) {
func.func @test_affine_to_jacobian_double(%point: !affine) -> !jacobian {
  // CHECK: %[[A0:.*]] = mod_arith.square %[[ARG0]] : !z97_i32
  // CHECK: %[[A1:.*]] = mod_arith.square %[[ARG1]] : !z97_i32
  // CHECK: %[[A2:.*]] = mod_arith.square %[[A1]] : !z97_i32
  // CHECK: %[[A3:.*]] = mod_arith.add %[[ARG0]], %[[A1]] : !z97_i32
  // CHECK: %[[A4:.*]] = mod_arith.add %[[A3]], %[[ARG0]] : !z97_i32
  // CHECK: %[[A5:.*]] = mod_arith.mul %[[A1]], %[[A4]] : !z97_i32
  // CHECK: %[[A6:.*]] = mod_arith.sub %[[A5]], %[[A2]] : !z97_i32
  // CHECK: %[[A7:.*]] = mod_arith.double %[[A6]] : !z97_i32
  // CHECK: %[[A8:.*]] = mod_arith.double %[[A0]] : !z97_i32
  // CHECK: %[[A9:.*]] = mod_arith.add %[[A8]], %[[A0]] : !z97_i32
  // CHECK: %[[A10:.*]] = mod_arith.square %[[A9]] : !z97_i32
  // CHECK: %[[A11:.*]] = mod_arith.double %[[A7]] : !z97_i32
  // CHECK: %[[A12:.*]] = mod_arith.sub %[[A10]], %[[A11]] : !z97_i32
  // CHECK: %[[A13:.*]] = mod_arith.sub %[[A7]], %[[A12]] : !z97_i32
  // CHECK: %[[A14:.*]] = mod_arith.mul %[[A9]], %[[A13]] : !z97_i32
  // CHECK: %[[A15:.*]] = mod_arith.double %[[A2]] : !z97_i32
  // CHECK: %[[A16:.*]] = mod_arith.double %[[A15]] : !z97_i32
  // CHECK: %[[A17:.*]] = mod_arith.double %[[A16]] : !z97_i32
  // CHECK: %[[A18:.*]] = mod_arith.sub %[[A14]], %[[A17]] : !z97_i32
  // CHECK: %[[A19:.*]] = mod_arith.double %[[ARG1]] : !z97_i32
  %double = elliptic_curve.double %point : !affine -> !jacobian
  // CHECK: return %[[A12]], %[[A18]], %[[A19]] : !z97_i32, !z97_i32, !z97_i32
  return %double : !jacobian
}

// CHECK-LABEL: @test_jacobian_to_jacobian_double
// CHECK-SAME: (%[[ARG0:.*]]: !z97_i32, %[[ARG1:.*]]: !z97_i32, %[[ARG2:.*]]: !z97_i32) -> (!z97_i32, !z97_i32, !z97_i32) {
func.func @test_jacobian_to_jacobian_double(%point: !jacobian) -> !jacobian {
  // CHECK: %[[J0:.*]] = mod_arith.square %[[ARG0]] : !z97_i32
  // CHECK: %[[J1:.*]] = mod_arith.square %[[ARG1]] : !z97_i32
  // CHECK: %[[J2:.*]] = mod_arith.square %[[J1]] : !z97_i32
  // CHECK: %[[J3:.*]] = mod_arith.square %[[ARG2]] : !z97_i32
  // CHECK: %[[J4:.*]] = mod_arith.add %[[ARG0]], %[[J1]] : !z97_i32
  // CHECK: %[[J5:.*]] = mod_arith.add %[[J4]], %[[ARG0]] : !z97_i32
  // CHECK: %[[J6:.*]] = mod_arith.mul %[[J1]], %[[J5]] : !z97_i32
  // CHECK: %[[J7:.*]] = mod_arith.sub %[[J6]], %[[J2]] : !z97_i32
  // CHECK: %[[J8:.*]] = mod_arith.double %[[J7]] : !z97_i32
  // CHECK: %[[J9:.*]] = mod_arith.double %[[J0]] : !z97_i32
  // CHECK: %[[J10:.*]] = mod_arith.add %[[J9]], %[[J0]] : !z97_i32
  // CHECK: %[[J11:.*]] = mod_arith.square %[[J10]] : !z97_i32
  // CHECK: %[[J12:.*]] = mod_arith.double %[[J8]] : !z97_i32
  // CHECK: %[[J13:.*]] = mod_arith.sub %[[J11]], %[[J12]] : !z97_i32
  // CHECK: %[[J14:.*]] = mod_arith.sub %[[J8]], %[[J13]] : !z97_i32
  // CHECK: %[[J15:.*]] = mod_arith.mul %[[J10]], %[[J14]] : !z97_i32
  // CHECK: %[[J16:.*]] = mod_arith.double %[[J2]] : !z97_i32
  // CHECK: %[[J17:.*]] = mod_arith.double %[[J16]] : !z97_i32
  // CHECK: %[[J18:.*]] = mod_arith.double %[[J17]] : !z97_i32
  // CHECK: %[[J19:.*]] = mod_arith.sub %[[J15]], %[[J18]] : !z97_i32
  // CHECK: %[[J20:.*]] = mod_arith.add %[[ARG1]], %[[ARG2]] : !z97_i32
  // CHECK: %[[J21:.*]] = mod_arith.add %[[J20]], %[[ARG1]] : !z97_i32
  // CHECK: %[[J22:.*]] = mod_arith.mul %[[ARG2]], %[[J21]] : !z97_i32
  // CHECK: %[[J23:.*]] = mod_arith.sub %[[J22]], %[[J3]] : !z97_i32
  %double = elliptic_curve.double %point : !jacobian -> !jacobian
  // CHECK: return %[[J13]], %[[J19]], %[[J23]] : !z97_i32, !z97_i32, !z97_i32
  return %double : !jacobian
}

// CHECK-LABEL: @test_affine_to_xyzz_double
// CHECK-SAME: (%[[ARG0:.*]]: !z97_i32, %[[ARG1:.*]]: !z97_i32) -> (!z97_i32, !z97_i32, !z97_i32, !z97_i32) {
func.func @test_affine_to_xyzz_double(%point: !affine) -> !xyzz {
  // CHECK: %[[X0:.*]] = mod_arith.double %[[ARG1]] : !z97_i32
  // CHECK: %[[X1:.*]] = mod_arith.square %[[X0]] : !z97_i32
  // CHECK: %[[X2:.*]] = mod_arith.mul %[[X0]], %[[X1]] : !z97_i32
  // CHECK: %[[X3:.*]] = mod_arith.mul %[[ARG0]], %[[X1]] : !z97_i32
  // CHECK: %[[X4:.*]] = mod_arith.square %[[ARG0]] : !z97_i32
  // CHECK: %[[X5:.*]] = mod_arith.double %[[X4]] : !z97_i32
  // CHECK: %[[X6:.*]] = mod_arith.add %[[X5]], %[[X4]] : !z97_i32
  // CHECK: %[[X7:.*]] = mod_arith.square %[[X6]] : !z97_i32
  // CHECK: %[[X8:.*]] = mod_arith.double %[[X3]] : !z97_i32
  // CHECK: %[[X9:.*]] = mod_arith.sub %[[X7]], %[[X8]] : !z97_i32
  // CHECK: %[[X10:.*]] = mod_arith.sub %[[X3]], %[[X9]] : !z97_i32
  // CHECK: %[[X11:.*]] = mod_arith.mul %[[X6]], %[[X10]] : !z97_i32
  // CHECK: %[[X12:.*]] = mod_arith.mul %[[X2]], %[[ARG1]] : !z97_i32
  // CHECK: %[[X13:.*]] = mod_arith.sub %[[X11]], %[[X12]] : !z97_i32
  %double = elliptic_curve.double %point : !affine -> !xyzz
  // CHECK: return %[[X9]], %[[X13]], %[[X1]], %[[X2]] : !z97_i32, !z97_i32, !z97_i32, !z97_i32
  return %double : !xyzz
}

// CHECK-LABEL: @test_xyzz_to_xyzz_double
// CHECK-SAME: (%[[ARG0:.*]]: !z97_i32, %[[ARG1:.*]]: !z97_i32, %[[ARG2:.*]]: !z97_i32, %[[ARG3:.*]]: !z97_i32) -> (!z97_i32, !z97_i32, !z97_i32, !z97_i32) {
func.func @test_xyzz_to_xyzz_double(%point: !xyzz) -> !xyzz {
  // CHECK: %[[X0:.*]] = mod_arith.double %[[ARG1]] : !z97_i32
  // CHECK: %[[X1:.*]] = mod_arith.square %[[X0]] : !z97_i32
  // CHECK: %[[X2:.*]] = mod_arith.mul %[[X0]], %[[X1]] : !z97_i32
  // CHECK: %[[X3:.*]] = mod_arith.mul %[[ARG0]], %[[X1]] : !z97_i32
  // CHECK: %[[X4:.*]] = mod_arith.square %[[ARG0]] : !z97_i32
  // CHECK: %[[X5:.*]] = mod_arith.double %[[X4]] : !z97_i32
  // CHECK: %[[X6:.*]] = mod_arith.add %[[X5]], %[[X4]] : !z97_i32
  // CHECK: %[[X7:.*]] = mod_arith.square %[[X6]] : !z97_i32
  // CHECK: %[[X8:.*]] = mod_arith.double %[[X3]] : !z97_i32
  // CHECK: %[[X9:.*]] = mod_arith.sub %[[X7]], %[[X8]] : !z97_i32
  // CHECK: %[[X10:.*]] = mod_arith.sub %[[X3]], %[[X9]] : !z97_i32
  // CHECK: %[[X11:.*]] = mod_arith.mul %[[X6]], %[[X10]] : !z97_i32
  // CHECK: %[[X12:.*]] = mod_arith.mul %[[X2]], %[[ARG1]] : !z97_i32
  // CHECK: %[[X13:.*]] = mod_arith.sub %[[X11]], %[[X12]] : !z97_i32
  // CHECK: %[[X14:.*]] = mod_arith.mul %[[X1]], %[[ARG2]] : !z97_i32
  // CHECK: %[[X15:.*]] = mod_arith.mul %[[X2]], %[[ARG3]] : !z97_i32
  %double = elliptic_curve.double %point : !xyzz -> !xyzz
  // CHECK: return %[[X9]], %[[X13]], %[[X14]], %[[X15]] : !z97_i32, !z97_i32, !z97_i32, !z97_i32
  return %double : !xyzz
}
