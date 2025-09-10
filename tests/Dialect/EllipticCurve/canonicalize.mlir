// RUN: cat %S/../../bn254_field_defs.mlir %S/../../bn254_ec_defs.mlir %s \
// RUN:   | zkir-opt -elliptic-curve-to-field -field-to-mod-arith -canonicalize \
// RUN:   | FileCheck %s -enable-var-scope

// CHECK-LABEL: @test_affine_to_jacobian_double
// CHECK-SAME: (%[[ARG0:.*]]: [[AFFINE:.*]]) -> [[JACOBIAN:.*]] {
func.func @test_affine_to_jacobian_double(%point: !affine) -> !jacobian {
  // CHECK: %[[COORDS:.*]]:2 = elliptic_curve.extract %[[ARG0]] : [[AFFINE]] -> [[MOD_INT:.*]], [[MOD_INT]]
  // CHECK: %[[A0:.*]] = mod_arith.square %[[COORDS]]#0 : [[MOD_INT]]
  // CHECK: %[[A1:.*]] = mod_arith.square %[[COORDS]]#1 : [[MOD_INT]]
  // CHECK: %[[A2:.*]] = mod_arith.square %[[A1]] : [[MOD_INT]]
  // CHECK: %[[A3:.*]] = mod_arith.add %[[COORDS]]#0, %[[A1]] : [[MOD_INT]]
  // CHECK: %[[A4:.*]] = mod_arith.add %[[A3]], %[[COORDS]]#0 : [[MOD_INT]]
  // CHECK: %[[A5:.*]] = mod_arith.mul %[[A1]], %[[A4]] : [[MOD_INT]]
  // CHECK: %[[A6:.*]] = mod_arith.sub %[[A5]], %[[A2]] : [[MOD_INT]]
  // CHECK: %[[A7:.*]] = mod_arith.double %[[A6]] : [[MOD_INT]]
  // CHECK: %[[A8:.*]] = mod_arith.double %[[A0]] : [[MOD_INT]]
  // CHECK: %[[A9:.*]] = mod_arith.add %[[A8]], %[[A0]] : [[MOD_INT]]
  // CHECK: %[[A10:.*]] = mod_arith.square %[[A9]] : [[MOD_INT]]
  // CHECK: %[[A11:.*]] = mod_arith.double %[[A7]] : [[MOD_INT]]
  // CHECK: %[[A12:.*]] = mod_arith.sub %[[A10]], %[[A11]] : [[MOD_INT]]
  // CHECK: %[[A13:.*]] = mod_arith.sub %[[A7]], %[[A12]] : [[MOD_INT]]
  // CHECK: %[[A14:.*]] = mod_arith.mul %[[A9]], %[[A13]] : [[MOD_INT]]
  // CHECK: %[[A15:.*]] = mod_arith.double %[[A2]] : [[MOD_INT]]
  // CHECK: %[[A16:.*]] = mod_arith.double %[[A15]] : [[MOD_INT]]
  // CHECK: %[[A17:.*]] = mod_arith.double %[[A16]] : [[MOD_INT]]
  // CHECK: %[[A18:.*]] = mod_arith.sub %[[A14]], %[[A17]] : [[MOD_INT]]
  // CHECK: %[[A19:.*]] = mod_arith.double %[[COORDS]]#1 : [[MOD_INT]]
  // CHECK: %[[RESULT:.*]] = elliptic_curve.point %[[A12]], %[[A18]], %[[A19]] : ([[MOD_INT]], [[MOD_INT]], [[MOD_INT]]) -> [[JACOBIAN]]
  %double = elliptic_curve.double %point : !affine -> !jacobian
  // CHECK: return %[[RESULT]] : [[JACOBIAN]]
  return %double : !jacobian
}

// CHECK-LABEL: @test_jacobian_to_jacobian_double
// CHECK-SAME: (%[[ARG0:.*]]: [[JACOBIAN:.*]]) -> [[JACOBIAN:.*]] {
func.func @test_jacobian_to_jacobian_double(%point: !jacobian) -> !jacobian {
  // CHECK: %[[COORDS:.*]]:3 = elliptic_curve.extract %[[ARG0]] : [[JACOBIAN]] -> [[MOD_INT:.*]], [[MOD_INT]], [[MOD_INT]]
  // CHECK: %[[J0:.*]] = mod_arith.square %[[COORDS]]#0 : [[MOD_INT]]
  // CHECK: %[[J1:.*]] = mod_arith.square %[[COORDS]]#1 : [[MOD_INT]]
  // CHECK: %[[J2:.*]] = mod_arith.square %[[J1]] : [[MOD_INT]]
  // CHECK: %[[J3:.*]] = mod_arith.square %[[COORDS]]#2 : [[MOD_INT]]
  // CHECK: %[[J4:.*]] = mod_arith.add %[[COORDS]]#0, %[[J1]] : [[MOD_INT]]
  // CHECK: %[[J5:.*]] = mod_arith.add %[[J4]], %[[COORDS]]#0 : [[MOD_INT]]
  // CHECK: %[[J6:.*]] = mod_arith.mul %[[J1]], %[[J5]] : [[MOD_INT]]
  // CHECK: %[[J7:.*]] = mod_arith.sub %[[J6]], %[[J2]] : [[MOD_INT]]
  // CHECK: %[[J8:.*]] = mod_arith.double %[[J7]] : [[MOD_INT]]
  // CHECK: %[[J9:.*]] = mod_arith.double %[[J0]] : [[MOD_INT]]
  // CHECK: %[[J10:.*]] = mod_arith.add %[[J9]], %[[J0]] : [[MOD_INT]]
  // CHECK: %[[J11:.*]] = mod_arith.square %[[J10]] : [[MOD_INT]]
  // CHECK: %[[J12:.*]] = mod_arith.double %[[J8]] : [[MOD_INT]]
  // CHECK: %[[J13:.*]] = mod_arith.sub %[[J11]], %[[J12]] : [[MOD_INT]]
  // CHECK: %[[J14:.*]] = mod_arith.sub %[[J8]], %[[J13]] : [[MOD_INT]]
  // CHECK: %[[J15:.*]] = mod_arith.mul %[[J10]], %[[J14]] : [[MOD_INT]]
  // CHECK: %[[J16:.*]] = mod_arith.double %[[J2]] : [[MOD_INT]]
  // CHECK: %[[J17:.*]] = mod_arith.double %[[J16]] : [[MOD_INT]]
  // CHECK: %[[J18:.*]] = mod_arith.double %[[J17]] : [[MOD_INT]]
  // CHECK: %[[J19:.*]] = mod_arith.sub %[[J15]], %[[J18]] : [[MOD_INT]]
  // CHECK: %[[J20:.*]] = mod_arith.add %[[COORDS]]#1, %[[COORDS]]#2 : [[MOD_INT]]
  // CHECK: %[[J21:.*]] = mod_arith.add %[[J20]], %[[COORDS]]#1 : [[MOD_INT]]
  // CHECK: %[[J22:.*]] = mod_arith.mul %[[COORDS]]#2, %[[J21]] : [[MOD_INT]]
  // CHECK: %[[J23:.*]] = mod_arith.sub %[[J22]], %[[J3]] : [[MOD_INT]]
  // CHECK: %[[RESULT:.*]] = elliptic_curve.point %[[J13]], %[[J19]], %[[J23]] : ([[MOD_INT]], [[MOD_INT]], [[MOD_INT]]) -> [[JACOBIAN]]
  %double = elliptic_curve.double %point : !jacobian -> !jacobian
  // CHECK: return %[[RESULT]] : [[JACOBIAN]]
  return %double : !jacobian
}

// CHECK-LABEL: @test_affine_to_xyzz_double
// CHECK-SAME: (%[[ARG0:.*]]: [[AFFINE:.*]]) -> [[XYZZ:.*]] {
func.func @test_affine_to_xyzz_double(%point: !affine) -> !xyzz {
  // CHECK: %[[COORDS:.*]]:2 = elliptic_curve.extract %[[ARG0]] : [[AFFINE]] -> [[MOD_INT:.*]], [[MOD_INT]]
  // CHECK: %[[X0:.*]] = mod_arith.double %[[COORDS]]#1 : [[MOD_INT]]
  // CHECK: %[[X1:.*]] = mod_arith.square %[[X0]] : [[MOD_INT]]
  // CHECK: %[[X2:.*]] = mod_arith.mul %[[X0]], %[[X1]] : [[MOD_INT]]
  // CHECK: %[[X3:.*]] = mod_arith.mul %[[COORDS]]#0, %[[X1]] : [[MOD_INT]]
  // CHECK: %[[X4:.*]] = mod_arith.square %[[COORDS]]#0 : [[MOD_INT]]
  // CHECK: %[[X5:.*]] = mod_arith.double %[[X4]] : [[MOD_INT]]
  // CHECK: %[[X6:.*]] = mod_arith.add %[[X5]], %[[X4]] : [[MOD_INT]]
  // CHECK: %[[X7:.*]] = mod_arith.square %[[X6]] : [[MOD_INT]]
  // CHECK: %[[X8:.*]] = mod_arith.double %[[X3]] : [[MOD_INT]]
  // CHECK: %[[X9:.*]] = mod_arith.sub %[[X7]], %[[X8]] : [[MOD_INT]]
  // CHECK: %[[X10:.*]] = mod_arith.sub %[[X3]], %[[X9]] : [[MOD_INT]]
  // CHECK: %[[X11:.*]] = mod_arith.mul %[[X6]], %[[X10]] : [[MOD_INT]]
  // CHECK: %[[X12:.*]] = mod_arith.mul %[[X2]], %[[COORDS]]#1 : [[MOD_INT]]
  // CHECK: %[[X13:.*]] = mod_arith.sub %[[X11]], %[[X12]] : [[MOD_INT]]
  // CHECK: %[[RESULT:.*]] = elliptic_curve.point %[[X9]], %[[X13]], %[[X1]], %[[X2]] : ([[MOD_INT]], [[MOD_INT]], [[MOD_INT]], [[MOD_INT]]) -> [[XYZZ]]
  %double = elliptic_curve.double %point : !affine -> !xyzz
  // CHECK: return %[[RESULT]] : [[XYZZ]]
  return %double : !xyzz
}

// CHECK-LABEL: @test_xyzz_to_xyzz_double
// CHECK-SAME: (%[[ARG0:.*]]: [[XYZZ:.*]]) -> [[XYZZ:.*]] {
func.func @test_xyzz_to_xyzz_double(%point: !xyzz) -> !xyzz {
  // CHECK: %[[COORDS:.*]]:4 = elliptic_curve.extract %[[ARG0]] : [[XYZZ]] -> [[MOD_INT:.*]], [[MOD_INT]], [[MOD_INT]], [[MOD_INT]]
  // CHECK: %[[X0:.*]] = mod_arith.double %[[COORDS]]#1 : [[MOD_INT]]
  // CHECK: %[[X1:.*]] = mod_arith.square %[[X0]] : [[MOD_INT]]
  // CHECK: %[[X2:.*]] = mod_arith.mul %[[X0]], %[[X1]] : [[MOD_INT]]
  // CHECK: %[[X3:.*]] = mod_arith.mul %[[COORDS]]#0, %[[X1]] : [[MOD_INT]]
  // CHECK: %[[X4:.*]] = mod_arith.square %[[COORDS]]#0 : [[MOD_INT]]
  // CHECK: %[[X5:.*]] = mod_arith.double %[[X4]] : [[MOD_INT]]
  // CHECK: %[[X6:.*]] = mod_arith.add %[[X5]], %[[X4]] : [[MOD_INT]]
  // CHECK: %[[X7:.*]] = mod_arith.square %[[X6]] : [[MOD_INT]]
  // CHECK: %[[X8:.*]] = mod_arith.double %[[X3]] : [[MOD_INT]]
  // CHECK: %[[X9:.*]] = mod_arith.sub %[[X7]], %[[X8]] : [[MOD_INT]]
  // CHECK: %[[X10:.*]] = mod_arith.sub %[[X3]], %[[X9]] : [[MOD_INT]]
  // CHECK: %[[X11:.*]] = mod_arith.mul %[[X6]], %[[X10]] : [[MOD_INT]]
  // CHECK: %[[X12:.*]] = mod_arith.mul %[[X2]], %[[COORDS]]#1 : [[MOD_INT]]
  // CHECK: %[[X13:.*]] = mod_arith.sub %[[X11]], %[[X12]] : [[MOD_INT]]
  // CHECK: %[[X14:.*]] = mod_arith.mul %[[X1]], %[[COORDS]]#2 : [[MOD_INT]]
  // CHECK: %[[X15:.*]] = mod_arith.mul %[[X2]], %[[COORDS]]#3 : [[MOD_INT]]
  // CHECK: %[[RESULT:.*]] = elliptic_curve.point %[[X9]], %[[X13]], %[[X14]], %[[X15]] : ([[MOD_INT]], [[MOD_INT]], [[MOD_INT]], [[MOD_INT]]) -> [[XYZZ]]
  %double = elliptic_curve.double %point : !xyzz -> !xyzz
  // CHECK: return %[[RESULT]] : [[XYZZ]]
  return %double : !xyzz
}
