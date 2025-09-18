// RUN: cat %S/../../bn254_field_defs.mlir %S/../../bn254_ec_mont_defs.mlir %s \
// RUN:   | zkir-opt -convert-to-llvm -split-input-file \
// RUN:   | FileCheck %s -enable-var-scope

// CHECK-LABEL: @test_point
func.func @test_point(%var1: i256, %var2: i256, %var3: i256) {
  // CHECK-NOT: elliptic_curve.point
  %affine = elliptic_curve.point %var1, %var2 : (i256, i256) -> !affine
  %jacobian = elliptic_curve.point %var1, %var2, %var3 : (i256, i256, i256) -> !jacobian
  %xyzz = elliptic_curve.point %var1, %var2, %var3, %var1 : (i256, i256, i256, i256) -> !xyzz
  return
}

// CHECK-LABEL: @test_extract
func.func @test_extract(%affine: !affine, %jacobian: !jacobian, %xyzz: !xyzz) {
  // CHECK-NOT: elliptic_curve.extract
  %affine_coords:2 = elliptic_curve.extract %affine : !affine -> i256, i256
  %jacobian_coords:3 = elliptic_curve.extract %jacobian : !jacobian -> i256, i256, i256
  %xyzz_coords:4 = elliptic_curve.extract %xyzz : !xyzz -> i256, i256, i256, i256
  return
}

// CHECK-LABEL: @test_point_g2
func.func @test_point_g2(%var1: !llvm.struct<(i256, i256)>, %var2: !llvm.struct<(i256, i256)>, %var3: !llvm.struct<(i256, i256)>) {
  // CHECK-NOT: elliptic_curve.point
  %affine = elliptic_curve.point %var1, %var2 : (!llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>) -> !g2affine
  %jacobian = elliptic_curve.point %var1, %var2, %var3 : (!llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>) -> !g2jacobian
  %xyzz = elliptic_curve.point %var1, %var2, %var3, %var1 : (!llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>) -> !g2xyzz
  return
}

// CHECK-LABEL: @test_extract_g2
func.func @test_extract_g2(%affine: !g2affine, %jacobian: !g2jacobian, %xyzz: !g2xyzz) {
  // CHECK-NOT: elliptic_curve.extract
  %affine_coords:2 = elliptic_curve.extract %affine : !g2affine -> !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>
  %jacobian_coords:3 = elliptic_curve.extract %jacobian : !g2jacobian -> !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>
  %xyzz_coords:4 = elliptic_curve.extract %xyzz : !g2xyzz -> !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>, !llvm.struct<(i256, i256)>
  return
}
