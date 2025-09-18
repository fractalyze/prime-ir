// RUN: cat %S/../../bn254_field_defs.mlir %S/../../bn254_ec_mont_defs.mlir %s \
// RUN:   | zkir-opt -convert-to-llvm -split-input-file \
// RUN:   | FileCheck %s -enable-var-scope

// CHECK-LABEL: @test_ext_from_coeffs
func.func @test_ext_from_coeffs(%var1: i256, %var2: i256) -> !QFm {
  // CHECK-NOT: field.ext_from_coeffs
  %ext = field.ext_from_coeffs %var1, %var2 : (i256, i256) -> !QFm
  return %ext : !QFm
}

// CHECK-LABEL: @test_ext_to_coeffs
func.func @test_ext_to_coeffs(%ext: !QFm) -> i256 {
  // CHECK-NOT: field.ext_to_coeffs
  %coeffs:2 = field.ext_to_coeffs %ext : (!QFm) -> (i256, i256)
  return %coeffs#1 : i256
}
