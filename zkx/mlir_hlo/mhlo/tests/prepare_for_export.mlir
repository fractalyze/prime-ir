// RUN: mhlo-opt %s -zkx-prepare-for-export -split-input-file | FileCheck %s

!pf_babybear_mont = !field.pf<2013265921 : i32, true>

// CHECK-LABEL: @splat_field_constant
func.func @splat_field_constant() -> tensor<32x!pf_babybear_mont> {
  // The splat constant has >=32 elements, so it should be decomposed into
  // a scalar constant + broadcast_in_dim. The scalar constant must preserve
  // the field element type.
  // CHECK: %[[CST:.*]] = mhlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
  // CHECK: %[[BCAST:.*]] = "mhlo.broadcast_in_dim"(%[[CST]])
  // CHECK-SAME: (tensor<!pf_babybear_mont>) -> tensor<32x!pf_babybear_mont>
  // CHECK: return %[[BCAST]]
  %0 = mhlo.constant() <{value = dense<0> : tensor<32xi32>}> : () -> tensor<32x!pf_babybear_mont>
  return %0 : tensor<32x!pf_babybear_mont>
}

// -----

!pf_babybear_mont = !field.pf<2013265921 : i32, true>

// Splat constants with <32 elements should not be decomposed.
// CHECK-LABEL: @small_splat_field_constant
func.func @small_splat_field_constant() -> tensor<16x!pf_babybear_mont> {
  // CHECK: mhlo.constant() <{value = dense<0> : tensor<16xi32>}> : () -> tensor<16x!pf_babybear_mont>
  // CHECK-NOT: broadcast_in_dim
  %0 = mhlo.constant() <{value = dense<0> : tensor<16xi32>}> : () -> tensor<16x!pf_babybear_mont>
  return %0 : tensor<16x!pf_babybear_mont>
}

// -----

// Non-field splat constants should still be decomposed normally.
// CHECK-LABEL: @splat_integer_constant
func.func @splat_integer_constant() -> tensor<32xi32> {
  // CHECK: %[[CST:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %[[BCAST:.*]] = "mhlo.broadcast_in_dim"(%[[CST]])
  // CHECK-SAME: (tensor<i32>) -> tensor<32xi32>
  // CHECK: return %[[BCAST]]
  %0 = mhlo.constant dense<0> : tensor<32xi32>
  return %0 : tensor<32xi32>
}
