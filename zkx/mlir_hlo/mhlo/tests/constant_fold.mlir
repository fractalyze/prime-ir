// RUN: mhlo-opt %s -canonicalize -split-input-file | FileCheck %s

!pf_babybear_mont = !field.pf<2013265921 : i32, true>

// CHECK-LABEL: @broadcast_fold_field
// CHECK-SAME: () -> [[T:.*]] {
func.func @broadcast_fold_field() -> tensor<4x2x!pf_babybear_mont> {
  // CHECK: %[[RESULT:.*]] = mhlo.constant() <{value = dense<536870908> : tensor<4x2xi32>}> : () -> [[T]]
  // CHECK-NOT: mhlo.broadcast
  // CHECK: return %[[RESULT]] : [[T]]
  %input = mhlo.constant dense<2> : tensor<2x!pf_babybear_mont>
  %broadcasted = "mhlo.broadcast"(%input) {
    broadcast_sizes = dense<[4]> : tensor<1xi64>
  } : (tensor<2x!pf_babybear_mont>) -> tensor<4x2x!pf_babybear_mont>
  return %broadcasted : tensor<4x2x!pf_babybear_mont>
}

// CHECK-LABEL: @broadcast_in_dim_fold_field
// CHECK-SAME: () -> [[T:.*]] {
func.func @broadcast_in_dim_fold_field() -> tensor<4x2x!pf_babybear_mont> {
  // CHECK: %[[RESULT:.*]] = mhlo.constant() <{value = dense<536870908> : tensor<4x2xi32>}> : () -> [[T]]
  // CHECK: return %[[RESULT]] : [[T]]
  // CHECK-NOT: mhlo.broadcast_in_dim
  %input = mhlo.constant dense<2> : tensor<2x!pf_babybear_mont>
  %broadcasted = "mhlo.broadcast_in_dim"(%input) {
    broadcast_dimensions = dense<[1]> : tensor<1xi64>
  } : (tensor<2x!pf_babybear_mont>) -> tensor<4x2x!pf_babybear_mont>
  return %broadcasted : tensor<4x2x!pf_babybear_mont>
}

// CHECK-LABEL: @concatenate_fold_field
// CHECK-SAME: () -> [[T:.*]] {
func.func @concatenate_fold_field() -> tensor<4x!pf_babybear_mont> {
  // CHECK: %[[RESULT:.*]] = mhlo.constant() <{value = dense<[268435454, 536870908, 805306362, 1073741816]> : tensor<4xi32>}> : () -> [[T]]
  // CHECK-NOT: mhlo.concatenate
  // CHECK: return %[[RESULT]] : [[T]]
  %input1 = mhlo.constant dense<[1, 2]> : tensor<2x!pf_babybear_mont>
  %input2 = mhlo.constant dense<[3, 4]> : tensor<2x!pf_babybear_mont>
  %result = "mhlo.concatenate"(%input1, %input2) {
    dimension = 0 : i64
  } : (tensor<2x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
  return %result : tensor<4x!pf_babybear_mont>
}

// CHECK-LABEL: @iota_fold_field
// CHECK-SAME: () -> [[T:.*]] {
func.func @iota_fold_field() -> tensor<1x!pf_babybear_mont> {
  // CHECK: %[[RESULT:.*]] = mhlo.constant() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1x!pf_babybear_mont>
  // CHECK-NOT: mhlo.iota
  // CHECK: return %[[RESULT]] : [[T]]
  %result = "mhlo.iota"() {
    iota_dimension = 0 : i64
  } : () -> tensor<1x!pf_babybear_mont>
  return %result : tensor<1x!pf_babybear_mont>
}

// CHECK-LABEL: @pad_fold_field_zero
// CHECK-SAME: () -> [[T:.*]] {
func.func @pad_fold_field_zero() -> tensor<8x!pf_babybear_mont> {
  // CHECK: %[[INPUT:.*]] = mhlo.constant() <{value = dense<[0, 0, 268435454, 536870908, 805306362, 1073741816, 0, 0]> : tensor<8xi32>}> : () -> [[T]]
  // CHECK-NOT: mhlo.pad
  // CHECK: return %[[INPUT]] : [[T]]
  %input = mhlo.constant dense<[1, 2, 3, 4]> : tensor<4x!pf_babybear_mont>
  %padding = mhlo.constant dense<0> : tensor<!pf_babybear_mont>
  %result = "mhlo.pad"(%input, %padding) {
    edge_padding_low = dense<2> : tensor<1xi64>,
    edge_padding_high = dense<2> : tensor<1xi64>
  } : (tensor<4x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
  return %result : tensor<8x!pf_babybear_mont>
}

// CHECK-LABEL: @reshape_fold_field
// CHECK-SAME: () -> [[T:.*]] {
func.func @reshape_fold_field() -> tensor<2x2x!pf_babybear_mont> {
  // CHECK: %[[RESULT:.*]] = mhlo.constant() <{value = dense<{{\[}}[268435454, 536870908], [805306362, 1073741816]]> : tensor<2x2xi32>}> : () -> [[T]]
  // CHECK-NOT: mhlo.reshape
  // CHECK: return %[[RESULT]] : [[T]]
  %input = mhlo.constant dense<[1, 2, 3, 4]> : tensor<4x!pf_babybear_mont>
  %result = "mhlo.reshape"(%input) : (tensor<4x!pf_babybear_mont>) -> tensor<2x2x!pf_babybear_mont>
  return %result : tensor<2x2x!pf_babybear_mont>
}

// CHECK-LABEL: @reverse_fold_field_static_size
// CHECK-SAME: () -> [[T:.*]] {
func.func @reverse_fold_field_static_size() -> tensor<4x!pf_babybear_mont> {
  // CHECK: %[[RESULT:.*]] = mhlo.constant() <{value = dense<[1073741816, 805306362, 536870908, 268435454]> : tensor<4xi32>}> : () -> [[T]]
  // CHECK-NOT: mhlo.reverse
  // CHECK: return %[[RESULT]] : [[T]]
  %input = mhlo.constant dense<[1, 2, 3, 4]> : tensor<4x!pf_babybear_mont>
  %result = "mhlo.reverse"(%input) {
    dimensions = dense<[0]> : tensor<1xi64>
  } : (tensor<4x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
  return %result : tensor<4x!pf_babybear_mont>
}

// CHECK-LABEL: @scatter_full_replace_field
// CHECK-SAME: () -> [[T:.*]] {
func.func @scatter_full_replace_field() -> tensor<3x!pf_babybear_mont> {
  // CHECK: %[[UPDATE:.*]] = mhlo.constant() <{value = dense<[1879048178, 134217711, 402653165]> : tensor<3xi32>}> : () -> [[T]]
  // CHECK-NOT: mhlo.scatter
  // CHECK: return %[[UPDATE]] : [[T]]
  %base = mhlo.constant dense<[1, 2, 3]> : tensor<3x!pf_babybear_mont>
  %index = mhlo.constant dense<0> : tensor<1xi32>
  %update = mhlo.constant dense<[7, 8, 9]> : tensor<3x!pf_babybear_mont>
  %0 = "mhlo.scatter"(%base, %index, %update) ({
  ^bb0(%arg0: tensor<!pf_babybear_mont>, %arg1: tensor<!pf_babybear_mont>):
    mhlo.return %arg1 : tensor<!pf_babybear_mont>
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [0],
      inserted_window_dims = [],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 0
    >,
    indices_are_sorted = false,
    unique_indices = false
  } : (tensor<3x!pf_babybear_mont>, tensor<1xi32>, tensor<3x!pf_babybear_mont>) -> tensor<3x!pf_babybear_mont>
  return %0 : tensor<3x!pf_babybear_mont>
}

// CHECK-LABEL: @scatter_fold_field
// CHECK-SAME: () -> [[T:.*]] {
func.func @scatter_fold_field() -> tensor<4x!pf_babybear_mont> {
  // CHECK: %[[RESULT:.*]] = mhlo.constant() <{value = dense<[268435454, 536870908, 671088619, 1073741816]> : tensor<4xi32>}> : () -> [[T]]
  // CHECK-NOT: mhlo.scatter
  // CHECK: return %[[RESULT]] : [[T]]
  %base = mhlo.constant dense<[1, 2, 3, 4]> : tensor<4x!pf_babybear_mont>
  %index = mhlo.constant dense<2> : tensor<1xi32>
  %update = mhlo.constant dense<10> : tensor<1x!pf_babybear_mont>
  %0 = "mhlo.scatter"(%base, %index, %update) ({
  ^bb0(%arg0: tensor<!pf_babybear_mont>, %arg1: tensor<!pf_babybear_mont>):
    mhlo.return %arg1 : tensor<!pf_babybear_mont>
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >,
    indices_are_sorted = false,
    unique_indices = false
  } : (tensor<4x!pf_babybear_mont>, tensor<1xi32>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
  return %0 : tensor<4x!pf_babybear_mont>
}

// CHECK-LABEL: @slice_fold_field
// CHECK-SAME: () -> [[T:.*]] {
func.func @slice_fold_field() -> tensor<3x!pf_babybear_mont> {
  // CHECK: %[[RESULT:.*]] = mhlo.constant() <{value = dense<[536870908, 805306362, 1073741816]> : tensor<3xi32>}> : () -> [[T]]
  // CHECK-NOT: mhlo.slice
  // CHECK: return %[[RESULT]] : [[T]]
  %input = mhlo.constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6x!pf_babybear_mont>
  %result = "mhlo.slice"(%input) {
    start_indices = dense<1> : tensor<1xi64>,
    limit_indices = dense<4> : tensor<1xi64>,
    strides = dense<1> : tensor<1xi64>
  } : (tensor<6x!pf_babybear_mont>) -> tensor<3x!pf_babybear_mont>
  return %result : tensor<3x!pf_babybear_mont>
}

// CHECK-LABEL: @transpose_fold_field_splat
// CHECK-SAME: () -> [[T:.*]] {
func.func @transpose_fold_field_splat() -> tensor<2x3x!pf_babybear_mont> {
  // CHECK: %[[RESULT:.*]] = mhlo.constant() <{value = dense<1342177270> : tensor<2x3xi32>}> : () -> [[T]]
  // CHECK-NOT: mhlo.transpose
  // CHECK: return %[[RESULT]] : [[T]]
  %input = mhlo.constant dense<5> : tensor<3x2x!pf_babybear_mont>
  %result = "mhlo.transpose"(%input) {
    permutation = dense<[1, 0]> : tensor<2xi64>
  } : (tensor<3x2x!pf_babybear_mont>) -> tensor<2x3x!pf_babybear_mont>
  return %result : tensor<2x3x!pf_babybear_mont>
}
