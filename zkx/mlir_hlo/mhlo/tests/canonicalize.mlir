// RUN: mhlo-opt %s -canonicalize | FileCheck %s

// Scatter with zero-element indices canonicalized into mhlo.map.

// CHECK-LABEL: @scatter_zero_indices
func.func @scatter_zero_indices(
    %operand: tensor<6xi32>, %indices: tensor<0xi32>,
    %updates: tensor<6xi32>) -> tensor<6xi32> {
  // CHECK: %[[RES:.*]] = "mhlo.map"(%arg0, %arg2)
  // CHECK-NOT: mhlo.scatter
  // CHECK: return %[[RES]] : tensor<6xi32>
  %0 = "mhlo.scatter"(%operand, %indices, %updates) ({
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
    %1 = mhlo.add %arg0, %arg1 : tensor<i32>
    mhlo.return %1 : tensor<i32>
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [0],
      inserted_window_dims = [],
      scatter_dims_to_operand_dims = [],
      index_vector_dim = 0
    >,
    indices_are_sorted = false,
    unique_indices = false
  } : (tensor<6xi32>, tensor<0xi32>, tensor<6xi32>) -> tensor<6xi32>
  return %0 : tensor<6xi32>
}
