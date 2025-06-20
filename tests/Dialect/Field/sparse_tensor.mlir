// RUN: zkir-opt -sparsification-and-bufferization -split-input-file %s | FileCheck %s -enable-var-scope

!PF = !field.pf<7:i32, true>

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
}>

// CHECK-LABEL: @sparse_tensor_assemble
func.func @sparse_tensor_assemble(%A.row_ptrs: tensor<?xindex>, %A.col_indices: tensor<?xindex>, %A.values: tensor<?x!PF>) -> tensor<4x3x!PF, #CSR> {
  %0 = sparse_tensor.assemble (%A.row_ptrs, %A.col_indices), %A.values
    : (tensor<?xindex>, tensor<?xindex>), tensor<?x!PF> to tensor<4x3x!PF, #CSR>
  return %0 : tensor<4x3x!PF, #CSR>
 }
