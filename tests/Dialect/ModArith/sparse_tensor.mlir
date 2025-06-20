// RUN: zkir-opt -sparsification-and-bufferization -split-input-file %s | FileCheck %s -enable-var-scope

!Zp = !mod_arith.int<65537 : i32>

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
}>

// CHECK-LABEL: @sparse_tensor_assemble
func.func @sparse_tensor_assemble(%A.row_ptrs: tensor<?xindex>, %A.col_indices: tensor<?xindex>, %A.values: tensor<?x!Zp>) -> tensor<4x3x!Zp, #CSR> {
  %0 = sparse_tensor.assemble (%A.row_ptrs, %A.col_indices), %A.values
    : (tensor<?xindex>, tensor<?xindex>), tensor<?x!Zp> to tensor<4x3x!Zp, #CSR>
  return %0 : tensor<4x3x!Zp, #CSR>
 }
