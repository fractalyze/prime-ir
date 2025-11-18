// Copyright 2025 The ZKIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

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
