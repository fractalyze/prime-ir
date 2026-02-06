// Copyright 2026 The PrimeIR Authors.
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

// Prime field operations benchmark using tensor-based operations
// This file contains only prime field ops which can be safely vectorized
// Iteration is handled by the C++ benchmark framework, not the MLIR code

!pf = !field.pf<2013265921 : i32, true>

// Square all elements in a tensor (1M elements)
func.func @square_buffer(%input: tensor<1048576x!pf>) -> tensor<1048576x!pf>
    attributes { llvm.emit_c_interface } {
  %result = field.square %input : tensor<1048576x!pf>
  return %result : tensor<1048576x!pf>
}

// Add two tensors element-wise
func.func @add_buffers(%a: tensor<1048576x!pf>, %b: tensor<1048576x!pf>) -> tensor<1048576x!pf>
    attributes { llvm.emit_c_interface } {
  %result = field.add %a, %b : tensor<1048576x!pf>
  return %result : tensor<1048576x!pf>
}

// Multiply-accumulate: result = a * b + c
func.func @mul_add_buffers(%a: tensor<1048576x!pf>, %b: tensor<1048576x!pf>,
                           %c: tensor<1048576x!pf>) -> tensor<1048576x!pf>
    attributes { llvm.emit_c_interface } {
  %prod = field.mul %a, %b : tensor<1048576x!pf>
  %result = field.add %prod, %c : tensor<1048576x!pf>
  return %result : tensor<1048576x!pf>
}
