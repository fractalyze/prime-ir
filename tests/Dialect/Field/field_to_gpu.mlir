// Copyright 2025 The PrimeIR Authors.
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

// RUN: prime-ir-opt %s -field-to-gpu="bufferize-function-boundaries target-format=llvm nvvm-use-bare-ptr-call-conv" | FileCheck %s

!PF = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256, true>

// CHECK-LABEL: @matvec
// CHECK-SAME: (%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> !llvm.ptr
func.func @matvec(%arg0: tensor<500000x30000x!PF>, %arg1: tensor<30000x!PF>) -> tensor<500000x!PF> {
  %temp = bufferization.alloc_tensor() : tensor<500000x!PF>
  %1 = linalg.matvec ins(%arg0, %arg1: tensor<500000x30000x!PF>, tensor<30000x!PF>) outs(%temp: tensor<500000x!PF>) -> tensor<500000x!PF>
  return %1 : tensor<500000x!PF>
}
