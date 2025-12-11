/* Copyright 2025 The ZKIR Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef ZKIR_DIALECT_TENSOREXT_CONVERSIONS_TENSOREXTTOTENSOR_TENSOREXTTOTENSOR_TD_
#define ZKIR_DIALECT_TENSOREXT_CONVERSIONS_TENSOREXTTOTENSOR_TENSOREXTTOTENSOR_TD_

include "zkir/Dialect/TensorExt/IR/TensorExtOps.td"
include "mlir/Dialect/Tensor/IR/TensorOps.td"
include "mlir/IR/PatternBase.td"
include "mlir/Pass/PassBase.td"

def TensorExtToTensor : Pass<"tensor-ext-to-tensor", "ModuleOp"> {
  let summary = "Lower `tensor_ext` to standard `tensor`.";
  let description = [{
    This pass lowers the `tensor_ext` dialect to their `tensor` equivalents.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
    "LLVM::LLVMDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

#endif  // ZKIR_DIALECT_TENSOREXT_CONVERSIONS_TENSOREXTTOTENSOR_TENSOREXTTOTENSOR_TD_
