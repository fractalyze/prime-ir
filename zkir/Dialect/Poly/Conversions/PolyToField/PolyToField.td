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

#ifndef ZKIR_DIALECT_POLY_CONVERSIONS_POLYTOFIELD_TD_
#define ZKIR_DIALECT_POLY_CONVERSIONS_POLYTOFIELD_TD_

include "zkir/Dialect/Field/IR/FieldOps.td"
include "mlir/IR/PatternBase.td"
include "mlir/Pass/PassBase.td"

def PolyToField : Pass<"poly-to-field", "ModuleOp"> {
  let summary = "Lower `poly` to `field`";
  let description = [{
    This pass lowers the `poly` dialect to the `field` dialect.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect",
    "zkir::field::FieldDialect",
    "zkir::tensor_ext::TensorExtDialect",
  ];
}

#endif // ZKIR_DIALECT_POLY_CONVERSIONS_POLYTOFIELD_TD_
