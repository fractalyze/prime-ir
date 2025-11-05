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
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect",
    "zkir::tensor_ext::TensorExtDialect",
  ];
}

#endif  // ZKIR_DIALECT_TENSOREXT_CONVERSIONS_TENSOREXTTOTENSOR_TENSOREXTTOTENSOR_TD_
