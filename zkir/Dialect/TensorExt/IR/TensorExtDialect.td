#ifndef ZKIR_DIALECT_TENSOREXT_IR_TENSOREXTDIALECT_TD_
#define ZKIR_DIALECT_TENSOREXT_IR_TENSOREXTDIALECT_TD_

include "mlir/IR/DialectBase.td"

def TensorExt_Dialect : Dialect {
  let name = "tensor_ext";
  let description = [{
    The `tensor_ext` dialect extends the tensor dialect with additional operations.
  }];
  let cppNamespace = "::mlir::zkir::tensor_ext";
  let dependentDialects = [
    "LLVM::LLVMDialect",
    "tensor::TensorDialect",
  ];
  let hasConstantMaterializer = 1;
}

#endif  // ZKIR_DIALECT_TENSOREXT_IR_TENSOREXTDIALECT_TD_
