#ifndef ZKIR_DIALECT_FIELD_CONVERSIONS_EXTFIELDTOLLVM_EXTFIELDTOLLVM_TD_
#define ZKIR_DIALECT_FIELD_CONVERSIONS_EXTFIELDTOLLVM_EXTFIELDTOLLVM_TD_

include "mlir/IR/PatternBase.td"
include "mlir/Pass/PassBase.td"

def ExtFieldToLLVM : Pass<"convert-ext-field-to-llvm"> {
  let summary = "Lower extension field to LLVM dialect.";

  let dependentDialects = [
    "LLVM::LLVMDialect",
  ];
}

#endif  // ZKIR_DIALECT_FIELD_CONVERSIONS_EXTFIELDTOLLVM_EXTFIELDTOLLVM_TD_
