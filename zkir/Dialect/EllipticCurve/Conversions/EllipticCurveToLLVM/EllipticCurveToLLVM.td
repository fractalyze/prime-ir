#ifndef ZKIR_DIALECT_ELLIPTICURVE_CONVERSIONS_ELLIPTICCURVETOLLVM_ELLIPTICCURVETOLLVM_TD_
#define ZKIR_DIALECT_ELLIPTICURVE_CONVERSIONS_ELLIPTICCURVETOLLVM_ELLIPTICCURVETOLLVM_TD_

include "mlir/IR/PatternBase.td"
include "mlir/Pass/PassBase.td"

def EllipticCurveToLLVM : Pass<"convert-ec-to-llvm"> {
  let summary = "Lower Elliptic Curve dialect to LLVM dialect.";

  let dependentDialects = [
    "LLVM::LLVMDialect",
  ];
}

#endif  // ZKIR_DIALECT_ELLIPTICURVE_CONVERSIONS_ELLIPTICCURVETOLLVM_ELLIPTICCURVETOLLVM_TD_
