#ifndef ZKIR_DIALECT_ELLIPTICURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_ELLIPTICCURVETOFIELD_TD_
#define ZKIR_DIALECT_ELLIPTICURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_ELLIPTICCURVETOFIELD_TD_

include "mlir/IR/PatternBase.td"
include "mlir/Pass/PassBase.td"

def EllipticCurveToField : Pass<"elliptic-curve-to-field", "ModuleOp"> {
  let summary = "Lower `elliptic_curve` to `field` dialect.";

  let description = [{
    This pass lowers the `elliptic_curve` dialect to their `field` equivalents.
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
    "tensor::TensorDialect",
    "zkir::field::FieldDialect",
  ];
}

#endif  // ZKIR_DIALECT_ELLIPTICURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_ELLIPTICCURVETOFIELD_TD_
