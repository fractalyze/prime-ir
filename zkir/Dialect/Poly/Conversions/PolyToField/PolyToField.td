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
    "mlir::affine::AffineDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::zkir::field::FieldDialect",
  ];
}

#endif // ZKIR_DIALECT_POLY_CONVERSIONS_POLYTOFIELD_TD_
