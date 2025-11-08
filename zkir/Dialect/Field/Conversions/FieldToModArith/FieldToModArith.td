#ifndef ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FIELDTOMODARITH_TD_
#define ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FIELDTOMODARITH_TD_

include "zkir/Dialect/ModArith/IR/ModArithOps.td"
include "mlir/IR/PatternBase.td"
include "mlir/Pass/PassBase.td"

def FieldToModArith : Pass<"field-to-mod-arith", "ModuleOp"> {
  let summary = "Lower `field` dialect to `mod_arith` dialect.";
  let description = [{
    This pass lowers the `field` dialect to their `mod_arith` equivalents.
  }];
  let dependentDialects = [
    "vector::VectorDialect",
    "zkir::mod_arith::ModArithDialect",
  ];
}

#endif  // ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FIELDTOMODARITH_TD_
