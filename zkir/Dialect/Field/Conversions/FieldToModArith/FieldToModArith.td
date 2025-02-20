#ifndef ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FIELDTOMODARITH_TD_
#define ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FIELDTOMODARITH_TD_

include "zkir/Dialect/ModArith/IR/ModArithOps.td"
include "mlir/IR/PatternBase.td"
include "mlir/Pass/PassBase.td"

def PrimeFieldToModArith : Pass<"prime-field-to-mod-arith", "ModuleOp"> {
  let summary = "Lower `field.pf` to `mod_arith` dialect.";

  let description = [{
    This pass lowers the `field.pf` dialect to their `mod_arith` equivalents.
  }];

  let dependentDialects = [
    "mlir::zkir::mod_arith::ModArithDialect",
  ];
}

#endif  // ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FIELDTOMODARITH_TD_
