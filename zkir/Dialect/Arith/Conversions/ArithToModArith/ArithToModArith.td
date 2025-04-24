#ifndef ZKIR_DIALECT_MODARITH_CONVERSIONS_ARITHTOMODARITH_ARITHTOMODARITH_TD_
#define ZKIR_DIALECT_MODARITH_CONVERSIONS_ARITHTOMODARITH_ARITHTOMODARITH_TD_

include "zkir/Dialect/ModArith/IR/ModArithOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Pass/PassBase.td"

def ArithToModArith : Pass<"arith-to-mod-arith", "ModuleOp"> {
  let summary = "Lower standard `arith` to `mod_arith`.";
  let description = [{
    This pass lowers the `arith` dialect to their `mod-arith` equivalents.

    The arith-to-mod-arith pass is required to lower a neural network TOSA
    model to a CGGI backend. This pass will transform the operations to the
    mod-arith dialect, where the find-mac pass can be used to convert
    consecutive multiply addition operations into a single operation. In a
    later pass, these large precision MAC operations (typically
    64 or 32-bit) will be lowered into small precision (8 or 4b) operations
    that can be mapped to CGGI operations.   }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::zkir::mod_arith::ModArithDialect",
  ];
}

#endif  // ZKIR_DIALECT_MODARITH_CONVERSIONS_ARITHTOMODARITH_ARITHTOMODARITH_TD_
