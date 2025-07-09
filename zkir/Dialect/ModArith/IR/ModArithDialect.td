#ifndef ZKIR_DIALECT_MODARITH_IR_MODARITHDIALECT_TD_
#define ZKIR_DIALECT_MODARITH_IR_MODARITHDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def ModArith_Dialect : Dialect {
  let name = "mod_arith";
  let description = [{
    The `mod_arith` dialect contains operations used for modular arithmetic.
  }];
  let cppNamespace = "::mlir::zkir::mod_arith";
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let dependentDialects = [
    "arith::ArithDialect",
  ];
  let hasConstantMaterializer = 1;
}

#endif  // ZKIR_DIALECT_MODARITH_IR_MODARITHDIALECT_TD_
