#ifndef ZKIR_DIALECT_FIELD_IR_FIELDDIALECT_TD_
#define ZKIR_DIALECT_FIELD_IR_FIELDDIALECT_TD_

include "mlir/IR/DialectBase.td"

def Field_Dialect : Dialect {
  let name = "field";
  let summary = "Dialect for field operations and types";
  let cppNamespace = "::mlir::zkir::field";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // ZKIR_DIALECT_FIELD_IR_FIELDDIALECT_TD_
