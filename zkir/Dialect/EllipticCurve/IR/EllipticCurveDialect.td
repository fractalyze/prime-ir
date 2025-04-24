#ifndef ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEDIALECT_TD_
#define ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEDIALECT_TD_

include "mlir/IR/DialectBase.td"

def EllipticCurve_Dialect : Dialect {
  let name = "elliptic_curve";
  let summary = "Dialect for elliptic curve operations and types";
  let description = "Dialect for elliptic curve operations and types. Contains Affine, Jacobian, and XYZZ point types.";
  let cppNamespace = "::mlir::zkir::elliptic_curve";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif  // ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEDIALECT_TD_
