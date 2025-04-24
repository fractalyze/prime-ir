#ifndef ZKIR_DIALECT_POLY_IR_POLYDIALECT_TD_
#define ZKIR_DIALECT_POLY_IR_POLYDIALECT_TD_

include "mlir/IR/DialectBase.td"

def Poly_Dialect : Dialect {
  let name = "poly";
  let description = [{
    The polynomial dialect.

    Example:
    ```
    // A constant polynomial
    !PF1 = field.pf<3:i32>
    !poly_ty = poly.polynomial<!PF1>
    %a = poly.constant <1+2x+3x^2> : !poly_ty
    ```
  }];
  let cppNamespace = "::mlir::zkir::poly";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // ZKIR_DIALECT_POLY_IR_POLYDIALECT_TD_
