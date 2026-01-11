/* Copyright 2025 The PrimeIR Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef PRIME_IR_DIALECT_POLY_IR_POLYDIALECT_TD_
#define PRIME_IR_DIALECT_POLY_IR_POLYDIALECT_TD_

include "mlir/IR/DialectBase.td"

def Poly_Dialect : Dialect {
  let name = "poly";
  let description = [{
    The polynomial dialect.

    Example:
    ```
    // A polynomial with base field F₃
    !PF1 = field.pf<3:i32>
    !poly_ty = poly.polynomial<!PF1>
    ```
  }];
  let cppNamespace = "::mlir::prime_ir::poly";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let dependentDialects = [
    "prime_ir::field::FieldDialect",
  ];
}

#endif // PRIME_IR_DIALECT_POLY_IR_POLYDIALECT_TD_
