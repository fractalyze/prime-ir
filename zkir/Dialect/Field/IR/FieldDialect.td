/* Copyright 2025 The ZKIR Authors.

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

#ifndef ZKIR_DIALECT_FIELD_IR_FIELDDIALECT_TD_
#define ZKIR_DIALECT_FIELD_IR_FIELDDIALECT_TD_

include "mlir/IR/DialectBase.td"

def Field_Dialect : Dialect {
  let name = "field";
  let summary = "Dialect for field operations and types";
  let cppNamespace = "::mlir::zkir::field";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let dependentDialects = [
    "arith::ArithDialect",
    "zkir::mod_arith::ModArithDialect",
    "zkir::tensor_ext::TensorExtDialect",
  ];
}

#endif // ZKIR_DIALECT_FIELD_IR_FIELDDIALECT_TD_
