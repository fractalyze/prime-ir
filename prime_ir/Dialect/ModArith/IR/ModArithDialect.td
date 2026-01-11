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

#ifndef PRIME_IR_DIALECT_MODARITH_IR_MODARITHDIALECT_TD_
#define PRIME_IR_DIALECT_MODARITH_IR_MODARITHDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def ModArith_Dialect : Dialect {
  let name = "mod_arith";
  let description = [{
    The `mod_arith` dialect contains operations used for modular arithmetic.
  }];
  let cppNamespace = "::mlir::prime_ir::mod_arith";
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let dependentDialects = [
    "arith::ArithDialect",
  ];
  let hasConstantMaterializer = 1;
}

#endif  // PRIME_IR_DIALECT_MODARITH_IR_MODARITHDIALECT_TD_
