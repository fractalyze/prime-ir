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

#ifndef PRIME_IR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEDIALECT_TD_
#define PRIME_IR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEDIALECT_TD_

include "mlir/IR/DialectBase.td"

def EllipticCurve_Dialect : Dialect {
  let name = "elliptic_curve";
  let summary = "Dialect for elliptic curve operations and types";
  let description = "Dialect for elliptic curve operations and types. Contains affine, jacobian, and xyzz point types.";
  let cppNamespace = "::mlir::prime_ir::elliptic_curve";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let dependentDialects = [
    "arith::ArithDialect",
    "LLVM::LLVMDialect",
    "prime_ir::field::FieldDialect",
    "prime_ir::mod_arith::ModArithDialect",
  ];
}

#endif  // PRIME_IR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEDIALECT_TD_
