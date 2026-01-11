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

#ifndef PRIME_IR_DIALECT_TENSOREXT_IR_TENSOREXTDIALECT_TD_
#define PRIME_IR_DIALECT_TENSOREXT_IR_TENSOREXTDIALECT_TD_

include "mlir/IR/DialectBase.td"

def TensorExt_Dialect : Dialect {
  let name = "tensor_ext";
  let description = [{
    The `tensor_ext` dialect extends the tensor dialect with additional operations.
  }];
  let cppNamespace = "::mlir::prime_ir::tensor_ext";
  let dependentDialects = [
    "arith::ArithDialect",
    "field::FieldDialect",
  ];
  let hasConstantMaterializer = 1;
}

#endif  // PRIME_IR_DIALECT_TENSOREXT_IR_TENSOREXTDIALECT_TD_
