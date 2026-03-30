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

#ifndef PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FIELDTOMODARITH_TD_
#define PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FIELDTOMODARITH_TD_

include "prime_ir/Dialect/ModArith/IR/ModArithOps.td"
include "mlir/IR/PatternBase.td"
include "mlir/Pass/PassBase.td"

def FieldToModArith : Pass<"field-to-mod-arith", "ModuleOp"> {
  let summary = "Lower `field` dialect to `mod_arith` dialect.";
  let description = [{
    This pass lowers the `field` dialect to their `mod_arith` equivalents.

    The `lowering-mode` option controls how operations are lowered:
    - `inline` (default): All operations are expanded inline.
    - `aot_runtime`: Extension field ops (degree >= 2, >64-bit prime) are
      lowered to func.call to pre-compiled AOT runtime functions.
    - `auto`: Uses heuristics — AOT for qualifying extension fields, inline
      for prime fields and small extension fields.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect",
    "ub::UBDialect",
    "vector::VectorDialect",
    "prime_ir::mod_arith::ModArithDialect",
  ];
  let options = [
    Option<"loweringMode", "lowering-mode", "std::string",
           /*default=*/"\"inline\"",
           "Lowering mode: 'inline' (default), 'auto', or 'aot_runtime'">,
    Option<"inlineConstantOps", "inline-constant-ops", "bool",
           /*default=*/"true",
           "When true, ops with 2+ operands where at least one is constant "
           "skip AOT and are inlined for constant folding/strength reduction.">,
  ];
}

#endif  // PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FIELDTOMODARITH_TD_
