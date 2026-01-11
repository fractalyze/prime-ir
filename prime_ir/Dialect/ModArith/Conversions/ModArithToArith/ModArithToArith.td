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

#ifndef PRIME_IR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_TD_
#define PRIME_IR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_TD_

include "prime_ir/Dialect/ModArith/IR/ModArithOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/IR/PatternBase.td"
include "mlir/Pass/PassBase.td"

def ModArithToArith : Pass<"mod-arith-to-arith", "ModuleOp"> {
  let summary = "Lower `mod_arith` to standard `arith`.";
  let description = [{
    This pass lowers the `mod_arith` dialect to their `arith` equivalents.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "math::MathDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect",
    "prime_ir::mod_arith::ModArithDialect",
  ];
}

#endif  // PRIME_IR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_TD_
