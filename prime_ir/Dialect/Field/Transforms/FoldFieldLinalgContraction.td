/* Copyright 2026 The PrimeIR Authors.

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

#ifndef PRIME_IR_DIALECT_FIELD_TRANSFORMS_FOLDFIELDLINALGCONTRACTION_TD_
#define PRIME_IR_DIALECT_FIELD_TRANSFORMS_FOLDFIELDLINALGCONTRACTION_TD_

include "mlir/Pass/PassBase.td"

def FoldFieldLinalgContraction
    : Pass<"fold-field-linalg-contraction", "func::FuncOp"> {
  let summary = "Fold linalg.matvec/dot with constant field tensors";
  let description = [{
    This pass folds `linalg.matvec` and `linalg.dot` operations when one operand
    is a constant field tensor, inlining scalar `field.mul` and `field.add`
    operations.

    For `linalg.matvec` with a constant matrix and variable vector:
      out[i] = c₀ᵢ * x[0] + c₁ᵢ * x[1] + ...

    For `linalg.dot` with a constant vector and variable vector:
      result = c₀ * x[0] + c₁ * x[1] + ...

    The pass includes optimizations to skip zero coefficient terms and use the
    value directly for one coefficient terms.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "tensor::TensorDialect",
    "prime_ir::field::FieldDialect",
  ];
  let options = [
    Option<"maxUnrollSize", "max-unroll-size", "unsigned", "16",
           "Maximum vector size to unroll">,
  ];
}

#endif // PRIME_IR_DIALECT_FIELD_TRANSFORMS_FOLDFIELDLINALGCONTRACTION_TD_
