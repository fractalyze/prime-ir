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

#ifndef PRIME_IR_DIALECT_ELLIPTICURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_ELLIPTICCURVETOFIELD_TD_
#define PRIME_IR_DIALECT_ELLIPTICURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_ELLIPTICCURVETOFIELD_TD_

include "mlir/IR/PatternBase.td"
include "mlir/Pass/PassBase.td"

def EllipticCurveToField : Pass<"elliptic-curve-to-field", "ModuleOp"> {
  let summary = "Lower `elliptic_curve` to `field` dialect.";

  let description = [{
    This pass lowers the `elliptic_curve` dialect to their `field` equivalents.

    The `lowering-mode` option controls how operations are lowered:
    - `inline` (default): All operations are expanded inline.
    - `intrinsic`: Complex operations (scalar_mul) are lowered to function calls
      instead of inline expansion.
    - `auto`: Uses heuristics to decide - scalar_mul uses intrinsic mode,
      while simpler operations (add, double) are inlined for vectorization.

    The intrinsic mode is useful for:
    - Reducing code size for scalar_mul (double-and-add generates many ops)
    - Avoiding NVPTX code generation issues with large aggregate returns
      (See https://github.com/fractalyze/zkx/issues/84)
    - Maintaining control flow for debugging
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
    "func::FuncDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect",
    "prime_ir::field::FieldDialect",
  ];
  let options = [
    Option<"loweringMode", "lowering-mode", "std::string",
           /*default=*/"\"inline\"",
           "Lowering mode: 'inline' (default), 'intrinsic', or 'auto'">,
  ];
}

#endif  // PRIME_IR_DIALECT_ELLIPTICURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_ELLIPTICCURVETOFIELD_TD_
