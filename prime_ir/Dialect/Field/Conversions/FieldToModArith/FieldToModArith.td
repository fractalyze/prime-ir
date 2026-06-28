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
    "linalg::LinalgDialect",
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
    Option<"useElementwiseInverse", "use-elementwise-inverse", "bool",
           /*default=*/"false",
           "When true, tensor field.inverse is lowered as per-element "
           "scalar inverse instead of Montgomery batch inverse. "
           "Preferred for GPU targets where parallel per-element inverse "
           "is faster than sequential batch inverse.">,
    Option<"inverseAlgorithm", "inverse-algorithm", "std::string",
           /*default=*/"\"bernstein-yang\"",
           "Algorithm for prime-field scalar inverse:\n"
           "  'bernstein-yang' (default) - safegcd loop (data-dependent "
           "control flow), the device-neutral choice;\n"
           "  'fermat'         - x^(p-2) as a fully unrolled, branch-free "
           "square-and-multiply chain (divergence-free, but its cost grows "
           "with field width);\n"
           "  'auto'           - Fermat for small prime fields (modulus "
           "<= 64 bits), Bernstein-Yang otherwise. The crossover is "
           "GPU-specific (Fermat avoids warp divergence for small fields but "
           "overtakes Bernstein-Yang's cost for wide ones), so a target that "
           "benefits (e.g. the GPU backend) opts in rather than this being the "
           "default. Extension-field inverse always uses Bernstein-Yang.">,
  ];
}

#endif  // PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FIELDTOMODARITH_TD_
