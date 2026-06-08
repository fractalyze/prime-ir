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

    When `lazy-reduction` is enabled, the pass runs Integer Range Analysis (IRA)
    to compute value bounds and eliminates redundant modular reductions. Values
    that provably stay within the storage type range skip the final conditional
    subtraction in add/sub/REDC, and boundary ops (return, store, yield) insert
    reductions only when needed.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "math::MathDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect",
    "ub::UBDialect",
    "vector::VectorDialect",
    "prime_ir::mod_arith::ModArithDialect",
  ];
  let options = [
    Option<"lazyReduction", "lazy-reduction", "bool", "false",
      "Enable lazy reduction optimization via integer range analysis">,
    Option<"limbBits", "limb-bits", "unsigned", /*default=*/"64",
      "Montgomery-multiply limb width. 64 (default) keeps the wide-int "
      "lowering, which LLVM legalizes to a native 64-bit-limb schoolbook "
      "(mul.lo/hi + carry chains). 32 emits an explicit 32-bit-limb CIOS "
      "(every limb product is a 32x32->64 widening multiply) for multi-limb "
      "moduli. 32 is EXPERIMENTAL and off by default: it measured ~0.81x "
      "(slower) than 64 on RTX 5090 (prime-ir#337 PR3) because the GPU has "
      "a strong 64-bit multiply; it is intended for a target whose native "
      "multiply is 32-bit (e.g. a 32-bit CPU), where the wide path emulates "
      "64-bit products. No such target in the fleet today. Sign-bit-set and "
      "single-limb moduli always keep the wide-int path. Values other than "
      "32 keep the wide-int path.">,
  ];
}

#endif  // PRIME_IR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_TD_
