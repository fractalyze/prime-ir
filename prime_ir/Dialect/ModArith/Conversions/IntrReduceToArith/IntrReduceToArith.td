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

#ifndef PRIME_IR_DIALECT_MODARITH_CONVERSIONS_INTRREDUCETOARITH_INTRREDUCETOARITH_TD_
#define PRIME_IR_DIALECT_MODARITH_CONVERSIONS_INTRREDUCETOARITH_INTRREDUCETOARITH_TD_

include "mlir/Pass/PassBase.td"

def IntrReduceToArith : Pass<"intr-reduce-to-arith", "ModuleOp"> {
  let summary = "Lower `mod_arith.intr.reduce` to `arith` operations.";
  let description = [{
    This pass lowers the `mod_arith.intr.reduce` operation to primitive
    `arith` operations. It handles:

    - Single-operand reduction (conditional add/subtract based on range)
    - Two-operand Montgomery reduction (using Montgomery constants)

    This is the second stage of the two-stage lowering pipeline:
    1. mod-arith-to-arith: Convert mod_arith ops -> arith ops + intr.reduce
    2. intr-reduce-to-arith: Lower intr.reduce -> conditional arith ops

    The separation allows optimization passes to run between stages to
    eliminate or merge reduce operations based on range analysis.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
  ];
}

#endif  // PRIME_IR_DIALECT_MODARITH_CONVERSIONS_INTRREDUCETOARITH_INTRREDUCETOARITH_TD_
