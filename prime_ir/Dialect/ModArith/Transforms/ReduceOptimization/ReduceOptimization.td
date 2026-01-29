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

#ifndef PRIME_IR_DIALECT_MODARITH_TRANSFORMS_REDUCEOPTIMIZATION_REDUCEOPTIMIZATION_TD_
#define PRIME_IR_DIALECT_MODARITH_TRANSFORMS_REDUCEOPTIMIZATION_REDUCEOPTIMIZATION_TD_

include "mlir/Pass/PassBase.td"

def ReduceOptimization : Pass<"mod-arith-reduce-opt", "ModuleOp"> {
  let summary = "Optimize `mod_arith.intr.reduce` operations based on range analysis.";
  let description = [{
    This pass optimizes chains of `mod_arith.intr.reduce` operations by
    analyzing the ranges of values and eliminating redundant reductions.

    Key optimizations:
    1. **Eliminate no-op reduces**: If the input range is already within
       [0, modulus), the reduce operation can be eliminated.

    2. **Skip intermediate reduces**: If a reduce is followed by arithmetic
       that keeps the result within a safe range for the next reduce, the
       first reduce can be skipped or merged.

    3. **Merge consecutive reduces**: Two consecutive reduces can often
       be merged into a single reduce with the combined range.

    Example:
    ```mlir
    // Before: mul + add chain with intermediate reduces
    %mul = arith.mului_extended %a, %b : i32
    %m = mod_arith.intr.reduce %mul#0, %mul#1 {output_range=[0, 2P)} : i32
    %sum = arith.addi %m, %c : i32
    %r = mod_arith.intr.reduce %sum {input_range=[0, 3P)} : i32

    // After: skip intermediate canonicalization since 3P < 4P fits safely
    %mul = arith.mului_extended %a, %b : i32
    %m_raw = mod_arith.intr.reduce %mul#0, %mul#1 {output_range=[0, 2P)} : i32
    %sum = arith.addi %m_raw, %c : i32
    %r = mod_arith.intr.reduce %sum {input_range=[0, 3P)} : i32
    ```

    This pass should run between mod-arith-to-arith and intr-reduce-to-arith.
  }];
  let dependentDialects = [
    "prime_ir::mod_arith::ModArithDialect",
    "arith::ArithDialect",
  ];
}

#endif  // PRIME_IR_DIALECT_MODARITH_TRANSFORMS_REDUCEOPTIMIZATION_REDUCEOPTIMIZATION_TD_
