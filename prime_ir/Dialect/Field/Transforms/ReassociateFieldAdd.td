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

#ifndef PRIME_IR_DIALECT_FIELD_TRANSFORMS_REASSOCIATEFIELDADD_TD_
#define PRIME_IR_DIALECT_FIELD_TRANSFORMS_REASSOCIATEFIELDADD_TD_

include "mlir/Pass/PassBase.td"

def ReassociateFieldAdd : Pass<"reassociate-field-add", "func::FuncOp"> {
  let summary = "Rebalance serial field.add chains into rank-ordered trees";
  let description = [{
    Field addition is associative and commutative, so a serial reduction
    chain

      %s1 = field.add %a, %b
      %s2 = field.add %s1, %c
      %s3 = field.add %s2, %d

    can be rewritten as a balanced tree

      %t1 = field.add %a, %b
      %t2 = field.add %c, %d
      %s3 = field.add %t1, %t2

    without changing a single output value. Only the dependency shape
    changes: the serial chain has depth n-1 while the tree has depth
    ceil(log2(n)), which matters because the later `field -> mod_arith ->
    arith` lowering interleaves canonicalization ops between the adds, so
    LLVM's `-reassociate` cannot see the associativity post-lowering.

    Operands are placed in the tree by *rank* (SSA depth of the defining
    chain), LLVM-reassociate style: trees are built by repeatedly combining
    the two lowest-rank nodes, so a late-arriving operand (e.g. a value at
    the end of a long multiply chain) folds into the tree last instead of
    serializing the whole sum behind it.

    The pass only touches maximal chains of same-type `field.add` ops whose
    intermediate results have a single use; multi-use intermediates and
    mixed-type adds (extension field + base field) are treated as chain
    boundaries. The rewrite is count-neutral: n-1 adds before and after.

    This is deliberately a standalone pass rather than a canonicalization
    pattern: reassociation patterns fight constant-folding canonicalization
    patterns in a greedy driver, and chain detection wants a single cheap
    walk, not fixpoint iteration.
  }];
  let statistics = [
    Statistic<"numReassociated", "num-reassociated",
              "Number of field.add chains rebalanced">,
  ];
}

#endif // PRIME_IR_DIALECT_FIELD_TRANSFORMS_REASSOCIATEFIELDADD_TD_
