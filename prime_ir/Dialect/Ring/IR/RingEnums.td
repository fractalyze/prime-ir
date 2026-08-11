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

#ifndef PRIME_IR_DIALECT_RING_IR_RINGENUMS_TD_
#define PRIME_IR_DIALECT_RING_IR_RINGENUMS_TD_

include "mlir/IR/EnumAttr.td"

def Ring_DomainCoeff : I32EnumAttrCase<"Coeff", 0, "coeff">;
def Ring_DomainEval  : I32EnumAttrCase<"Eval",  1, "eval">;

def Ring_Domain : I32EnumAttr<"Domain",
    "Which CRT basis an RNS ring element is written in",
    [Ring_DomainCoeff, Ring_DomainEval]> {
  let description = [{
    The basis a `!ring.rq` value is expressed in. `X^N+1` splits into distinct
    linear factors over each `Z_q_i` (given `2N | q_i - 1`), so CRT gives a ring
    isomorphism onto the product of the residue fields:

      coeff:  Z_Q[X]/(X^N+1)      multiplication is negacyclic convolution
      eval:   prod_j Z_Q          multiplication is pointwise

    Both bases occupy the same `[L, N]` residue storage, so the basis cannot be
    recovered from the data — it has to be carried in the type. Without it
    nothing stops a coefficient-domain `ring.mul` from being applied to values
    that have already been transformed, which silently convolves twice.

    No op in this dialect changes the basis: the CRT map is the NTT, and that
    lives above this dialect, in XLA. This parameter is the contract at that
    boundary — a value handed back after a transform is `eval`, and `ring.mul`
    on it is a single pointwise multiply per limb.
  }];
  let cppNamespace = "::mlir::prime_ir::ring";
}

#endif // PRIME_IR_DIALECT_RING_IR_RINGENUMS_TD_
