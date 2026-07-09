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

#ifndef PRIME_IR_DIALECT_RING_IR_RINGTYPES_TD_
#define PRIME_IR_DIALECT_RING_IR_RINGTYPES_TD_

include "mlir/IR/AttrTypeBase.td"
include "prime_ir/Dialect/Ring/IR/RingDialect.td"

class Ring_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Ring_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Ring_RqType : Ring_Type<"Rq", "rq"> {
  let summary = "A quotient polynomial ring R_Q = Z_Q[X]/(X^N + 1).";
  let description = [{
    The quotient polynomial ring R_Q = Z_Q[X]/(X^N+1) with N = `ringDegree` (a
    power of two). The coefficient modulus Q is given by `moduli`, a list of
    pairwise coprime NTT-word primes:
    - length 1  -> a single-prime ring (Z_q[X]/(X^N+1));
    - length >1 -> an RNS ring, Q = prod(moduli).
    (A 2^k coefficient kind is a planned extension — a discriminator added
    alongside `moduli`.)

    Named for the algebraic structure. RNS is a *representation* of Q, realized
    in the lowering: the limb axis (length == #moduli) is a tensor dimension (a
    [L, N] layout of residue words), NOT part of the element — so the element
    storage width is constant and dropping a modulus is a tensor slice. That is
    why the moduli live on the type, not in a per-element class.

    A rank-k module over R_Q is `tensor<k x !ring.rq<...>>` (componentwise), so
    element-times-element is left to the scalar action, not a ring×ring op here.

    Example:
    ```
    // N = 4096, two ~60-bit RNS limbs
    !R = !ring.rq<[1152921504606830593, 1152921504598720513], 4096 : i32>
    ```
  }];
  let parameters = (ins "::mlir::DenseI64ArrayAttr":$moduli,
                        "::mlir::IntegerAttr":$ringDegree);
  // Custom `<[q0, q1, ...], N>` format: DenseI64ArrayAttr's generic field parser
  // wants `[...]` but prints `array<i64:...>`, so it can't round-trip on its own.
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

#endif // PRIME_IR_DIALECT_RING_IR_RINGTYPES_TD_
