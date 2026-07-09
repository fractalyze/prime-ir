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

#ifndef PRIME_IR_DIALECT_RING_IR_RINGDIALECT_TD_
#define PRIME_IR_DIALECT_RING_IR_RINGDIALECT_TD_

include "mlir/IR/DialectBase.td"

def Ring_Dialect : Dialect {
  let name = "ring";
  let description = [{
    Quotient polynomial rings R_Q = Z_Q[X]/(X^N+1) whose coefficient modulus Q is
    an RNS product of coprime word primes.

    The `ring.rq` type's distinguishing feature over the single-prime `poly`
    dialect is that the coefficient modulus Q is a *descriptor*: a list of coprime
    NTT-word primes (a single prime when the list has length 1; an RNS product
    when longer), with room for a 2^k coefficient kind later. Named for the
    algebraic structure (a ring), not the RNS storage technique that realizes it,
    and independent of any application. A rank-k module over the ring is just a
    `tensor<k x !ring.rq<...>>` — no dedicated module type is needed.

    (Lattice/RNS schemes — e.g. RNS-FHE — build on these rings, but that
    semantics lives above this dialect.)

    Example:
    ```
    !R = !ring.rq<[1152921504606830593, 1152921504598720513], 4096 : i32>
    ```
  }];
  let cppNamespace = "::mlir::prime_ir::ring";
  let useDefaultTypePrinterParser = 1;
}

#endif // PRIME_IR_DIALECT_RING_IR_RINGDIALECT_TD_
