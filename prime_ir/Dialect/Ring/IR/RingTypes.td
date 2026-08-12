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
include "mlir/IR/EnumAttr.td"
include "prime_ir/Dialect/Ring/IR/RingDialect.td"
include "prime_ir/Dialect/Ring/IR/RingEnums.td"

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

    `storageType` is the integer word one residue occupies. NTT-word primes
    usually fit 32 bits, and a limb crossing the boundary is already typed by
    its own field dtype, so pinning the ring to 64-bit words would cost a
    widening copy at the bridge and double the traffic on every limb. It is
    omitted from the assembly when it is `i64`.

    A rank-k module over R_Q is `tensor<k x !ring.rq<...>>` (componentwise), so
    element-times-element is left to the scalar action, not a ring×ring op here.

    `domain` records which CRT basis the value is written in — the coefficient
    basis or the fully split evaluation basis. It defaults to `coeff` and is
    omitted from the assembly in that case. The two bases share the same `[L, N]`
    storage but not the same multiplication, so the basis is not recoverable from
    the data and has to live in the type; see `Ring_Domain`. Nothing here changes
    it: the transform between the two is the NTT, which lives in XLA.

    Example:
    ```
    // N = 4096, two ~60-bit RNS limbs
    !R = !ring.rq<[1152921504606830593, 1152921504598720513], 4096 : i32>
    // the same ring in the evaluation basis
    !Rhat = !ring.rq<[1152921504606830593, 1152921504598720513], 4096 : i32, eval>
    // two 30-bit limbs held in 32-bit words
    !Rsmall = !ring.rq<[1073479681, 1071513601], 4096 : i32, i32, eval>
    ```
  }];
  let parameters = (ins "::mlir::DenseI64ArrayAttr":$moduli,
                        "::mlir::IntegerAttr":$ringDegree,
                        "::mlir::IntegerType":$storageType,
                        EnumParameter<Ring_Domain>:$domain);
  let builders = [
    // Coefficient basis is the default spelling; an eval-basis ring is named
    // explicitly, at the boundary with whoever ran the transform.
    TypeBuilder<(ins "::mlir::DenseI64ArrayAttr":$moduli,
                     "::mlir::IntegerAttr":$ringDegree,
                     "::mlir::IntegerType":$storageType), [{
      return $_get($_ctxt, moduli, ringDegree, storageType, Domain::Coeff);
    }]>,
  ];
  let extraClassDeclaration = [{
    RqType withDomain(Domain d) const {
      return RqType::get(getContext(), getModuli(), getRingDegree(),
                         getStorageType(), d);
    }
    unsigned getStorageWidth() const { return getStorageType().getWidth(); }
    bool isCoeff() const { return getDomain() == Domain::Coeff; }
    bool isEval() const { return getDomain() == Domain::Eval; }
  }];
  // Custom `<[q0, q1, ...], N>` format: DenseI64ArrayAttr's generic field parser
  // wants `[...]` but prints `array<i64:...>`, so it can't round-trip on its
  // own, and the trailing storage/domain fields are both optional.
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

#endif // PRIME_IR_DIALECT_RING_IR_RINGTYPES_TD_
