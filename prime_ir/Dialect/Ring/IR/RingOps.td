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

#ifndef PRIME_IR_DIALECT_RING_IR_RINGOPS_TD_
#define PRIME_IR_DIALECT_RING_IR_RINGOPS_TD_

include "prime_ir/Dialect/Ring/IR/RingDialect.td"
include "prime_ir/Dialect/Ring/IR/RingTypes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for ring operations (all side-effect free).
class Ring_Op<string mnemonic, list<Trait> traits = []>
    : Op<Ring_Dialect, mnemonic, traits # [Pure]>;

def Ring_FromTensorOp : Ring_Op<"from_tensor"> {
  let summary = "Build an RNS ring value from its [L, N] residue tensor.";
  let description = [{
    Wraps a limb-major `tensor<LxNxiW>` of residues as an `!ring.rq`. The tensor
    IS the lowered representation, so this lowers to identity — it exists to feed
    tensor-typed data (constants, host inputs) into ring ops.
  }];
  let arguments = (ins RankedTensorOf<[AnyInteger]>:$input);
  let results = (outs Ring_RqType:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

def Ring_ToTensorOp : Ring_Op<"to_tensor"> {
  let summary = "Extract the [L, N] residue tensor of an RNS ring value.";
  let arguments = (ins Ring_RqType:$input);
  let results = (outs RankedTensorOf<[AnyInteger]>:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

def Ring_AutomorphismOp : Ring_Op<"automorphism", [SameOperandsAndResultType]> {
  let summary = "Galois automorphism X -> X^g of R_Q = Z_Q[X]/(X^N+1).";
  let description = [{
    Applies sigma_g: X |-> X^g (g a positive odd integer, hence coprime to 2N for
    the power-of-two N), a signed coefficient permutation of the ring element:
    coefficient a_k moves to position `(g*k) mod N` with sign `-1` iff
    `(g*k) mod 2N >= N` (the X^N = -1 wrap). Applied per RNS limb. A generic ring
    endomorphism (cyclotomic Galois action).
  }];
  let arguments = (ins Ring_RqType:$input, I64Attr:$exponent);
  let results = (outs Ring_RqType:$output);
  let assemblyFormat = "$input attr-dict `:` type($output)";
  let hasVerifier = 1;
}

def Ring_NTTOp : Ring_Op<"ntt", [SameOperandsAndResultType]> {
  let summary = "Per-limb number-theoretic transform of an RNS ring element.";
  let description = [{
    Applies the (cyclic) NTT to each RNS limb independently, using an N-th root of
    unity in that limb's prime field. `inverse = true` applies the inverse NTT.
    A generic transform (number theory); lowers each limb via `field.bitcast` +
    `poly.ntt` over `field.pf<q_i>`, reusing the shared Cooley-Tukey/GS kernel.
    Requires each modulus q_i to admit an N-th root of unity (N | q_i - 1).
  }];
  let arguments = (ins Ring_RqType:$input,
                       DefaultValuedAttr<BoolAttr, "false">:$inverse);
  let results = (outs Ring_RqType:$output);
  let assemblyFormat = "$input attr-dict `:` type($output)";
}

def Ring_GadgetDecomposeOp : Ring_Op<"gadget_decompose"> {
  let summary = "Base-2^baseBits digit decomposition of each residue.";
  let description = [{
    Splits each residue into `levels` base-`2^baseBits` digits (least-significant
    first): `x_i = sum_j digit_j[i] * 2^(j*baseBits)`. Returns `levels` ring
    elements (the digit polynomials). Pure bit manipulation (shift + mask) — no
    modular reduction. A generic ring/number-theory primitive.
  }];
  let arguments = (ins Ring_RqType:$input, I64Attr:$baseBits, I64Attr:$levels);
  let results = (outs Variadic<Ring_RqType>:$digits);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($digits)";
  let hasVerifier = 1;
}

def Ring_GadgetProductOp : Ring_Op<"gadget_product"> {
  let summary = "Digit-decomposed inner product: sum_j decompose(x)_j * keys[j].";
  let description = [{
    Base-`2^baseBits`-decomposes `input`, then negacyclically multiplies each
    digit polynomial by the corresponding `keys` ring element and sums:
    `sum_j gadget_decompose(x)_j * keys[j]`. A generic ring operation (the digit
    decomposition inner product / "external product"). `keys` are ordinary ring
    operands. (An FHE key-switch is this fed a key-switching key; the "key"
    semantics live in the scheme layer, not here.) The number of keys is the
    number of decomposition levels.
  }];
  let arguments = (ins Ring_RqType:$input, Variadic<Ring_RqType>:$keys,
                       I64Attr:$baseBits);
  let results = (outs Ring_RqType:$output);
  let assemblyFormat =
      "$input `,` $keys attr-dict `:` type($input) `,` type($keys) `->` type($output)";
  let hasVerifier = 1;
}

def Ring_RescaleOp : Ring_Op<"rescale"> {
  let summary = "Exact division by the trailing (dropped) RNS modulus.";
  let description = [{
    Divides the ring element exactly by its trailing modulus `q_last` — the one
    present in the input basis but absent from the (prefix) output basis:
    `r = (x - (x mod q_last)) / q_last`. A generic RNS operation; per output limb
    `out_i = (x_i - x_last) * q_last^{-1} mod q_i`, where `x_last` is the residue
    in the dropped modulus. Lowers to per-limb `mod_arith.mul`/`sub`.
  }];
  let arguments = (ins Ring_RqType:$input);
  let results = (outs Ring_RqType:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
  let hasVerifier = 1;
}

def Ring_AddOp : Ring_Op<"add", [SameOperandsAndResultType, Commutative]> {
  let summary = "Addition in R_Q of two ring elements sharing a basis.";
  let description = [{
    The ring sum `a + b`. Both operands must share the RNS basis and degree N.
    Lowers to a per-limb `mod_arith.add<q_i>`. (A scheme that encrypts data in
    R_Q gets homomorphic addition by using this on ciphertext components; that
    semantics is not part of this dialect.)
  }];
  let arguments = (ins Ring_RqType:$lhs, Ring_RqType:$rhs);
  let results = (outs Ring_RqType:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($output)";
}

def Ring_SubOp : Ring_Op<"sub", [SameOperandsAndResultType]> {
  let summary = "Subtraction in R_Q of two ring elements sharing a basis.";
  let arguments = (ins Ring_RqType:$lhs, Ring_RqType:$rhs);
  let results = (outs Ring_RqType:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($output)";
}

def Ring_NegateOp : Ring_Op<"negate", [SameOperandsAndResultType]> {
  let summary = "Negation in R_Q (per-limb modular negate).";
  let arguments = (ins Ring_RqType:$input);
  let results = (outs Ring_RqType:$output);
  let assemblyFormat = "$input attr-dict `:` type($output)";
}

def Ring_MulOp : Ring_Op<"mul", [SameOperandsAndResultType, Commutative]> {
  let summary = "Negacyclic polynomial product in R_Q = Z_Q[X]/(X^N+1).";
  let description = [{
    The ring product `a * b` modulo X^N + 1, computed per RNS limb via the
    negacyclic NTT: twist both operands by psi^k (psi a 2N-th root of unity),
    forward NTT, pointwise multiply, inverse NTT, then untwist by psi^-k. A
    generic ring operation. Both operands must share the RNS basis and degree N.
    Requires each modulus q_i to admit a 2N-th root of unity (2N | q_i - 1).
  }];
  let arguments = (ins Ring_RqType:$lhs, Ring_RqType:$rhs);
  let results = (outs Ring_RqType:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($output)";
}

def Ring_BaseConvertOp : Ring_Op<"base_convert"> {
  let summary = "RNS fast basis conversion: reinterpret a value into a new RNS basis.";
  let description = [{
    Fast basis extension (BConv). Given `input` in RNS basis B = moduli(input),
    produce its residues in basis B' = moduli(output), representing (approximately)
    the same element of the ring. The two rings must share the degree N.

    A generic RNS primitive (CRT number representation). The "fast" form drops the
    exact CRT correction (the `-vQ` term), leaving a small bounded residual; the
    input/output bases determine whether it extends or contracts the basis. (RNS
    schemes compose it into basis-lift / reduce / rescale operations; that layering
    is not part of this dialect.)

    Lowers to a per-output-limb reduction over the input limbs:
      y_i     = x_i * (Q/q_i)^{-1} mod q_i          (in basis q_i)
      out_j   = sum_i y_i * ((Q/q_i) mod p_j) mod p_j
    reusing the single-prime field/mod_arith arithmetic per limb.

    Example:
    ```
    %y = ring.base_convert %x
       : !ring.rq<[q0, q1], N : i32> to !ring.rq<[p0, p1, p2], N : i32>
    ```
  }];
  let arguments = (ins Ring_RqType:$input);
  let results = (outs Ring_RqType:$output);
  let assemblyFormat =
      "$input attr-dict `:` type($input) `to` type($output)";
  let hasVerifier = 1;
}

#endif // PRIME_IR_DIALECT_RING_IR_RINGOPS_TD_
