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

    Raw words carry no basis, so naming one here asserts it, and for words that
    come from outside the module that assertion is the caller's to make. Words
    that came back out of `ring.to_tensor` are not: they still carry the basis
    of the ring they left, so naming a different one is refused. The same pair
    of ops round-trips to nothing when the basis does agree.
  }];
  let arguments = (ins RankedTensorOf<[AnyInteger]>:$input);
  let results = (outs Ring_RqType:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def Ring_ToTensorOp : Ring_Op<"to_tensor"> {
  let summary = "Extract the [L, N] residue tensor of an RNS ring value.";
  let arguments = (ins Ring_RqType:$input);
  let results = (outs RankedTensorOf<[AnyInteger]>:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def Ring_FromLimbsOp : Ring_Op<"from_limbs"> {
  let summary = "Build an RNS ring value from one field tensor per modulus.";
  let description = [{
    Limb `i` is a `tensor<Nx!field.pf<q_i>>` holding the residues mod `q_i`, so
    the modulus arrives as part of the limb's type and the verifier checks it
    against the ring rather than against a restated attribute. This is the
    boundary where a set of independently typed residue vectors becomes one
    value that CRT relates.

    ```mlir
    %r = ring.from_limbs %l0, %l1
        : tensor<8x!field.pf<12289:i64>>, tensor<8x!field.pf<40961:i64>>
        to !ring.rq<[12289, 40961], 8 : i32, eval>
    ```
  }];
  let arguments = (ins Variadic<RankedTensorOf<[AnyType]>>:$limbs);
  let results = (outs Ring_RqType:$output);
  let assemblyFormat =
      "$limbs attr-dict `:` type($limbs) `to` type($output)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def Ring_ToLimbsOp : Ring_Op<"to_limbs"> {
  let summary = "Split an RNS ring value into one field tensor per modulus.";
  let description = [{
    The inverse of `ring.from_limbs`: residue row `i` comes back typed
    `!field.pf<q_i>`, so downstream field passes see an ordinary field tensor.
  }];
  let arguments = (ins Ring_RqType:$input);
  let results = (outs Variadic<RankedTensorOf<[AnyType]>>:$limbs);
  let assemblyFormat =
      "$input attr-dict `:` type($input) `to` type($limbs)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def Ring_AutomorphismOp : Ring_Op<"automorphism", [SameOperandsAndResultType]> {
  let summary = "Galois automorphism X -> X^g of R_Q = Z_Q[X]/(X^N+1).";
  let description = [{
    Applies sigma_g: X |-> X^g (g a positive odd integer, hence coprime to 2N for
    the power-of-two N), a signed coefficient permutation of the ring element:
    coefficient a_k moves to position `(g*k) mod N` with sign `-1` iff
    `(g*k) mod 2N >= N` (the X^N = -1 wrap). Applied per RNS limb. A generic ring
    endomorphism (cyclotomic Galois action).

    Defined on either basis. In the evaluation basis it is a pure permutation of
    the slots: slot `j` evaluates at `psi^(2j+1)`, and sigma_g carries that to
    `psi^(g(2j+1))`, an exponent that reduces exactly mod 2N — so no signs
    appear on that side. RNS libraries rotate in the evaluation basis for this
    reason, rather than paying transforms around the coefficient form.
  }];
  let arguments = (ins Ring_RqType:$input, I64Attr:$exponent);
  let results = (outs Ring_RqType:$output);
  let assemblyFormat = "$input attr-dict `:` type($output)";
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
  let summary = "The ring product: one pointwise mod_arith.mul per RNS limb.";
  let description = [{
    The ring product `a * b`, in the evaluation basis where CRT has already
    diagonalised the ring — so it is a single pointwise `mod_arith.mul` per limb.

    Coefficient-basis operands are rejected. The product there is a negacyclic
    convolution, which needs the transform, and the transform is not this
    dialect's to own: callers reach the evaluation basis first and multiply here.
    That also matches production RNS libraries, which keep ciphertexts in the
    evaluation basis rather than paying transforms per product.
  }];
  let arguments = (ins Ring_RqType:$lhs, Ring_RqType:$rhs);
  let results = (outs Ring_RqType:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($output)";
  let hasVerifier = 1;
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
