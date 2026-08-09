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

#ifndef PRIME_IR_DIALECT_POLY_IR_POLYOPS_TD_
#define PRIME_IR_DIALECT_POLY_IR_POLYOPS_TD_

include "mlir/IR/BuiltinAttributes.td"
include "prime_ir/Dialect/Poly/IR/PolyAttributes.td"
include "prime_ir/Dialect/Poly/IR/PolyDialect.td"
include "prime_ir/Dialect/Poly/IR/PolyTypes.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for polynomial operations.
class Poly_Op<string mnemonic, list<Trait> traits = []> :
    Op<Poly_Dialect, mnemonic, traits # [Pure]> {
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Poly_ToTensorOp : Poly_Op<"to_tensor"> {
  let summary = "Create a tensor containing the coefficients of a polynomial.";
  let description = [{
    Creates a dense tensor value containing the coefficients of the input polynomial.
    The output tensor contains the coefficients in degree-increasing order.

    Example:
    ```
    !base_field = !field.pf<3:i32>
    !poly_ty = !poly.polynomial<!base_field, 32>
    %0 = poly.constant <1 + x**2> : !poly_ty
    %tensor = poly.to_tensor %0
    ```
  }];
  let arguments = (ins Poly_PolyType:$input);
  let results = (outs RankedTensorOf<[AnyType]>:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasCanonicalizer = 1;
}

def Poly_FromTensorOp : Poly_Op<"from_tensor"> {
  let summary = "Create a polynomial from integer coefficients stored in a tensor.";
  let description = [{
    `polynomial.from_tensor` creates a polynomial value from a tensor of coefficients.
    The input tensor must list the coefficients in degree-increasing order.

    The input one-dimensional tensor may have size at most the return polynomial's
    `maxDegree + 1`, with smaller dimension implying that all higher-degree terms have
    zero coefficient.

    Example:
    ```
    !base_field = !field.pf<3:i32>
    !poly_ty = !poly.polynomial<!base_field, 2>
    %coeffs = tensor.from_elements [2, 2, 5] : tensor<3xi32>
    %poly = poly.from_tensor %coeffs : tensor<3xi32> -> !poly_ty
    ```
  }];
  let arguments = (ins RankedTensorOf<[AnyType]>:$input);
  let results = (outs Poly_PolyType:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasCanonicalizer = 1;
}

// A rank-1 table over the transform length, whatever the batch shape of the
// transformed tensor is. Spelled as a type transform rather than by
// `SameOperandsAndResultType` so the batch axes stay off the table, and so the
// assembly format can still infer `$twiddles` from `type($output)` alone.
class Poly_TwiddlesOverMinorDim<string src, string tw> : OptionalTypesMatchWith<
  "twiddles is a rank-1 table over the transform length", src, tw,
  "::mlir::RankedTensorType::get("
      "{::llvm::cast<::mlir::ShapedType>($_self).getShape().back()}, "
      "::llvm::cast<::mlir::ShapedType>($_self).getElementType())">;

def Poly_NTTOp : Poly_Op<"ntt", [DestinationStyleOpInterface,
                                 AllTypesMatch<["source", "dest", "output"]>,
                                 Poly_TwiddlesOverMinorDim<"output", "twiddles">]> {
  let summary = "Compute the number theoretic transform.";
  let description = [{
    Computes the forward integer Number Theoretic Transform
    (NTT) on the input coefficients. It returns a tensor containing a point-value
    representation of the input polynomial.

    Given an input polynomial `F(x)` with degree `n`, and a primitive `n`-th root of
    unity `omega[n]`, the output is the list of $n$ evaluations

      `f[k] = F(omega[n]^k) ; k = {0, ..., n-1}`

    Example:
    ```
    %coeffs = poly.to_tensor %poly : !poly_ty -> tensor<4x!pf>
    %evals = poly.ntt %coeffs {root=#root}: tensor<4x!pf>
    ```

    The transform runs along the **minor** dimension; every leading dimension is
    a batch axis, so one op transforms `64x11` independent length-256
    polynomials:

    ```
    %evals = poly.ntt %coeffs into %dest {root=#root} : tensor<64x11x256x!pf>
    ```

    Batching belongs to the op rather than to the caller's loop because the
    transform is `log2(n)` barrier-separated stages. A length-256 transform
    offers 128 butterflies per stage, which does not fill a machine, and looping
    the op pays those barriers once per polynomial. Carrying the batch inside
    puts `batch * n/2` butterflies in every stage and leaves the barrier count
    at `log2(n)` for the whole batch, which is what `tileX` / `gridSize` then
    tile.

    `twiddles` stays rank-1 of length `n` however the source is batched: the
    table is indexed by position within a stage's block, which is the same for
    every polynomial in the batch, so a batched table would only duplicate it.

    With `negacyclic`, the transform is over `Z_q[X]/(X^n + 1)` instead — the ring
    lattice cryptography multiplies in, where the wrap-around comes back negated.
    `root` then carries a primitive `2n`-th root `psi` rather than an `n`-th root,
    and the cyclic core runs on `omega = psi^2`. Taking `psi` rather than deriving
    it is deliberate: a square root of `omega` is not unique, so `omega` alone does
    not determine the transform.

    It reduces to the cyclic transform with a twist on each side — forward
    pre-twists `a[j]` by `psi^j` on the natural index, inverse post-untwists by
    `psi^-j` — so the staged butterflies are shared, and `bit_reverse` is
    required because that is what puts slot `j` on coefficient `j`.

    `twiddles` cannot carry the twist and is rejected with `negacyclic`: the
    butterflies index one table by the position within a block, so a
    per-coefficient diagonal has nowhere to live.

    A length-`n` negacyclic transform exists only where the field has a primitive
    `2n`-th root, which the 2-adicity of `q - 1` decides. That is why FIPS 203's
    ML-KEM (`q = 3329`, 2-adicity 8) has no length-256 negacyclic transform and
    uses two length-128 ones, while FIPS 204's ML-DSA (`q = 8380417`, 2-adicity 13)
    has one.
  }];
  let arguments = (ins
    RankedTensorOf<[Field_PrimeFieldType]>:$source,
    RankedTensorOf<[Field_PrimeFieldType]>:$dest,
    Optional<RankedTensorOf<[Field_PrimeFieldType]>>:$twiddles,
    OptionalAttr<Field_RootOfUnityAttr>:$root,
    OptionalAttr<Builtin_IntegerAttr>:$tileX,
    OptionalAttr<Builtin_IntegerAttr>:$gridSize,
    DefaultValuedAttr<BoolAttr, "true">:$bitReverse,
    DefaultValuedAttr<BoolAttr, "false">:$inverse,
    DefaultValuedAttr<BoolAttr, "false">:$negacyclic
  );
  let results = (outs RankedTensorOf<[Field_PrimeFieldType]>:$output);
  let assemblyFormat = "$source `into` $dest (`with` $twiddles^)? attr-dict (`inverse` `=` $inverse^)? (`bit_reverse` `=` $bitReverse^)? (`negacyclic` `=` $negacyclic^)? `:` type($output)";
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getDestMutable(); }
  }];
}

#endif  // PRIME_IR_DIALECT_POLY_IR_POLYOPS_TD_
