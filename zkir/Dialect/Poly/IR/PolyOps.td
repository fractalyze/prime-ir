/* Copyright 2025 The ZKIR Authors.

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

#ifndef ZKIR_DIALECT_POLY_IR_POLYOPS_TD_
#define ZKIR_DIALECT_POLY_IR_POLYOPS_TD_

include "mlir/IR/BuiltinAttributes.td"
include "zkir/Dialect/Poly/IR/PolyAttributes.td"
include "zkir/Dialect/Poly/IR/PolyDialect.td"
include "zkir/Dialect/Poly/IR/PolyTypes.td"
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

def Poly_NTTOp : Poly_Op<"ntt", [DestinationStyleOpInterface, SameOperandsAndResultType]> {
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
  }];
  let arguments = (ins
    RankedTensorOf<[Field_PrimeFieldType]>:$source,
    RankedTensorOf<[Field_PrimeFieldType]>:$dest,
    Optional<RankedTensorOf<[Field_PrimeFieldType]>>:$twiddles,
    OptionalAttr<Field_RootOfUnityAttr>:$root,
    OptionalAttr<Builtin_IntegerAttr>:$tileX,
    OptionalAttr<Builtin_IntegerAttr>:$gridSize,
    DefaultValuedAttr<BoolAttr, "true">:$bitReverse,
    DefaultValuedAttr<BoolAttr, "false">:$inverse
  );
  let results = (outs RankedTensorOf<[Field_PrimeFieldType]>:$output);
  let assemblyFormat = "$source `into` $dest (`with` $twiddles^)? attr-dict (`inverse` `=` $inverse^)? (`bit_reverse` `=` $bitReverse^)? `:` type($output)";
  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getDestMutable(); }
  }];
}

#endif  // ZKIR_DIALECT_POLY_IR_POLYOPS_TD_
