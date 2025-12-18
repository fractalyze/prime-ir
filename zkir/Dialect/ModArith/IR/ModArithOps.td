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

#ifndef ZKIR_DIALECT_MODARITH_IR_MODARITHOPS_TD_
#define ZKIR_DIALECT_MODARITH_IR_MODARITHOPS_TD_

include "zkir/Dialect/ModArith/IR/ModArithAttributes.td"
include "zkir/Dialect/ModArith/IR/ModArithDialect.td"
include "zkir/Dialect/ModArith/IR/ModArithTypes.td"
include "zkir/Utils/CommonTraits.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for modular operations.
class ModArith_Op<string mnemonic, list<Trait> traits = []> :
  Op<ModArith_Dialect, mnemonic, traits #[Pure] # ElementwiseMappable.traits>;

// Base class for modular arithmetic ops. All ops have one result,
// and the result type is the same as the operand type.
class ModArith_ArithOp<string mnemonic, list<Trait> traits = []> :
    ModArith_Op<mnemonic, traits # [SameOperandsAndResultType]>;

// Base class for unary modular arithmetic.
class ModArith_UnaryOp<string mnemonic, list<Trait> traits = []> :
    ModArith_ArithOp<mnemonic, traits>,
  Arguments<(ins ModArithLike:$input)>,
  Results<(outs ModArithLike:$output)> {
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

// Base class for binary modular arithmetic.
class ModArith_BinaryOp<string mnemonic, list<Trait> traits = []> :
  ModArith_ArithOp<mnemonic, traits>,
  Arguments<(ins ModArithLike:$lhs, ModArithLike:$rhs)>,
  Results<(outs ModArithLike:$output)> {
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

def ModArith_ConstantOp : Op<ModArith_Dialect, "constant", [ConstantLike, Pure]> {
  let summary = "Define a constant value via an attribute.";
  let description = [{
    Example:

    ```
    %0 = mod_arith.constant 123 : !mod_arith.int<65537:i32>
    ```
  }];
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs ModArithLike:$output);
  let extraClassDeclaration = [{
    /// Build the constant op with `value` and `type` if possible, otherwise
    /// returns null.
    static ConstantOp materialize(OpBuilder &builder, Attribute value,
                                  Type type, Location loc);
  }];
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
}

def ModArith_BitcastOp : Op<ModArith_Dialect, "bitcast", [Pure,
 SameOperandsAndResultShape, AllSameTensorDims<["input", "output"]>,
 DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "Bitcast between modular types and to/from integer types";
  let description = [{
    Bitcasts between modular types and to/from integer types.

    The storage bitwidth of the input type is required to be the same storage
    bitwidth of the output type. Integer to integer bitcast is disallowed as it
    should be handled by `arith.bitcast`.

    Example:
    ```
    mod_arith.bitcast %c0 : i32 -> !mod_arith.int<65537 : i32>
    mod_arith.bitcast %m1 : !mod_arith.int<65537 : i64> -> i64
    mod_arith.bitcast %m2 : !mod_arith.int<65537 : i64> -> !mod_arith.int<7 : i64>
    ```
  }];
  let arguments = (ins AnyTypeOf<[ModArithLike, SignlessIntegerLike]>:$input);
  let results = (outs AnyTypeOf<[ModArithLike, SignlessIntegerLike]>:$output);
  let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";
  let hasFolder = 1;
}

def ModArith_MontReduceOp : ModArith_Op<"mont_reduce", [SameTypeOperands]> {
  let summary = "Montgomery reduction";
  let description = [{
    Performs Montgomery reduction on an integer that is twice the bitwidth of the modulus,
    represented as separate low and high parts. This is typically used for reducing the output of extended
    multiplication operations.

    Example:
    ```
    %c0 = arith.constant 65538231231312311 : i512
    %m0 = mod_arith.mont_reduce %c0 : i512 -> !mod_arith.int<65537 : i256>
    ```
  }];
  let arguments = (ins
    SignlessIntegerLike:$low,
    SignlessIntegerLike:$high
  );
  let results = (outs ModArithLike:$output);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($low) `->` type($output)";
}

def ModArith_ToMontOp : ModArith_Op<"to_mont", [TypesMatchWith<
  "input type is standard form of output type", "output", "input", "getStandardFormType($_self)">]> {
  let summary = "Convert from standard form to montgomery representation";
  let description = [{
    Transforms modular type integer to the Montgomery domain.
    In other words, it computes x * R⁻¹ mod q for the given operand `x`, where `R` is the montgomery factor.

    Example:
    ```
    %m0 = mod_arith.to_mont %c0 : !mod_arith.int<65537 : i256>
    ```
  }];
  let arguments = (ins ModArithLike:$input);
  let results = (outs ModArithLike:$output);
  let hasVerifier = 1;
  let hasFolder = 1;
  let assemblyFormat = "$input attr-dict `:` type($output)";
}

def ModArith_FromMontOp : ModArith_Op<"from_mont", [TypesMatchWith<
  "input type is montgomery form of output type", "output", "input", "getMontgomeryFormType($_self)">]> {
  let summary = "Convert from montgomery representation to standard form";
  let description = [{
    Transforms modular type integer back from the Montgomery domain.
    In other words, it computes x * R mod q for the given operand `x`, where `R` is the montgomery factor.

    Example:
    ```
    %m0 = mod_arith.from_mont %c0 : !mod_arith.int<65537 : i256>
    ```
  }];
  let arguments = (ins ModArithLike:$input);
  let results = (outs ModArithLike:$output);
  let hasVerifier = 1;
  let hasFolder = 1;
  let assemblyFormat = "$input attr-dict `:` type($output)";
}

def ModArith_CmpOp : ModArith_Op<"cmp", [SameTypeOperands, TypesMatchWith<
  "result type has i1 element type and same shape as operands",
  "lhs", "result", "getI1SameShape($_self)">]> {
  let summary = "Modular comparison operation";
  let description = [{
    **From description of `Arith_CmpIOp`**
    Its first argument is an attribute that defines which type of comparison is
    performed. The following comparisons are supported:

    -   equal (mnemonic: `"eq"`; integer value: `0`)
    -   not equal (mnemonic: `"ne"`; integer value: `1`)
    -   unsigned less than (mnemonic: `"ult"`; integer value: `6`)
    -   unsigned less than or equal (mnemonic: `"ule"`; integer value: `7`)
    -   unsigned greater than (mnemonic: `"ugt"`; integer value: `8`)
    -   unsigned greater than or equal (mnemonic: `"uge"`; integer value: `9`)

    The result is `1` if the comparison is true and `0` otherwise. For vector or
    tensor operands, the comparison is performed elementwise and the element of
    the result indicates whether the comparison is true for the operand elements
    with the same indices as those of the result.

    Example:
    ```
    %res = mod_arith.cmp slt, %a, %b : !mod_arith.int<65537 : i256>
    ```
  }];
  let arguments = (ins Arith_CmpIPredicateAttr:$predicate,
                       ModArithLike:$lhs,
                       ModArithLike:$rhs);
  let results = (outs BoolLike:$result);
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
  let hasFolder = 1;
}

def ModArith_NegateOp : ModArith_UnaryOp<"negate", [Involution]> {
  let summary = "Modular negation";
  let description = [{
    Computes the negation of a modular type integer.

    Example:
    ```
    %1 = mod_arith.negate %0 : !mod_arith.int<65537: i32>
    ```
  }];
  let arguments = (ins ModArithLike:$input);
  let results = (outs ModArithLike:$output);
  let hasFolder = 1;
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

def ModArith_SquareOp : ModArith_UnaryOp<"square"> {
  let summary = "Modular squaring operation";
  let description = [{
    Computes the square of a modular type integer.

    Example:
    ```
    %c0 = mod_arith.square %0 : !mod_arith.int<7:i4>
    ```
  }];
  let hasFolder = 1;
}

def ModArith_MontSquareOp : ModArith_UnaryOp<"mont_square"> {
  let summary = "Montgomery squaring operation";
  let description = [{
    Computes the Montgomery squaring of a modular type integer.

    Example:
    ```
    %m0 = mod_arith.mont_square %a : !mod_arith.int<65537 : i256>
    ```
  }];
  let hasFolder = 1;
}

def ModArith_InverseOp : Op<ModArith_Dialect, "inverse", [Pure, SameOperandsAndResultType, Involution]> {
  let summary = "Modular inverse operation";
  let description = [{
    Computes the modular inverse of a given modular type integer.

    Example:
    ```
    %0 = mod_arith.constant 5 : !mod_arith.int<7:i4>
    %c0 = mod_arith.inverse %0 : !mod_arith.int<7:i4>
    ```
  }];
  let arguments = (ins ModArithLike:$input);
  let results = (outs ModArithLike:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";
  let hasFolder = 1;
}

def ModArith_DoubleOp : ModArith_UnaryOp<"double"> {
  let summary = "Modular doubling operation";
  let description = [{
    Computes the double of a modular type integer.

    Example:
    ```
    %c0 = mod_arith.double %0 : !mod_arith.int<7:i4>
    ```
  }];
  let hasFolder = 1;
}

def ModArith_MontInverseOp : ModArith_UnaryOp<"mont_inverse"> {
  let summary = "Montgomery modular inverse operation";
  let description = [{
    Computes the Montgomery modular inverse of a given modular type integer.

    Example:
    ```
    %0 = mod_arith.constant 5 : !mod_arith.int<7:i4>
    %c0 = mod_arith.mont_inverse %0 : !mod_arith.int<7:i4>
    ```
  }];
  let hasFolder = 1;
}

def ModArith_AddOp : ModArith_BinaryOp<"add", [Commutative]> {
  let summary = "Modular addition operation";
  let description = [{
    Performs modular addition on the given two modular type integers.
  }];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def ModArith_SubOp : ModArith_BinaryOp<"sub"> {
  let summary = "Modular subtraction operation";
  let description = [{
    Performs modular subtraction on the given two modular type integers.
  }];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def ModArith_MulOp : ModArith_BinaryOp<"mul", [Commutative]> {
  let summary = "Modular multiplication operation";
  let description = [{
    Performs modular multiplication on the given two modular type integers.
    WARNING: This operation assumes that both operands are in the standard form.
  }];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def ModArith_MontMulOp : ModArith_BinaryOp<"mont_mul", [Commutative]> {
  let summary = "Montgomery multiplication operation";
  let description = [{
    Performs Montgomery multiplication in the Montgomery domain.
    In other words, it computes a * b * R⁻¹ mod q, where `R` is the Montgomery factor.

    Example:
    ```
    %m0 = mod_arith.mont_mul %a, %b : !mod_arith.int<65537 : i256>
    ```
  }];
  let hasFolder = 1;
}

#endif  // ZKIR_DIALECT_MODARITH_IR_MODARITHOPS_TD_
