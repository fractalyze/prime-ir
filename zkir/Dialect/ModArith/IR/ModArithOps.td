#ifndef ZKIR_DIALECT_MODARITH_IR_MODARITHOPS_TD_
#define ZKIR_DIALECT_MODARITH_IR_MODARITHOPS_TD_

include "zkir/Dialect/ModArith/IR/ModArithAttributes.td"
include "zkir/Dialect/ModArith/IR/ModArithDialect.td"
include "zkir/Dialect/ModArith/IR/ModArithTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class ModArith_Op<string mnemonic, list<Trait> traits = [Pure]>
    : Op<ModArith_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::zkir::mod_arith";
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// type conversion operations
def ModArith_EncapsulateOp : ModArith_Op<"encapsulate", [Pure, ElementwiseMappable]> {
  let summary = "Encapsulate an integer into a modular type integer";
  let description = [{
    Converts the integer to be of modular type integer.

    It is required that the bitwidth of the input integer type is the same
    as that of the storage type of the output modular integer type.

    Example:
    ```
    mod_arith.encapsulate %c0 : i32 -> mod_arith.int<65537 : i32>
    mod_arith.encapsulate %c1 : i64 -> mod_arith.int<65537>
    ```
  }];
  let arguments = (ins SignlessIntegerLike:$input);
  let results = (outs ModArithLike:$output);
  let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";
}

def ModArith_ExtractOp : ModArith_Op<"extract", [Pure, ElementwiseMappable]> {
  let summary = "Extract the integer from inside modular type integer";
  let description = [{
    Extracts the integer inside the modular type integer.

    It is required that the bitwidth of the output integer type is the same
    as that of the storage type of the input modular type integer.

    Example:
    ```
    %m0 = mod_arith.encapsulate %c0 : i32 -> mod_arith.int<65537 : i32>
    %m1 = mod_arith.encapsulate %c1 : i64 -> mod_arith.int<65537>
    %c2 = mod_arith.extract %m0 : mod_arith.int<65537 : i32> -> i32
    %c3 = mod_arith.extract %m1 : mod_arith.int<65537> -> i64
    ```
  }];
  let arguments = (ins ModArithLike:$input);
  let results = (outs SignlessIntegerLike:$output);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";
}

def ModArith_ModSwitchOp : ModArith_Op<"mod_switch", [Pure, SameOperandsAndResultShape]> {
  let summary = "Change the modulus";
  let description = [{
    Switches the modulus of a modular type integer.

    Example:
    ```
    `mod_arith.mod_switch %c0 : mod_arith.int<65537 : i32> to mod_arith.int<65539 : i32>`
    ```
  }];
  let arguments = (ins ModArith_ModArithType:$input);
  let results = (outs ModArith_ModArithType:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

def ModArith_ConstantOp : Op<ModArith_Dialect, "constant",
    [Pure]> {
  let summary = "Define a constant value via an attribute.";
  let description = [{
    Example:

    ```
    %0 = mod_arith.constant 123 : !mod_arith.int<65537:i32>
    ```
  }];
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs ModArithLike:$output);
  let hasCustomAssemblyFormat = 1;
}

def ModArith_NegateOp : ModArith_Op<"negate", [Pure, ElementwiseMappable, SameOperandsAndResultType, Involution]> {
  let summary = "Modular negation";
  let description = [{
    Computes the negation of a modular type integer.

    Example:
    ```
    %1 = mod_arith.negate %0 : mod_arith.int<65537: i32>
    ```
  }];
  let arguments = (ins ModArithLike:$input);
  let results = (outs ModArithLike:$output);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

def ModArith_ReduceOp : ModArith_Op<"reduce", [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
  let summary = "Modular reduction";
  let description = [{
    Computes the modulo of a given integer.

    Example:
    ```
    %c0 = arith.constant 65538 : i32
    %m0 = mod_arith.encapsulate %c0 : i32 -> mod_arith.int<65537 : i32>
    // mod_arith.extract %m0 produces 65538
    %m1 = mod_arith.reduce %m0 : mod_arith.int<65537: i32>
    // mod_arith.extract %m1 produces 1
    ```
  }];
  let arguments = (ins ModArithLike:$input);
  let results = (outs ModArithLike:$output);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

def ModArith_MontReduceOp : ModArith_Op<"mont_reduce", [Pure, ElementwiseMappable, SameTypeOperands]> {
  let summary = "Montgomery reduction";
  let description = [{
    Performs Montgomery reduction on an integer that is twice the bitwidth of the modulus,
    represented as separate low and high parts. This is typically used for reducing the output of extended
    multiplication operations.

    Example:
    ```
    %c0 = arith.constant 65538231231312311 : i512
    %m0 = mod_arith.mont_reduce %c0 {montgomery=#mont} : i512 -> mod_arith.int<65537 : i256>
    ```
  }];
  let arguments = (ins
    SignlessIntegerLike:$low,
    SignlessIntegerLike:$high,
    ModArith_MontgomeryAttr:$montgomery
  );
  let results = (outs ModArithLike:$output);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($low) `->` type($output)";
}

def ModArith_ToMontOp : ModArith_Op<"to_mont", [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
  let summary = "Convert from standard form to montgomery representation";
  let description = [{
    Transforms modular type integer to the Montgomery domain.
    In other words, it computes x * R⁻¹ mod q for the given operand `x`, where `R` is the montgomery factor.

    Example:
    ```
    %m0 = mod_arith.to_mont %c0 {montgomery=#mont} : mod_arith.int<65537 : i256>
    ```
  }];
  let arguments = (ins
    ModArithLike:$input,
    ModArith_MontgomeryAttr:$montgomery
  );
  let results = (outs ModArithLike:$output);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

def ModArith_FromMontOp : ModArith_Op<"from_mont", [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
  let summary = "Convert from montgomery representation to standard form";
  let description = [{
    Transforms modular type integer back from the Montgomery domain.
    In other words, it computes x * R mod q for the given operand `x`, where `R` is the montgomery factor.

    Example:
    ```
    %m0 = mod_arith.from_mont %c0 {montgomery=#mont} : mod_arith.int<65537 : i256>
    ```
  }];
  let arguments = (ins
    ModArithLike:$input,
    ModArith_MontgomeryAttr:$montgomery
  );
  let results = (outs ModArithLike:$output);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

def ModArith_InverseOp : ModArith_Op<"inverse", [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
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
}

class ModArith_BinaryOp<string mnemonic, list<Trait> traits = []> :
    ModArith_Op<mnemonic, traits # [SameOperandsAndResultType, Pure, ElementwiseMappable]>,
    Arguments<(ins ModArithLike:$lhs, ModArithLike:$rhs)>,
    Results<(outs ModArithLike:$output)> {
  let hasVerifier = 1;
  let assemblyFormat ="operands attr-dict `:` type($output)";
}

def ModArith_AddOp : ModArith_BinaryOp<"add", [Commutative]> {
  let summary = "Modular addition operation";
  let description = [{
    Performs modular addition on the given two modular type integers.
  }];
}

def ModArith_SubOp : ModArith_BinaryOp<"sub"> {
  let summary = "Modular subtraction operation";
  let description = [{
    Performs modular subtraction on the given two modular type integers.
  }];
}

def ModArith_MulOp : ModArith_BinaryOp<"mul", [Commutative]> {
  let summary = "Modular multiplication operation";
  let description = [{
    Performs modular multiplication on the given two modular type integers.
    WARNING: This operation assumes that both operands are in the standard form.
  }];
}

def ModArith_MontMulOp : ModArith_Op<"mont_mul", [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
  let summary = "Montgomery multiplication operation";
  let description = [{
    Performs Montgomery multiplication in the Montgomery domain.
    In other words, it computes a * b * R⁻¹ mod q, where `R` is the Montgomery factor.

    Example:
    ```
    %m0 = mod_arith.mont_mul %a, %b : mod_arith.int<65537 : i256>
    ```
  }];
  let arguments = (ins
    ModArithLike:$lhs,
    ModArithLike:$rhs,
    ModArith_MontgomeryAttr:$montgomery
  );
  let results = (outs ModArithLike:$output);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($output)";
}


def ModArith_MacOp : ModArith_Op<"mac", [SameOperandsAndResultType, Pure, ElementwiseMappable]> {
  let summary = "Modular multiplication-and-accumulation operation";
  let description = [{
    Multiplies the first operand with the second operand and adds the result to the third operand.
  }];
  let arguments = (ins ModArithLike:$lhs, ModArithLike:$rhs, ModArithLike:$acc);
  let results = (outs ModArithLike:$output);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

def ModArith_SubIfGEOp : ModArith_Op<"subifge", [SameOperandsAndResultType]> {
  let summary = "Subtract if greater than or equal to";
  let description = [{
    Subtracts the second operand from the first operand if the first operand is greater than or equal to the second operand.
    Otherwise, it returns the first operand unchanged.
  }];
  let arguments = (ins
    SignlessIntegerLike:$lhs,
    SignlessIntegerLike:$rhs
  );
  let results = (outs SignlessIntegerLike:$output);
  let assemblyFormat = "operands attr-dict `:` qualified(type($output))";
}

#endif  // ZKIR_DIALECT_MODARITH_IR_MODARITHOPS_TD_
