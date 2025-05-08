#ifndef ZKIR_DIALECT_FIELD_IR_FIELDOPS_TD_
#define ZKIR_DIALECT_FIELD_IR_FIELDOPS_TD_

include "zkir/Dialect/Field/IR/FieldAttributes.td"
include "zkir/Dialect/Field/IR/FieldDialect.td"
include "zkir/Dialect/Field/IR/FieldTypes.td"
include "zkir/Dialect/ModArith/IR/ModArithAttributes.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def PrimeField_ConstantOp : Op<Field_Dialect, "pf.constant", [Pure]> {
  let summary = "Define a constant prime field element via an attribute";
  let description = [{
    Example:
    ```
    %0 = field.pf.constant 123 : field.pf<7>
    ```
  }];
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs PrimeFieldLike:$output);
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "PrimeFieldType":$ty, "llvm::APInt":$value), [{
      return build($_builder, $_state, ty, IntegerAttr::get(ty.getModulus().getType(), value));
    }]>,
    OpBuilder<(ins "PrimeFieldType":$ty, "uint64_t":$value), [{
      return build($_builder, $_state, ty, IntegerAttr::get(ty.getModulus().getType(), value));
    }]>
  ];
}

// Base class for Prime Field operations.
class PrimeField_Op<string mnemonic, list<Trait> traits = [Pure]> :
  Op<Field_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::zkir::field";
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

def PrimeField_EncapsulateOp : PrimeField_Op<"pf.encapsulate", [Pure, ElementwiseMappable]> {
  let summary = "Encapsulate an integer into a field type";
  let description = [{
    Converts the integer to be of field type.

    Example:
    ```
    field.pf.encapsulate %c0 : i32 -> field.pf<7>
    field.pf.encapsulate %c1 : i64 -> field.pf<7>
    ```
  }];
  let arguments = (ins SignlessIntegerLike:$input);
  let results = (outs PrimeFieldLike:$output);
  let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";
}

def PrimeField_ExtractOp : PrimeField_Op<"pf.extract", [Pure, ElementwiseMappable]> {
  let summary = "Extract the integer stored inside field type";
  let description = [{
    Extracts the integer inside the field type.

    It is required that the bitwidth of the output integer type is the same
    as that of the storage type of the input field type.

    Example:
    ```
    %c0 = field.pf.extract %f0 : field.pf<7 : i32> -> i32
    ```
  }];
  let arguments = (ins PrimeFieldLike:$input);
  let results = (outs SignlessIntegerLike:$output);
  let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";
}

def PrimeField_ToMontOp : PrimeField_Op<"pf.to_mont", [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
  let summary = "Convert from standard form to montgomery representation";
  let description = [{
    Computes x * R mod q, where `R` is the montgomery factor.

    Example:
    ```
    %m0 = field.pf.to_mont %c0 {montgomery=#mont} : !PF
    ```
  }];
  let arguments = (ins PrimeFieldLike:$input, ModArith_MontgomeryAttr:$montgomery);
  let results = (outs PrimeFieldLike:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

def PrimeField_FromMontOp : PrimeField_Op<"pf.from_mont", [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
  let summary = "Convert from montgomery representation to standard form";
  let description = [{
    Computes x * R⁻¹ mod q, where `R` is the montgomery factor.

    Example:
    ```
    %m0 = field.pf.from_mont %c0 {montgomery=#mont} : !PF
    ```
  }];
  let arguments = (ins PrimeFieldLike:$input, ModArith_MontgomeryAttr:$montgomery);
  let results = (outs PrimeFieldLike:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

// Prime field multiplicative inverse
def PrimeField_InverseOp : PrimeField_Op<"pf.inverse", [Involution, SameOperandsAndResultType]> {
  let summary = "Prime field multiplicative inverse";
  let description = [{
    Computes the multiplicative inverse of a field element.

    Example:
    ```
    %inv_a = field.pf.inverse %a : field.pf<primeModulus>
    ```
  }];
  let arguments = (ins PrimeFieldLike:$input);
  let results = (outs PrimeFieldLike:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

// Prime field negation
def PrimeField_NegateOp : PrimeField_Op<"pf.negate", [Pure, ElementwiseMappable, SameOperandsAndResultType, Involution]> {
  let summary = "Prime field negation";
  let description = [{
    Computes the negation of a field element.

    Example:
    ```
    %negation = field.pf.negate %a : field.pf<primeModulus>
    ```
  }];
  let arguments = (ins PrimeFieldLike:$input);
  let results = (outs PrimeFieldLike:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

// A helper class for binary prime field operations.
class PrimeField_BinaryOp<string mnemonic, list<Trait> traits = []> :
  PrimeField_Op<mnemonic, traits # [SameOperandsAndResultType, Pure, ElementwiseMappable]>,
  Arguments<(ins PrimeFieldLike:$lhs, PrimeFieldLike:$rhs)>,
  Results<(outs PrimeFieldLike:$output)> {
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

// Prime field addition.
def PrimeField_AddOp : PrimeField_BinaryOp<"pf.add", [Commutative]> {
  let summary = "Prime field addition";
  let description = [{
    Computes the sum of two prime field elements.

    Example:
    ```
    %r = field.pf.add %a, %b : field.pf<primeModulus>
    ```
  }];
}

// Prime field double.
def PrimeField_DoubleOp : PrimeField_Op<"pf.double", [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
  let summary = "Prime field double";
  let description = [{
    Computes the double of a field element.

    Example:
    ```
    %double = field.pf.double %a : field.pf<primeModulus>
    ```
  }];
  let arguments = (ins PrimeFieldLike:$input);
  let results = (outs PrimeFieldLike:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

// Prime field subtraction.
def PrimeField_SubOp : PrimeField_BinaryOp<"pf.sub"> {
  let summary = "Prime field subtraction";
  let description = [{
    Computes the difference between two prime field elements.

    Example:
    ```
    %r = field.pf.sub %a, %b : field.pf<primeModulus>
    ```
  }];
}

// Prime field multiplication.
def PrimeField_MulOp : PrimeField_BinaryOp<"pf.mul", [Commutative]> {
  let summary = "Prime field multiplication";
  let description = [{
    Computes the product of two prime field elements.

    Example:
    ```
    %r = field.mul %a, %b : field.pf<primeModulus>
    ```
  }];
}

// Prime field square.
def PrimeField_SquareOp : PrimeField_Op<"pf.square", [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
  let summary = "Prime field square";
  let description = [{
    Computes the square of a field element.

    Example:
    ```
    %square = field.pf.square %a : field.pf<primeModulus>
    ```
  }];

  let arguments = (ins PrimeFieldLike:$input);
  let results = (outs PrimeFieldLike:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

def PrimeField_MontMulOp : PrimeField_BinaryOp<"pf.mont_mul", [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
  let summary = "Montgomery multiplication";
  let description = [{
    Computes a * b * R⁻¹ mod q, where `R` is the montgomery factor.

    Example:
    ```
    %m0 = field.pf.mont_mul %a, %b {montgomery=#mont} : !PF
    ```
  }];
  let arguments = (ins
    PrimeFieldLike:$lhs,
    PrimeFieldLike:$rhs,
    ModArith_MontgomeryAttr:$montgomery
  );
  let results = (outs PrimeFieldLike:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

def PrimeField_CmpOp : PrimeField_Op<"pf.cmp", [SameTypeOperands, TypesMatchWith<
  "result type has i1 element type and same shape as operands",
  "lhs", "result", "getI1SameShape($_self)">]> {
  let summary = "Prime field comparison operation";
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
    %res = field.pf.cmp slt, %a, %b : !PF
    ```
  }];
  let arguments = (ins Arith_CmpIPredicateAttr:$predicate,
                       PrimeFieldLike:$lhs,
                       PrimeFieldLike:$rhs);
  let results = (outs BoolLike:$result);
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}

#endif  // ZKIR_DIALECT_FIELD_IR_FIELDOPS_TD_
