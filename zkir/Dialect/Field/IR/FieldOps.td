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

// Base class for field operations.
class Field_Op<string mnemonic, list<Trait> traits = []> :
  Op<Field_Dialect, mnemonic, traits #
   [NoMemoryEffect, Pure] # ElementwiseMappable.traits>;

// Base class for field arithmetic ops. All ops have one result,
// and the result type is the same as the operand type.
class Field_ArithOp<string mnemonic, list<Trait> traits = []> :
    Field_Op<mnemonic, traits # [SameOperandsAndResultType]>;

// Base class for unary field arithmetic.
class Field_UnaryOp<string mnemonic, list<Trait> traits = []> :
    Field_ArithOp<mnemonic, traits>,
  Arguments<(ins FieldLike:$input)>,
  Results<(outs FieldLike:$output)> {
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

// Base class for binary field arithmetic.
class Field_BinaryOp<string mnemonic, list<Trait> traits = []> :
  Field_ArithOp<mnemonic, traits>,
  Arguments<(ins FieldLike:$lhs, FieldLike:$rhs)>,
  Results<(outs FieldLike:$output)> {
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

def Field_ConstantOp : Op<Field_Dialect, "constant", [ConstantLike, Pure]> {
  let summary = "Define a constant prime field element via an attribute";
  let description = [{
    Example:
    ```
    %0 = field.constant 123 : field.pf<7>
    ```
  }];
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs FieldLike:$output);
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let builders = [
    OpBuilder<(ins "Type":$ty, "uint64_t":$value), [{
      if (auto pfType = dyn_cast<PrimeFieldType>(ty)) {
        return build($_builder, $_state, pfType, value);
      } else if (auto f2Type = dyn_cast<QuadraticExtFieldType>(ty)) {
        return build($_builder, $_state, f2Type, value);
      } else {
        llvm_unreachable("Unsupported field type for constant");
      }
    }]>,
    OpBuilder<(ins "PrimeFieldType":$ty, "llvm::APInt":$value), [{
      return build($_builder, $_state, ty, PrimeFieldAttr::get(ty, value));
    }]>,
    OpBuilder<(ins "PrimeFieldType":$ty, "uint64_t":$value), [{
      return build($_builder, $_state, ty, PrimeFieldAttr::get(ty, value));
    }]>,
    OpBuilder<(ins "PrimeFieldType":$ty, "IntegerAttr":$value), [{
      return build($_builder, $_state, ty, PrimeFieldAttr::get(ty.getContext(), ty, value));
    }]>,
    OpBuilder<(ins "QuadraticExtFieldType":$ty, "uint64_t":$low), [{
      PrimeFieldAttr lowAttr = PrimeFieldAttr::get(ty.getBaseField(), low);
      PrimeFieldAttr highAttr = PrimeFieldAttr::get(ty.getBaseField(), 0);
      return build($_builder, $_state, ty, QuadraticExtFieldAttr::get(ty.getContext(), ty, lowAttr, highAttr));
    }]>,
    OpBuilder<(ins "QuadraticExtFieldType":$ty, "uint64_t":$low, "uint64_t":$high), [{
      PrimeFieldAttr lowAttr = PrimeFieldAttr::get(ty.getBaseField(), low);
      PrimeFieldAttr highAttr = PrimeFieldAttr::get(ty.getBaseField(), high);
      return build($_builder, $_state, ty, QuadraticExtFieldAttr::get(ty.getContext(), ty, lowAttr, highAttr));
    }]>
  ];
}

def Field_F2ConstantOp : Op<Field_Dialect, "f2.constant", [Pure, TypesMatchWith<
  "result type is the same as the operand type", "output", "low", "cast<QuadraticExtFieldType>($_self).getBaseField()">,
  TypesMatchWith< "result type is the same as the operand type", "output", "high", "cast<QuadraticExtFieldType>($_self).getBaseField()">]> {
  let summary = "Define a constant quadratic extension field element";
  let description = [{
    Example:
    ```
    !QF = !field.f2<!PF, #beta>
    %0 = field.f2.constant %low, %high : !QF
    ```
  }];
  let arguments = (ins Field_PrimeFieldType:$low, Field_PrimeFieldType:$high);
  let results = (outs Field_QuadraticExtFieldType:$output);
  let assemblyFormat = "$low `,` $high attr-dict `:` type($output)";
}

def Field_EncapsulateOp : Field_Op<"pf.encapsulate"> {
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

def Field_ExtractOp : Field_Op<"pf.extract"> {
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

def Field_CmpOp : Field_Op<"cmp", [SameTypeOperands, TypesMatchWith<
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
    %res = field.cmp slt, %a, %b : !PF
    ```
  }];
  let arguments = (ins Arith_CmpIPredicateAttr:$predicate,
                       FieldLike:$lhs,
                       FieldLike:$rhs);
  let results = (outs BoolLike:$result);
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}

def Field_ToMontOp : Field_Op<"to_mont", [TypesMatchWith<
  "input type is standard form of output type", "output", "input",
  "getStandardFormType($_self)">]> {
  let summary = "Convert from standard form to montgomery representation";
  let description = [{
    Computes x * R, where `R` is the montgomery factor.

    Example:
    ```
    %m0 = field.to_mont %c0 : !PF
    ```
  }];
  let arguments = (ins FieldLike:$input);
  let results = (outs FieldLike:$output);
  let hasVerifier = 1;
  let assemblyFormat = "$input attr-dict `:` type($output)";
}

def Field_FromMontOp : Field_Op<"from_mont", [TypesMatchWith<
  "input type is montgomery form of output type", "output", "input",
  "getMontgomeryFormType($_self)">]> {
  let summary = "Convert from montgomery representation to standard form";
  let description = [{
    Computes x * R⁻¹, where `R` is the montgomery factor.

    Example:
    ```
    %m0 = field.from_mont %c0 : !PF
    ```
  }];
  let arguments = (ins FieldLike:$input);
  let results = (outs FieldLike:$output);
  let hasVerifier = 1;
  let assemblyFormat = "$input attr-dict `:` type($output)";
}

// Field multiplicative inverse
def Field_InverseOp : Field_UnaryOp<"inverse", [Involution]> {
  let summary = "Field multiplicative inverse";
  let description = [{
    Computes the multiplicative inverse of a field element.

    Example:
    ```
    %inv_a = field.inverse %a : field.pf<primeModulus>
    ```
  }];
  let hasVerifier = 1;
}

// Field double.
def Field_DoubleOp : Field_UnaryOp<"double"> {
  let summary = "Field double";
  let description = [{
    Computes the double of a field element.

    Example:
    ```
    %double = field.double %a : field.pf<primeModulus>
    ```
  }];
}

// Field square.
def Field_SquareOp : Field_UnaryOp<"square"> {
  let summary = "Field square";
  let description = [{
    Computes the square of a field element.

    Example:
    ```
    %square = field.square %a : field.pf<primeModulus>
    ```
  }];
}

// Field negation
def Field_NegateOp : Field_UnaryOp<"negate", [Involution]> {
  let summary = "Field negation";
  let description = [{
    Computes the negation of a field element.

    Example:
    ```
    %negation = field.negate %a : field.pf<primeModulus>
    ```
  }];
  let hasVerifier = 1;
}

// Field addition.
def Field_AddOp : Field_BinaryOp<"add", [Commutative]> {
  let summary = "Field addition";
  let description = [{
    Computes the sum of two field elements.

    Example:
    ```
    %r = field.add %a, %b : field.pf<primeModulus>
    ```
  }];
  let hasVerifier = 1;
}

// Field subtraction.
def Field_SubOp : Field_BinaryOp<"sub"> {
  let summary = "Field subtraction";
  let description = [{
    Computes the difference between two field elements.

    Example:
    ```
    %r = field.sub %a, %b : field.pf<primeModulus>
    ```
  }];
  let hasVerifier = 1;
}

// Field multiplication.
def Field_MulOp : Field_BinaryOp<"mul", [Commutative]> {
  let summary = "Field multiplication";
  let description = [{
    Computes the product of two field elements.

    Example:
    ```
    %r = field.mul %a, %b : field.pf<primeModulus>
    ```
  }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

#endif  // ZKIR_DIALECT_FIELD_IR_FIELDOPS_TD_
