#ifndef ZKIR_DIALECT_POLY_IR_POLYOPS_TD_
#define ZKIR_DIALECT_POLY_IR_POLYOPS_TD_

include "zkir/Dialect/Poly/IR/PolyAttributes.td"
include "zkir/Dialect/Poly/IR/PolyDialect.td"
include "zkir/Dialect/Poly/IR/PolyTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Poly_Op<string mnemonic, list<Trait> traits = []> :
    Op<Poly_Dialect, mnemonic, traits # [Pure]> {
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Poly_ConstantOp : Op<Poly_Dialect, "constant",
    [Pure]> {
  let summary = "Define a constant polynomial via an attribute.";
  let description = [{
    Example:

    ```mlir
    !base_field = !field.pf<3:i32>
    !poly_ty = !polynomial.polynomial<!base_field>
    %0 = poly.constant <1 + x**2> : !poly_ty
    ```
  }];
  let arguments = (ins Poly_UnivariatePolyAttr:$value);
  let results = (outs Poly_PolyType:$output);
  let hasCustomAssemblyFormat = 1;
}

class Poly_UnaryOp<string mnemonic, list<Trait> traits = []> :
    Poly_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins Poly_PolyType:$operand);
  let results = (outs Poly_PolyType:$result);
}

class Poly_BinaryOp<string mnemonic, list<Trait> traits = []> :
    Poly_Op<mnemonic, !listconcat(traits, [Pure, SameOperandsAndResultType, ElementwiseMappable])> {
  let arguments = (ins PolyLike:$lhs, PolyLike:$rhs);
  let results = (outs PolyLike:$result);
  let assemblyFormat = "operands attr-dict `:` type($result)";
}

def Poly_AddOp : Poly_BinaryOp<"add", [Commutative]> {
  let summary = "Addition operation between polynomials.";
  let description = [{
    Performs polynomial addition on the operands. The operands may be single
    polynomials or containers of identically-typed polynomials, i.e., polynomials
    from the same underlying ring with the same coefficient types.

    This op is defined to occur in the ring defined by the ring attribute of
    the two operands, meaning the arithmetic is taken modulo the
    polynomialModulus of the ring as well as modulo any semantics defined by
    the coefficient type.

    Example:

    ```mlir
    !base_field = !field.pf<3:i32>
    !poly_ty = !polynomial.polynomial<!base_field>
    %0 = poly.constant int<1 + x**2> : !poly_ty
    %1 = poly.constant int<x**5 - x + 1> : !poly_ty
    %2 = poly.add %0, %1 : !poly_ty
    ```
  }];
}

def Poly_SubOp : Poly_BinaryOp<"sub"> {
  let summary = "Subtraction operation between polynomials.";
  let description = [{
    Performs polynomial subtraction on the operands. The operands may be single
    polynomials or containers of identically-typed polynomials, i.e., polynomials
    from the same underlying ring with the same coefficient types.

    This op is defined to occur in the ring defined by the ring attribute of
    the two operands, meaning the arithmetic is taken modulo the
    polynomialModulus of the ring as well as modulo any semantics defined by
    the coefficient type.

    Example:

    ```mlir
    !base_field = !field.pf<3:i32>
    !poly_ty = !polynomial.polynomial<!base_field>
    %0 = poly.constant int<1 + x**2> : !poly_ty
    %1 = poly.constant int<x**5 - x + 1> : !poly_ty
    %2 = poly.sub %0, %1 : !poly_ty
    ```
  }];
}

def Poly_MulOp : Poly_BinaryOp<"mul", [Commutative]> {
  let summary = "Multiplication operation between polynomials.";
  let description = [{
    Performs polynomial multiplication on the operands. The operands may be single
    polynomials or containers of identically-typed polynomials, i.e., polynomials
    from the same underlying ring with the same coefficient types.

    This op is defined to occur in the ring defined by the ring attribute of
    the two operands, meaning the arithmetic is taken modulo the
    polynomialModulus of the ring as well as modulo any semantics defined by
    the coefficient type.

    Example:

    ```mlir
    !base_field = !field.pf<3:i32>
    !poly_ty = !polynomial.polynomial<!base_field>
    %0 = poly.constant int<1 + x**2> : !poly_ty
    %1 = poly.constant int<x**5 - x + 1> : !poly_ty
    %2 = poly.mul %0, %1 : !poly_ty
    ```
  }];
}

#endif  // ZKIR_DIALECT_POLY_IR_POLYOPS_TD_
