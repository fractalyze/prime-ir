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
    !poly_ty = !poly.polynomial<!base_field, 32>
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
    !poly_ty = !poly.polynomial<!base_field, 32>
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
    !poly_ty = !poly.polynomial<!base_field, 32>
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
    !poly_ty = !poly.polynomial<!base_field, 32>
    %0 = poly.constant int<1 + x**2> : !poly_ty
    %1 = poly.constant int<x**5 - x + 1> : !poly_ty
    %2 = poly.mul %0, %1 : !poly_ty
    ```
  }];
}

def Poly_ToTensorOp : Poly_Op<"to_tensor", [Pure]> {
  let summary = "Creates a tensor containing the coefficients of a polynomial.";
  let description = [{
    `poly.to_tensor` creates a dense tensor value containing the
    coefficients of the input polynomial. The output tensor contains the
    coefficients in degree-increasing order.

    Example:

    ```mlir
    !base_field = !field.pf<3:i32>
    !poly_ty = !poly.polynomial<!base_field, 32>
    %0 = poly.constant <1 + x**2> : !poly_ty
    %tensor = poly.to_tensor %0
    ```
  }];
  let arguments = (ins Poly_PolyType:$input);
  let results = (outs RankedTensorOf<[AnyType]>:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

def Poly_FromTensorOp : Poly_Op<"from_tensor", [Pure]> {
  let summary = "Creates a polynomial from integer coefficients stored in a tensor.";
  let description = [{
    `polynomial.from_tensor` creates a polynomial value from a tensor of coefficients.
    The input tensor must list the coefficients in degree-increasing order.

    The input one-dimensional tensor may have size at most the return polynomial's
    `maxDegree + 1`, with smaller dimension implying that all higher-degree terms have
    zero coefficient.

    Example:

    ```mlir
    !base_field = !field.pf<3:i32>
    !poly_ty = !poly.polynomial<!base_field, 2>
    %coeffs = tensor.from_elements [2, 2, 5] : tensor<3xi32>
    %poly = poly.from_tensor %coeffs : tensor<3xi32> -> !poly_ty
    ```
  }];
  let arguments = (ins RankedTensorOf<[AnyType]>:$input);
  let results = (outs Poly_PolyType:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

#endif  // ZKIR_DIALECT_POLY_IR_POLYOPS_TD_
