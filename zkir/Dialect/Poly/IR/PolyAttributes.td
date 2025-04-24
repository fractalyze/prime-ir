#ifndef ZKIR_DIALECT_POLY_IR_POLYATTRIBUTES_TD_
#define ZKIR_DIALECT_POLY_IR_POLYATTRIBUTES_TD_

include "zkir/Dialect/Poly/IR/PolyDialect.td"
include "zkir/Dialect/Poly/IR/PolyTypes.td"
include "zkir/Dialect/Field/IR/FieldAttributes.td"

class Poly_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Poly_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Poly_UnivariatePolyAttr : Poly_Attr<"UnivariatePoly", "univariate_polynomial"> {
  let summary = "The attribute representing a univariate polynomial with field coefficients";
  let description = [{
    A polynomial attribute represents a single-variable polynomial with field
    coefficients, which is used to define the modulus of a `QuotientRingAttr`, as well
    as to define constants and perform constant folding for `polynomial` ops.

    The polynomial must be expressed as a list of monomial terms, with addition
    or subtraction between them. The choice of variable name is arbitrary, but
    must be consistent across all the monomials used to define a single
    attribute. The order of monomial terms is arbitrary, each monomial degree
    must occur at most once.

    Example:
    ```
    !PF1 = !field.pf<7:i32>
    !poly_ty = !poly.polynomial<!PF1>
    #poly = #poly.univariate_polynomial<x**3 + x + 1> : !poly_ty
    ```
  }];
  let parameters = (ins Poly_PolyType:$type, "::mlir::polynomial::IntPolynomialAttr":$value);
  let assemblyFormat = "$value `:` $type";
}

def Poly_PrimitiveRootAttr: Poly_Attr<"PrimitiveRoot", "primitive_root"> {
  let summary = "an attribute containing an integer and its degree as a root of unity";
  let description = [{
    A primitive root attribute stores an integer root `value` and an integer
    `degree`, corresponding to a primitive root of unity of the given degree in
    an unspecified ring.

    This is used as an attribute on `poly.ntt` and `poly.intt` ops
    to specify the root of unity used in lowering the transform.

    Example:
    ```
    !PF1 = !field.pf<7:i32>
    #elem = #field.elem<2> : !PF1
    #root = #poly.primitive_root<root=%generator, degree=3>
    ```
  }];
  let extraClassDeclaration = [{
    field::PrimeFieldAttr getInvRoot() const;
    field::PrimeFieldAttr getInvDegree() const;
    DenseElementsAttr getRoots() const;
    DenseElementsAttr getInvRoots() const;
  }];
  let parameters = (ins Field_PrimeFieldAttr:$root, "IntegerAttr":$degree, OptionalParameter<"mod_arith::MontgomeryAttr">:$montgomery);
  let assemblyFormat = "`<` struct(params) `>`";
  let genStorageClass = 0;
}

#endif  // ZKIR_DIALECT_POLY_IR_POLYATTRIBUTES_TD_
