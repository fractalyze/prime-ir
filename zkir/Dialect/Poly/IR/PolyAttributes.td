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
  let summary = "an attribute containing a primitive root of unity and its twiddles";
  let description = [{
    A primitive root attribute stores a primitive root of unity specified by the `RootOfUnityAttr`
    attribute and also some other necessary precomputed values for (I)NTT operations.

    Example:
    ```
    !PF1 = !field.pf<7:i32>
    #elem = #field.elem<2> : !PF1
    #root_of_unity = #field.root_of_unity<root=#elem, degree=3>
    #root = #poly.primitive_root<#root_of_unity>
    ```
  }];
  let extraClassDeclaration = [{
    IntegerAttr getDegree() const;
    field::PrimeFieldAttr getRoot() const;
    field::PrimeFieldAttr getInvRoot() const;
    field::PrimeFieldAttr getInvDegree() const;
    DenseElementsAttr getRoots() const;
    DenseElementsAttr getInvRoots() const;
  }];
  let parameters = (ins Field_RootOfUnityAttr:$rootOfUnity, OptionalParameter<"mod_arith::MontgomeryAttr">:$montgomery);
  let assemblyFormat = "`<` params `>`";
  let genStorageClass = 0;
}

#endif  // ZKIR_DIALECT_POLY_IR_POLYATTRIBUTES_TD_
