#ifndef ZKIR_DIALECT_POLY_IR_POLYATTRIBUTES_TD_
#define ZKIR_DIALECT_POLY_IR_POLYATTRIBUTES_TD_

include "zkir/Dialect/Poly/IR/PolyDialect.td"
include "zkir/Dialect/Poly/IR/PolyTypes.td"

class Poly_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Poly_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Poly_UnivariatePolyAttr : Poly_Attr<"UnivariatePoly", "univariate_polynomial"> {
  let summary = "an attribute containing a single-variable polynomial with field coefficients";
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

    ```mlir
    !PF1 = !field.pf<7:i32>
    !poly_ty = !poly.polynomial<!PF1>
    #poly = #poly.univariate_polynomial<x**3 + x + 1> : !poly_ty
    ```
  }];
  let parameters = (ins Poly_PolyType:$type, "::mlir::polynomial::IntPolynomialAttr":$value);
  let assemblyFormat = "$value `:` $type";
}

#endif  // ZKIR_DIALECT_POLY_IR_POLYATTRIBUTES_TD_
