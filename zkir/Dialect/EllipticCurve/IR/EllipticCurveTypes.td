#ifndef ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVETYPES_TD_
#define ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVETYPES_TD_

include "zkir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.td"
include "zkir/Dialect/EllipticCurve/IR/EllipticCurveDialect.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/AttrTypeBase.td"

// Base type definition for EllipticCurve types.
class EllipticCurve_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<EllipticCurve_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def EllipticCurve_AffineType : EllipticCurve_Type<"Affine", "affine", [MemRefElementTypeInterface]> {
  let summary = "An affine point within an elliptic curve";
  let description = [{
    An affine point within an elliptic curve. It must be instantiated with an elliptic curve attribute
    such as `#elliptic_curve.sw` that defines the elliptic curve.

    Example:
    ```
    !PF = !field.pf<35:i32>
    #3 = #field.pf_elem<3> : !PF
    #curve = #elliptic_curve.sw<#3, #3, (#3, #3)>

    !affine = !elliptic_curve.affine<#curve>
    ```
  }];
  let parameters = (ins EllipticCurve_ShortWeierstrassAttr:$curve);
  let assemblyFormat = "`<` $curve `>`";
}

def EllipticCurve_JacobianType : EllipticCurve_Type<"Jacobian", "jacobian", [MemRefElementTypeInterface]> {
  let summary = "A Jacobian point within an elliptic curve";
  let description = [{
    A Jacobian point within an elliptic curve. If the point in affine form is represented as (x,y),
    then the Jacobian form is (X:Y:Z) where x = X/Z², y = Y/Z³. It must be instantiated with an elliptic
    curve attribute such as `#elliptic_curve.sw` that defines the elliptic curve.

    Example:
    ```
    !PF = !field.pf<35:i32>
    #3 = #field.pf_elem<3> : !PF
    #curve = #elliptic_curve.sw<#3, #3, (#3, #3)>

    !jacobian = !elliptic_curve.affine<#curve> AKA AffineType
    ```
  }];
  let parameters = (ins EllipticCurve_ShortWeierstrassAttr:$curve);
  let assemblyFormat = "`<` $curve `>`";
}

def EllipticCurve_XYZZType : EllipticCurve_Type<"XYZZ", "xyzz", [MemRefElementTypeInterface, ]> {
  let summary = "A XYZZ point within an elliptic curve";
  let description = [{
    A XYZZ point within an elliptic curve. The XYZZ form is a different way of storing
    a Jacobian point. While a Jacobian point is stored as (X:Y:Z), the XYZZ form is
    stored as (X:Y:Z²:Z³) to make computing x = X/Z², y = Y/Z³ easier. It must be instantiated
    with an elliptic curve attribute such as `#elliptic_curve.sw` that defines the elliptic curve.

    Example:
    ```
    !PF = !field.pf<35:i32>
    #3 = #field.pf_elem<3> : !PF
    #curve = #elliptic_curve.sw<#3, #3, (#3, #3)>

    !xyzz = !elliptic_curve.xyzz<#curve>
    ```
  }];
  let parameters = (ins EllipticCurve_ShortWeierstrassAttr:$curve);
  let assemblyFormat = "`<` $curve `>`";
}

#endif  // ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVETYPES_TD_
