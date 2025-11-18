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
    #3 = #field.pf.elem<3> : !PF
    #curve = #elliptic_curve.sw<#3, #3, (#3, #3)>

    !affine = !elliptic_curve.affine<#curve>
    ```
  }];
  let parameters = (ins EllipticCurve_ShortWeierstrassAttr:$curve);
  let assemblyFormat = "`<` $curve `>`";
}

def EllipticCurve_JacobianType : EllipticCurve_Type<"Jacobian", "jacobian", [MemRefElementTypeInterface]> {
  let summary = "A jacobian point within an elliptic curve";
  let description = [{
    A jacobian point within an elliptic curve. If the point in affine form is represented as (x,y),
    then the jacobian form is (X:Y:Z) where x = X/Z², y = Y/Z³. It must be instantiated with an elliptic
    curve attribute such as `#elliptic_curve.sw` that defines the elliptic curve.

    Example:
    ```
    !PF = !field.pf<35:i32>
    #3 = #field.pf.elem<3> : !PF
    #curve = #elliptic_curve.sw<#3, #3, (#3, #3)>

    !jacobian = !elliptic_curve.affine<#curve> AKA affine
    ```
  }];
  let parameters = (ins EllipticCurve_ShortWeierstrassAttr:$curve);
  let assemblyFormat = "`<` $curve `>`";
}

def EllipticCurve_XYZZType : EllipticCurve_Type<"XYZZ", "xyzz", [MemRefElementTypeInterface, ]> {
  let summary = "A xyzz point within an elliptic curve";
  let description = [{
    A xyzz point within an elliptic curve. The xyzz form is a different way of storing
    a jacobian point. While a jacobian point is stored as (X:Y:Z), the xyzz form is
    stored as (X:Y:Z²:Z³) to make computing x = X/Z², y = Y/Z³ easier. It must be instantiated
    with an elliptic curve attribute such as `#elliptic_curve.sw` that defines the elliptic curve.

    Example:
    ```
    !PF = !field.pf<35:i32>
    #3 = #field.pf.elem<3> : !PF
    #curve = #elliptic_curve.sw<#3, #3, (#3, #3)>

    !xyzz = !elliptic_curve.xyzz<#curve>
    ```
  }];
  let parameters = (ins EllipticCurve_ShortWeierstrassAttr:$curve);
  let assemblyFormat = "`<` $curve `>`";
}

def PointType : AnyTypeOf<[
    EllipticCurve_AffineType, EllipticCurve_JacobianType,
    EllipticCurve_XYZZType]>;

def AffineLike : TypeOrValueSemanticsContainer<
    EllipticCurve_AffineType, "affine-like">;

def JacobianLike : TypeOrValueSemanticsContainer<
    EllipticCurve_JacobianType, "jacobian-like">;

def XYZZLike : TypeOrValueSemanticsContainer<
    EllipticCurve_XYZZType, "xyzz-like">;

def PointLike : TypeConstraint<Or<[
    AffineLike.predicate, JacobianLike.predicate, XYZZLike.predicate]>,
    "affine-like or jacobian-like or xyzz-like">;
#endif  // ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVETYPES_TD_
