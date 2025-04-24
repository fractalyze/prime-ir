#ifndef ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEOPS_TD_
#define ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEOPS_TD_

include "zkir/Dialect/EllipticCurve/IR/EllipticCurveDialect.td"
include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.td"

include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "zkir/Dialect/Field/IR/FieldAttributes.td"

def EllipticCurve_AnyPointType : AnyTypeOf<[
    EllipticCurve_AffineType,
    EllipticCurve_JacobianType,
    EllipticCurve_XYZZType
]>;

// Base class for elliptic curve operations.
class EllipticCurve_Op<string mnemonic, list<Trait> traits = [Pure]> :
  Op<EllipticCurve_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::zkir::elliptic_curve";
}

class EllipticCurve_UnaryOp<string mnemonic, list<Trait> traits = []> :
    EllipticCurve_Op<mnemonic, traits # [Pure, ElementwiseMappable]> {
  let arguments = (ins EllipticCurve_AnyPointType:$input);
  let results = (outs EllipticCurve_AnyPointType:$output);
}

class EllipticCurve_BinaryOp<string mnemonic, list<Trait> traits = []> :
  EllipticCurve_Op<mnemonic, traits # [Pure, ElementwiseMappable]>,
  Arguments<(ins EllipticCurve_AnyPointType:$lhs, EllipticCurve_AnyPointType:$rhs)>,
  Results<(outs EllipticCurve_AnyPointType:$output)> {
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)";
}

////////////// POINT INITIALIZATIONS //////////////

def EllipticCurve_PointOp : EllipticCurve_Op<"point"> {
  let summary = "An elliptic curve point";
  let description = [{
    Defines an elliptic curve point in affine(x, y), Jacobian(x, y, z), or XYZZ(x, y, z², z³) form.

    Example:
    ```
    %0 = elliptic_curve.point %x, %y : !elliptic_curve.affine<#curve>
    %1 = elliptic_curve.point %x, %y, %z : !elliptic_curve.jacobian<#curve>
    %2 = elliptic_curve.point %x, %y, %zz, %zzz : !elliptic_curve.xyzz<#curve>
    ```
  }];
  let arguments = (ins Field_PrimeFieldAttr:$x, Field_PrimeFieldAttr:$y,
                   OptionalAttr<Field_PrimeFieldAttr>:$z, OptionalAttr<Field_PrimeFieldAttr>:$zz,
                   OptionalAttr<Field_PrimeFieldAttr>:$zzz);
  let results = (outs EllipticCurve_AnyPointType:$output);
  let hasCustomAssemblyFormat = 1;
}

////////////// BINARY OPERATIONS //////////////

// // Elliptic curve point addition.
def EllipticCurve_AddOp : EllipticCurve_BinaryOp<"add", [Commutative]> {
  let summary = "Elliptic curve point addition operation";
  let description = [{
    Computes the sum of two elliptic curve points.

    affine, affine -> Jacobian
    affine, Jacobian -> Jacobian
    affine, XYZZ -> XYZZ
    Jacobian, Jacobian -> Jacobian
    XYZZ, XYZZ -> XYZZ

    Example:
    ```
    %sum = elliptic_curve.add %a, %b : !affine, !jacobian -> !jacobian
    ```
  }];
}

// // Elliptic curve point subtraction.
def EllipticCurve_SubOp : EllipticCurve_BinaryOp<"sub"> {
  let summary = "Elliptic curve point subtraction operation";
  let description = [{
    Computes the difference between two elliptic curve points.

    affine, affine -> Jacobian
    affine, Jacobian -> Jacobian
    affine, XYZZ -> XYZZ
    Jacobian, Jacobian -> Jacobian
    XYZZ, XYZZ -> XYZZ

    Example:
    ```
    %difference = elliptic_curve.sub %a, %b : !affine, !jacobian -> !jacobian
    ```
  }];
}

// Elliptic curve point negation.
def EllipticCurve_NegOp : EllipticCurve_UnaryOp<"neg", [Involution, SameOperandsAndResultType]> {
  let summary = "Elliptic curve point negation operation";
  let description = [{
    Computes the negation of an elliptic curve point.

    affine -> Jacobian
    Jacobian -> Jacobian
    XYZZ -> XYZZ

    Example:
    ```
    %negation = elliptic_curve.neg %a : !jacobian
    ```
  }];
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

// Elliptic curve point doubling.
def EllipticCurve_DblOp : EllipticCurve_UnaryOp<"dbl"> {
  let summary = "Elliptic curve point doubling operation";
  let description = [{
    Computes the double of an elliptic curve point. If an affine point is doubled,
    it becomes Jacobian. If a Jacobian or XYZZ point is doubled, its type remains the same.

    affine -> Jacobian
    Jacobian -> Jacobian
    XYZZ -> XYZZ

    Example:
    ```
    %doubled = elliptic_curve.dbl %a : !affine -> !jacobian
    ```
  }];
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";
}

// Elliptic curve point scalar multiplication.
def EllipticCurve_ScalarMulOp : EllipticCurve_Op<"scalar_mul", [Pure, Commutative, ElementwiseMappable]> {
  let summary = "Elliptic curve point scalar multiplication";
  let description = [{
    Computes the product of an elliptic curve point and a scalar prime field value.

    affine -> Jacobian
    Jacobian -> Jacobian
    XYZZ -> XYZZ

    Example:
    ```
    %product = elliptic_curve.scalar_mul %affine1, %b : !affine, !PF -> !jacobian
    ```
  }];
  let arguments = (ins EllipticCurve_AnyPointType:$input, Field_PrimeFieldType:$multiplier);
  let results = (outs EllipticCurve_AnyPointType:$output);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($input) `,` type($multiplier) `->` type($output)";
}

////////////// POINT CONVERSIONS //////////////

def EllipticCurve_ConvertPointTypeOp : EllipticCurve_UnaryOp<"convert_point_type"> {
  let summary = "Convert a point from one form to another.";
  let description = [{
    Converts a point on an elliptic curve from one form to another.
    Possible forms are Affine, Jacobian, and XYZZ.

    Example:
    ```
    !affine = !elliptic_curve.affine<#sw>
    !jacobian = !elliptic_curve.jacobian<#sw>

    %converted = elliptic_curve.convert_point_type %0 : !affine -> !jacobian
    ```
  }];
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";
}

#endif  // ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEOPS_TD_
