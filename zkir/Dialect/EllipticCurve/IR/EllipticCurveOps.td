#ifndef ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEOPS_TD_
#define ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEOPS_TD_

include "zkir/Dialect/EllipticCurve/IR/EllipticCurveDialect.td"
include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.td"

include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "zkir/Dialect/Field/IR/FieldAttributes.td"
include "zkir/Dialect/Field/IR/FieldTypes.td"

// Base class for elliptic curve operations.
class EllipticCurve_Op<string mnemonic, list<Trait> traits = [Pure]> :
  Op<EllipticCurve_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::zkir::elliptic_curve";
}

class EllipticCurve_UnaryOp<string mnemonic, list<Trait> traits = []> :
    EllipticCurve_Op<mnemonic, traits # [Pure, ElementwiseMappable]> {
  let arguments = (ins PointLike:$input);
  let results = (outs PointLike:$output);
}

class EllipticCurve_BinaryOp<string mnemonic, list<Trait> traits = []> :
  EllipticCurve_Op<mnemonic, traits # [Pure, ElementwiseMappable]>,
  Arguments<(ins PointLike:$lhs, PointLike:$rhs)>,
  Results<(outs PointLike:$output)> {
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)";
}

////////////// POINT INITIALIZATIONS //////////////

def EllipticCurve_PointOp : EllipticCurve_Op<"point", [TypesMatchWith<
  "result type has point-like element type and same shape as operands",
  "output", "coords", "SmallVector<Type, 2>(getNumCoordsFromPointLike($_self), getCurveFromPointLike($_self).getBaseField())">]> {
  let summary = "An elliptic curve point";
  let description = [{
    Defines an elliptic curve point in Affine(x, y), Jacobian(x, y, z), or XYZZ(x, y, z², z³) form.

    Example:
    ```
    %0 = elliptic_curve.point %x, %y : elliptic_curve.affine<#curve>
    %1 = elliptic_curve.point %x, %y, %z : elliptic_curve.jacobian<#curve>
    %2 = elliptic_curve.point %x, %y, %zz, %zzz : elliptic_curve.xyzz<#curve>
    ```
  }];
  let arguments = (ins Variadic<Field_AnyFieldType>:$coords);
  let results = (outs PointLike:$output);
  let assemblyFormat = "$coords attr-dict `:` type($output)";
}

////////////// POINT CHECKS //////////////

def EllipticCurve_IsZeroOp : EllipticCurve_Op<"is_zero", [TypesMatchWith<
  "result type has i1 element type and same shape as operands",
  "input", "output", "getI1SameShape($_self)">]> {
  let summary = "Checks whether an elliptic curve point is zero";
  let description = [{
    Outputs a bool true (1) if the elliptic curve point input is zero, and a bool false (0) otherwise.

    Example:
    ```
    %0 = elliptic_curve.is_zero %point : !jacobian
    ```
  }];
  let arguments = (ins PointLike:$input);
  let results = (outs BoolLike:$output);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($input)";
}

/////////// POINT EXTRACTION //////////////

def EllipticCurve_ExtractOp : EllipticCurve_Op<"extract"> {
  let summary = "Extracts coordinates of a point";
  let description = [{
    Outputs coordinates of an elliptic curve point as integers.

    Example:
    ```
    %coords = elliptic_curve.extract %point : elliptic_curve.affine<#curve> -> !PF
    ```
  }];

  let arguments = (ins PointLike:$input);
  let results = (outs Variadic<Field_AnyFieldType>:$output);
  let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";
}

////////////// POINT ARITHMETIC OPERATIONS //////////////

// Elliptic curve point addition.
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

// Elliptic curve point subtraction.
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
def EllipticCurve_NegateOp : EllipticCurve_UnaryOp<"negate", [Involution, SameOperandsAndResultType]> {
  let summary = "Elliptic curve point negation operation";
  let description = [{
    Computes the negation of an elliptic curve point.

    affine -> Jacobian
    Jacobian -> Jacobian
    XYZZ -> XYZZ

    Example:
    ```
    %negation = elliptic_curve.negate %a : !jacobian
    ```
  }];
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

// Elliptic curve point doubling.
def EllipticCurve_DoubleOp : EllipticCurve_UnaryOp<"double"> {
  let summary = "Elliptic curve point doubling operation";
  let description = [{
    Computes the double of an elliptic curve point. If an affine point is doubled,
    it becomes Jacobian. If a Jacobian or XYZZ point is doubled, its type remains the same.

    affine -> Jacobian
    Jacobian -> Jacobian
    XYZZ -> XYZZ

    Example:
    ```
    %doubled = elliptic_curve.double %a : !affine -> !jacobian
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
    %product = elliptic_curve.scalar_mul %f, %affine1 : !PF, !affine -> !jacobian
    ```
  }];
  let arguments = (ins PrimeFieldLike:$scalar, PointLike:$point);
  let results = (outs PointLike:$output);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($scalar) `,` type($point) `->` type($output)";
}

// Elliptic curve point scalar multiplication.
def EllipticCurve_MSMOp : EllipticCurve_Op<"msm", []> {
  let summary = "Elliptic curve multi-scalar multiplication";
  let description = [{
    Computes multi-scalar multiplication (MSM) of elliptic curve points and scalars.

    affine -> Jacobian
    Jacobian -> Jacobian
    XYZZ -> XYZZ

    Example:
    ```
    %msm_result = elliptic_curve.msm %scalars, %points : tensor<3x!PF>, tensor<3x!affine> -> !jacobian
    ```
  }];
  let arguments = (ins RankedTensorOf<[Field_PrimeFieldType]>:$scalars, RankedTensorOf<[PointType]>:$points);
  let results = (outs PointType:$output);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($scalars) `,` type($points) `->` type($output)";
}

////////////// POINT CONVERSIONS //////////////

def EllipticCurve_ConvertPointTypeOp : EllipticCurve_UnaryOp<"convert_point_type"> {
  let summary = "Convert a point from one form to another.";
  let description = [{
    Converts a point on an elliptic curve from one form to another.
    Possible forms are Affine, Jacobian, and XYZZ.

    Example:
    ```
    !affine = elliptic_curve.affine<#sw>
    !jacobian = elliptic_curve.jacobian<#sw>

    %converted = elliptic_curve.convert_point_type %0 : !affine -> !jacobian
    ```
  }];
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";
}

#endif  // ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEOPS_TD_
