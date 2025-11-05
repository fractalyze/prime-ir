#ifndef ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEOPS_TD_
#define ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEOPS_TD_

include "zkir/Dialect/EllipticCurve/IR/EllipticCurveDialect.td"
include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.td"

include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/LLVMIR/LLVMTypes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "zkir/Dialect/Field/IR/FieldAttributes.td"
include "zkir/Dialect/Field/IR/FieldTypes.td"
include "zkir/Dialect/ModArith/IR/ModArithTypes.td"
include "zkir/Utils/CommonTraits.td"
include "zkir/Utils/CommonTypeConstraints.td"

// Base class for elliptic curve operations.
class EllipticCurve_Op<string mnemonic, list<Trait> traits = []> :
  Op<EllipticCurve_Dialect, mnemonic, traits # [Pure]> {
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

def EllipticCurve_PointOp : EllipticCurve_Op<"point"> {
  let summary = "An elliptic curve point";
  let description = [{
    Defines an elliptic curve point in affine(x, y), jacobian(x, y, z), or xyzz(x, y, z², z³) form.

    Example:
    ```
    %0 = elliptic_curve.point %x, %y : elliptic_curve.affine<#curve>
    %1 = elliptic_curve.point %x, %y, %z : elliptic_curve.jacobian<#curve>
    %2 = elliptic_curve.point %x, %y, %zz, %zzz : elliptic_curve.xyzz<#curve>
    ```
  }];
  let arguments = (ins Variadic<AnyTypeOf<[Field_AnyFieldType, ModArith_ModArithType, LLVMStructType, AnySignlessInteger]>>:$coords);
  let results = (outs PointLike:$output);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
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
  let results = (outs Variadic<AnyTypeOf<[Field_AnyFieldType, ModArith_ModArithType, LLVMStructType, AnySignlessInteger]>>:$output);
  let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
}

////////////// POINT ARITHMETIC OPERATIONS //////////////

// Elliptic curve point addition.
def EllipticCurve_AddOp : EllipticCurve_BinaryOp<"add", [Commutative]> {
  let summary = "Elliptic curve point addition operation";
  let description = [{
    Computes the sum of two elliptic curve points.

    affine, affine -> jacobian
    affine, affine -> xyzz
    affine, jacobian -> jacobian
    affine, xyzz -> xyzz
    jacobian, jacobian -> jacobian
    xyzz, xyzz -> xyzz

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

    affine, affine -> jacobian
    affine, affine -> xyzz
    affine, jacobian -> jacobian
    affine, xyzz -> xyzz
    jacobian, jacobian -> jacobian
    xyzz, xyzz -> xyzz

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

    affine -> affine
    jacobian -> jacobian
    xyzz -> xyzz

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
    it becomes jacobian. If a jacobian or xyzz point is doubled, its type remains the same.

    affine -> jacobian
    affine -> xyzz
    jacobian -> jacobian
    xyzz -> xyzz

    Example:
    ```
    %doubled = elliptic_curve.double %a : !affine -> !jacobian
    ```
  }];
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";
}

////////////// MSM FUNCTIONS //////////////

// Elliptic curve point scalar multiplication.
def EllipticCurve_ScalarMulOp : EllipticCurve_Op<"scalar_mul", [ElementwiseMappable]> {
  let summary = "Elliptic curve point scalar multiplication";
  let description = [{
    Computes the product of an elliptic curve point and a scalar prime field value.

    affine -> jacobian
    affine -> xyzz
    jacobian -> jacobian
    xyzz -> xyzz

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

// Elliptic curve point comparison.
def EllipticCurve_CmpOp : EllipticCurve_Op<"cmp", [SameTypeOperands, Pure, ElementwiseMappable,
  TypesMatchWith<"result type has i1 element type and same shape as operands", "lhs", "result", "getI1SameShape($_self)">]> {
  let summary = "Elliptic curve point comparison operation";
  let description = [{
    **From description of `Arith_CmpIOp`**
    Its first argument is an attribute that defines which type of comparison is
    performed. The following comparisons are supported:

    -   equal (mnemonic: `"eq"`; integer value: `0`)
    -   not equal (mnemonic: `"ne"`; integer value: `1`)

    The result is `1` if the comparison is true and `0` otherwise. For vector or
    tensor operands, the comparison is performed elementwise and the element of
    the result indicates whether the comparison is true for the operand elements
    with the same indices as those of the result.

    Example:
    ```
    !affine = elliptic_curve.affine<#sw>

    %equal = elliptic_curve.cmp eq, %a, %b : !affine
    ```
  }];
  let arguments = (ins Arith_CmpIPredicateAttr:$predicate,
                       PointLike:$lhs,
                       PointLike:$rhs);
  let results = (outs BoolLike:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}

// Elliptic curve point scalar multiplication.
def EllipticCurve_MSMOp : EllipticCurve_Op<"msm", []> {
  let summary = "Elliptic curve multi-scalar multiplication";
  let description = [{
    Computes multi-scalar multiplication (MSM) of elliptic curve points and scalars.

    Output type determines internal computation type of buckets and window sums.
    xyzz is recommended for best performance.

    If both scalars and points have a static shape, the size of scalars and points must be equal.
    Otherwise, both must have a dynamic shape.

    Additionally, 2^{degree} must be greater than or equal to the size of scalars
    if both scalars and points have a static shape.

    window_bits is the number of bits to use for the window. If window_bits is 0, window_bits is computed from the degree.

    When parallel is enabled, window-based parallelization is applied. If term-based parallelization is needed,
    parallel should be disabled and the caller should implement it.

    affine -> affine, jacobian, xyzz
    jacobian -> jacobian, xyzz
    xyzz -> xyzz

    Example:

    If you want to run window-based parallelization, use the parallel attribute as follows:

    ```
    %msm_result = elliptic_curve.msm %scalars, %points degree=2 parallel : tensor<3x!PF>, tensor<3x!affine> -> !xyzz
    ```

    If you want to run serial execution, simply omit the parallel attribute:

    ```
    %msm_result = elliptic_curve.msm %scalars, %points degree=2 : tensor<3x!PF>, tensor<3x!affine> -> !xyzz
    ```

    If you want to configure the MSM with a specific window bits, use the window_bits attribute as follows:

    ```
    %msm_result = elliptic_curve.msm %scalars, %points degree=2 window_bits=4 : tensor<3x!PF>, tensor<3x!affine> -> !xyzz
    ```
  }];
  let arguments = (ins
    RankedTensorOf<[Field_PrimeFieldType]>:$scalars,
    RankedTensorOf<[PointType]>:$points,
    I32Attr:$degree,
    DefaultValuedAttr<I32Attr, "0">:$windowBits,
    UnitAttr:$parallel
  );
  let results = (outs PointType:$output);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `degree` `=` $degree (`window_bits` `=` $windowBits^)? (`parallel` $parallel^)? `:` type($scalars) `,` type($points) `->` type($output)";
}

def EllipticCurve_ScalarDecompOp : EllipticCurve_Op<"scalar_decomp", [
  TypesMatchWith<
    "bucket_indices and point_indices must have the same size",
    "bucketIndices", "pointIndices", "$_self">,
  VarPositiveAndBounded<"getBitsPerWindow()", "cast<field::PrimeFieldType>(getScalars().getType().getElementType()).getStorageBitWidth()">]> {
  let summary = "Decompose scalars for MSM bucket computation";
  let description = [{
    Decomposes scalars in preparation for multi-scalar multiplication (MSM) bucket accumulation.
    This operation takes the scalars tensor and bitsPerWindow, then produces two tensors of the same size:
    - bucket_indices: tensor of unique bucket indices for each scalar-window combination
    - point_indices: tensor of point indices that is to be stored at the corresponding bucket

    A unique bucket index is calculated like so:
    (current window #) << bitsPerWindow | (split part of scalar for given window)

    When the maximum value of the scalars is known to have a smaller bit width than the scalar field modulus's,
    the maximum bit width should be specified with scalarMaxBits. This can reduce the number of unnecessary loops
    that calculate a zero bucket. By default, the bit size of the scalar modulus will be used.

    Example with bitsPerWindow=2 and scalarMaxBits=4 (MSB -> LSB):
    - Scalar 3 (0011): scalar per window = [11, 00] -> bucket indices [3, 0] -> point indices [0, 0]
    - Scalar 5 (0101): scalar per window = [01, 01] -> bucket indices [1, 5] -> point indices [1, 1]
    - Scalar 7 (0111): scalar per window = [11, 01] -> bucket indices [3, 5] -> point indices [2, 2]
    - Total: bucket_indices [3, 0, 1, 5, 3, 5], point_indices [0, 0, 1, 1, 2, 2]

    Example:
    ```
    %bucket_indices, %point_indices = elliptic_curve.scalar_decomp %scalars {bitsPerWindow = 2 : i16, scalarMaxBits = 4 : i16} : (tensor<3x!SF>) -> (tensor<6xindex>, tensor<6xindex>)
    ```
  }];
  let arguments = (ins
    1DTensorOf<[Field_PrimeFieldType]>:$scalars,
    I16Attr:$bitsPerWindow,
    OptionalAttr<I16Attr>:$scalarMaxBits
  );
  let results = (outs
    1DTensorOf<[Index]>:$bucketIndices,
    1DTensorOf<[Index]>:$pointIndices
  );
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Elliptic curve MSM bucket accumulation for GPU.
// Note that parameters differ from the bucket accumulation inside the other MSMOp.
def EllipticCurve_BucketAccOp : EllipticCurve_Op<"bucket_acc", []> {
  let summary = "MSM Bucket Accumulation";
  let description = [{
    Computes the accumulation of points for each bucket given a set of points, sorted point indices, sorted unique bucket indices, and bucket offsets.

    The sorted point indices are the indices of the points in the original set sorted by unique bucket indices and bucket sizes.
    The sorted unique bucket indices are unique indices of the buckets in the original set in ascending order.
    The bucket offsets are the starting and ending indices of the points for each bucket.

    Returns a tensor of accumulated points for each bucket. All buckets are in a 1D tensor.

    Example:
    ```
    %bucket_results = elliptic_curve.bucket_acc %points, %sorted_point_indices, %sorted_unique_bucket_indices, %bucket_offsets: (tensor<3x!affine>, tensor<6xindex>, tensor<4xindex>, tensor<5xindex>) -> tensor<8x!jacobian>
    ```

    Allowed input/output point type combinations:

    points    |   return type
    --------------------------
    affine    |   jacobian, xyzz
    jacobian  |   jacobian, xyzz
    xyzz      |   xyzz
  }];
  let arguments = (ins
    1DTensorOf<[PointType]>:$points,
    1DTensorOf<[Index]>:$sortedPointIndices,
    1DTensorOf<[Index]>:$sortedUniqueBucketIndices,
    1DTensorOf<[Index]>:$bucketOffsets
  );
  let results = (outs 1DTensorOf<[PointType]>:$bucketResults);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Elliptic curve MSM bucket reduction.
def EllipticCurve_BucketReduceOp : EllipticCurve_Op<"bucket_reduce", [SameOperandsAndResultElementType]> {
  let summary = "MSM Bucket Reduction";
  let description = [{
    Computes the reduction of buckets into windows given all buckets and the scalar field type as an attribute.

    Assumes buckets point inputs will be the same type as the output point type.
    Buckets are assumed to be in a 2D tensor <#windows x #bucketsPerWindow x pointType>.

    Returns a 1D windows tensor of accumulated buckets for each window.

    Example:
    ```
    %windows = elliptic_curve.bucket_reduce %buckets {scalarType = !SF}: (tensor<4x4x!jacobian>) -> tensor<4x!jacobian>
    ```
  }];
  let arguments = (ins
    2DTensorOf<[PointType]>:$buckets,
    TypeAttrOf<Field_PrimeFieldType>: $scalarType
  );
  let results = (outs 1DTensorOf<[PointType]>:$windows);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Elliptic curve MSM window reduction.
def EllipticCurve_WindowReduceOp : EllipticCurve_Op<"window_reduce", [TypesMatchWith<
  "output type must be the same as the element type of the windows",
  "windows", "output", "::mlir::getElementTypeOrSelf($_self)">,
  VarPositiveAndBounded<"getBitsPerWindow()", "getScalarType().getStorageBitWidth()">]> {
  let summary = "MSM Window Reduction";
  let description = [{
    Computes the weighted reduction of windows into a point given all windows, the number of bits per window, and the scalar field type.

    Assumes windows point inputs will be the same type as the output point type.
    Windows are assumed to be in a 1D tensor <#windows x pointType>.

    Returns a singular point.

    Example:
    ```
    %result = elliptic_curve.window_reduce %windows {bitsPerWindow = 2 : i16, scalarType = !SF}: (tensor<4x!jacobian>) -> !jacobian
    ```
  }];
  let arguments = (ins
    Static1DTensorOf<[PointType]>:$windows,
    I16Attr: $bitsPerWindow,
    TypeAttrOf<Field_PrimeFieldType>: $scalarType
  );
  let results = (outs PointType:$output);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

////////////// POINT CONVERSIONS //////////////

def EllipticCurve_ConvertPointTypeOp : EllipticCurve_UnaryOp<"convert_point_type"> {
  let summary = "Convert a point from one form to another.";
  let description = [{
    Converts a point on an elliptic curve from one form to another.
    Possible forms are affine, jacobian, and xyzz.

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
