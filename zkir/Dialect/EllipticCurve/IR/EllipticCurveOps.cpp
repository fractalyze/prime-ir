#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"

namespace mlir::zkir::elliptic_curve {
namespace {
template <typename OpType>
LogicalResult verifyMSMPointTypes(OpType op, Type inputType, Type outputType) {
  if (isa<JacobianType>(inputType) && isa<AffineType>(outputType)) {
    return op.emitError()
           << "jacobian input points require a jacobian or xyzz output type";
  } else if (isa<XYZZType>(inputType) &&
             (isa<AffineType>(outputType) || isa<JacobianType>(outputType))) {
    return op.emitError() << "xyzz input points require an xyzz output type";
  }
  return success();
}
} // namespace

/////////////// VERIFY OPS /////////////////

LogicalResult IsZeroOp::verify() {
  Type inputType = getInput().getType();
  if (isa<AffineType>(getElementTypeOrSelf(inputType)) ||
      isa<JacobianType>(getElementTypeOrSelf(inputType)) ||
      isa<XYZZType>(getElementTypeOrSelf(inputType)))
    return success();
  return emitError() << "invalid input type";
}

template <typename OpType>
LogicalResult verifyBinaryOp(OpType op) {
  Type lhsType = op.getLhs().getType();
  Type rhsType = op.getRhs().getType();
  Type outputType = op.getOutput().getType();
  if (isa<AffineType>(lhsType) || isa<AffineType>(rhsType)) {
    if (lhsType == rhsType &&
        (isa<JacobianType>(outputType) || isa<XYZZType>(outputType))) {
      // affine, affine -> jacobian
      // affine, affine -> xyzz
      return success();
    } else if (!isa<AffineType>(outputType) &&
               (lhsType == outputType || rhsType == outputType)) {
      // affine, jacobian -> jacobian
      // affine, xyzz -> xyzz
      return success();
    }
  } else if (lhsType == rhsType && rhsType == outputType) {
    // jacobian, jacobian -> jacobian
    // xyzz, xyzz -> xyzz
    return success();
  }
  // TODO(ashjeong): check the curves of given types are the same
  return op->emitError() << "input or output types are wrong";
}

LogicalResult AddOp::verify() { return verifyBinaryOp(*this); }

LogicalResult SubOp::verify() { return verifyBinaryOp(*this); }

LogicalResult DoubleOp::verify() {
  Type inputType = getInput().getType();
  Type outputType = getOutput().getType();
  if ((isa<AffineType>(inputType) &&
       (isa<JacobianType>(outputType) || isa<XYZZType>(outputType))) ||
      inputType == outputType)
    return success();
  // TODO(ashjeong): check curves/fields are the same
  return emitError() << "wrong output type given input type";
}

LogicalResult ScalarMulOp::verify() {
  Type pointType = getPoint().getType();
  Type outputType = getOutput().getType();
  if ((isa<AffineType>(pointType) &&
       (isa<JacobianType>(outputType) || isa<XYZZType>(outputType))) ||
      pointType == outputType)
    return success();
  // TODO(ashjeong): check curves/fields are the same
  return emitError() << "wrong output type given point type";
}

LogicalResult MSMOp::verify() {
  TensorType scalarsType = cast<TensorType>(getScalars().getType());
  TensorType pointsType = cast<TensorType>(getPoints().getType());
  if (scalarsType.getRank() != pointsType.getRank()) {
    return emitError() << "scalars and points must have the same rank";
  }
  Type inputType = pointsType.getElementType();
  Type outputType = getOutput().getType();
  if (failed(verifyMSMPointTypes(*this, inputType, outputType))) {
    return failure();
  }

  int32_t degree = getDegree();
  if (degree <= 0) {
    return emitError() << "degree must be greater than 0";
  }

  if (scalarsType.hasStaticShape() && pointsType.hasStaticShape()) {
    if (scalarsType.getNumElements() != pointsType.getNumElements()) {
      return emitError()
             << "scalars and points must have the same number of elements";
    }
    if (scalarsType.getNumElements() > (int64_t{1} << degree)) {
      return emitError() << "scalars must have at least 2^degree elements";
    }
  } else if (scalarsType.hasStaticShape() && !pointsType.hasStaticShape()) {
    return emitError() << "scalars has static shape and points does not";
  } else if (!scalarsType.hasStaticShape() && pointsType.hasStaticShape()) {
    return emitError() << "points has static shape and scalars does not";
  }

  int32_t windowBits = getWindowBits();
  if (windowBits < 0) {
    return emitError() << "window bits must be greater than or equal to 0";
  }

  return success();
  // TODO(ashjeong): check curves/fields are the same
}

LogicalResult BucketAccOp::verify() {
  TensorType sortedUniqueBucketIndices =
      cast<TensorType>(getSortedUniqueBucketIndices().getType());
  TensorType bucketOffsets = cast<TensorType>(getBucketOffsets().getType());
  TensorType bucketResults = cast<TensorType>(getBucketResults().getType());
  TensorType points = cast<TensorType>(getPoints().getType());

  if (sortedUniqueBucketIndices.getNumElements() !=
      bucketOffsets.getNumElements() - 1) {
    return emitError() << "bucket_offsets must have one more element than "
                          "sorted_unique_bucket_indices";
  }

  Type inputType = points.getElementType();
  Type outputType = bucketResults.getElementType();
  if (failed(verifyMSMPointTypes(*this, inputType, outputType))) {
    return failure();
  }

  // TODO(ashjeong): check summed result of all bucket sizes equals to
  // points.size()
  // TODO(ashjeong): verify output numElements = number of buckets

  return success();
}

//////////////// VERIFY POINT CONVERSIONS ////////////////

LogicalResult ConvertPointTypeOp::verify() {
  Type inputType = getInput().getType();
  Type outputType = getOutput().getType();
  if (inputType == outputType)
    return emitError() << "Converting on same types";
  // TODO(ashjeong): check curves are the same
  return success();
}

} // namespace mlir::zkir::elliptic_curve
