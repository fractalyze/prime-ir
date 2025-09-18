#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "zkir/Dialect/Field/IR/FieldDialect.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"

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

template <typename OpType>
LogicalResult verifyPointCoordTypes(OpType op, Type pointType, Type coordType) {
  Type ptBaseField = getCurveFromPointLike(pointType).getBaseField();
  if (isa<field::FieldDialect>(coordType.getDialect())) {
    if (ptBaseField != coordType) {
      return op.emitError() << "coord must be base field of point; but got "
                            << coordType << " expected " << ptBaseField;
    }
    return success();
  }
  if (auto pfType = dyn_cast<field::PrimeFieldType>(ptBaseField)) {
    if (auto modArithType = dyn_cast<mod_arith::ModArithType>(coordType)) {
      if (pfType.getModulus() != modArithType.getModulus())
        return op.emitError()
               << "output must have the same modulus as the base "
                  "field of input";
      if (pfType.isMontgomery() != modArithType.isMontgomery())
        return op.emitError()
               << "output must have the same montgomery form as the "
                  "base field of input";
    } else if (auto intType = dyn_cast<IntegerType>(coordType)) {
      if (intType.getWidth() != pfType.getStorageBitWidth())
        return op.emitError()
               << "output must have the same bitwidth as the base "
                  "field of input";
    } else {
      return op.emitError()
             << "output must be a mod_arith type or an integer "
                "type if the base field is a prime field; but got "
             << coordType;
    }
  } else if (auto f2Type =
                 dyn_cast<field::QuadraticExtFieldType>(ptBaseField)) {
    if (auto structType = dyn_cast<LLVM::LLVMStructType>(coordType)) {
      if (structType.getBody().size() != 2)
        return op.emitError() << "output struct must have two elements for "
                                 "quadratic extension field";
      // NOTE: In case of extension fields, the types are not lowered to modular
      // types since struct cannot contain modular types so we can ignore that
      // case.
      if (structType.getBody()[0] != f2Type.getBaseField().getStorageType() ||
          structType.getBody()[1] != f2Type.getBaseField().getStorageType())
        return op.emitError() << "output struct element must have the same "
                                 "bitwidth as the base field of input";
    } else {
      return op.emitError() << "output must be a struct type for quadratic "
                               "extension field; but got "
                            << coordType;
    }
  }
  return success();
}
} // namespace

/////////////// VERIFY OPS /////////////////

LogicalResult PointOp::verify() {
  Type outputType = getType();
  if (getNumCoordsFromPointLike(outputType) != getCoords().size()) {
    return emitError() << outputType << " should have "
                       << getNumCoordsFromPointLike(outputType)
                       << " coordinates";
  }
  return verifyPointCoordTypes(*this, outputType, getCoords()[0].getType());
}

LogicalResult ExtractOp::verify() {
  Type inputType = getInput().getType();
  if (getNumCoordsFromPointLike(inputType) != getOutput().size()) {
    return emitError() << inputType << " should have "
                       << getNumCoordsFromPointLike(inputType)
                       << " coordinates";
  }
  return verifyPointCoordTypes(*this, inputType, getType(0));
}

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
  Type outputType = op.getType();
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
  Type outputType = getType();
  if ((isa<AffineType>(inputType) &&
       (isa<JacobianType>(outputType) || isa<XYZZType>(outputType))) ||
      inputType == outputType)
    return success();
  // TODO(ashjeong): check curves/fields are the same
  return emitError() << "wrong output type given input type";
}

LogicalResult ScalarMulOp::verify() {
  Type pointType = getPoint().getType();
  Type outputType = getType();
  if ((isa<AffineType>(pointType) &&
       (isa<JacobianType>(outputType) || isa<XYZZType>(outputType))) ||
      pointType == outputType)
    return success();
  // TODO(ashjeong): check curves/fields are the same
  return emitError() << "wrong output type given point type";
}

LogicalResult MSMOp::verify() {
  TensorType scalarsType = getScalars().getType();
  TensorType pointsType = getPoints().getType();
  if (scalarsType.getRank() != pointsType.getRank()) {
    return emitError() << "scalars and points must have the same rank";
  }
  Type inputType = pointsType.getElementType();
  Type outputType = getType();
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
      getSortedUniqueBucketIndices().getType();
  TensorType bucketOffsets = getBucketOffsets().getType();
  TensorType bucketResults = getBucketResults().getType();
  TensorType points = getPoints().getType();

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

LogicalResult BucketReduceOp::verify() {
  TensorType bucketsType = getBuckets().getType();
  TensorType windowsType = getWindows().getType();
  if (bucketsType.getShape()[0] != windowsType.getShape()[0]) {
    return emitError() << "dimension 0 of buckets and windows must be the same";
  }
  return success();
}

LogicalResult WindowReduceOp::verify() {
  unsigned scalarBitWidth = getScalarType().getStorageBitWidth();
  int16_t bitsPerWindow = getBitsPerWindow();
  TensorType windowsType = getWindows().getType();

  unsigned numWindows = (scalarBitWidth + bitsPerWindow - 1) / bitsPerWindow;
  if (numWindows != windowsType.getNumElements()) {
    return emitError() << "number of calculated windows (" << numWindows
                       << ") must be the same as the number of windows in the "
                          "windows tensor ("
                       << windowsType.getNumElements() << ")";
  }
  return success();
}

//////////////// VERIFY POINT CONVERSIONS ////////////////

LogicalResult ConvertPointTypeOp::verify() {
  Type inputType = getInput().getType();
  Type outputType = getType();
  if (inputType == outputType)
    return emitError() << "Converting on same types";
  // TODO(ashjeong): check curves are the same
  return success();
}

} // namespace mlir::zkir::elliptic_curve
