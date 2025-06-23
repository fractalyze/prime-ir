#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"

namespace mlir::zkir::elliptic_curve {

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
  Type outputType = getOutput().getType();
  if (isa<AffineType>(outputType)) {
    return emitError() << "output type cannot be affine";
  } else {
    return success();
  }
  // TODO(ashjeong): check curves/fields are the same
}

//////////////// VERIFY POINT CONVERSIONS ////////////////

LogicalResult ConvertPointTypeOp::verify() {
  Type inputType = getInput().getType();
  Type outputType = getOutput().getType();
  if (inputType == outputType) return emitError() << "Converting on same types";
  // TODO(ashjeong): check curves are the same
  return success();
}

}  // namespace mlir::zkir::elliptic_curve
