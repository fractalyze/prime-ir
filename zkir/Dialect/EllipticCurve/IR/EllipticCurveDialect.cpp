#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"

#include <cassert>
#include <optional>

#include "llvm/include/llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "zkir/Dialect/Field/IR/FieldAttributes.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::zkir::elliptic_curve {

size_t getNumCoordsFromPointLike(Type pointLike) {
  Type pointType = getElementTypeOrSelf(pointLike);
  if (isa<AffineType>(pointType)) {
    return 2;
  } else if (isa<JacobianType>(pointType)) {
    return 3;
  } else if (isa<XYZZType>(pointType)) {
    return 4;
  } else {
    llvm_unreachable("Unsupported point-like type for curve extraction");
    return 0;
  }
}

ShortWeierstrassAttr getCurveFromPointLike(Type pointLike) {
  Type pointType = getElementTypeOrSelf(pointLike);
  if (auto affineType = dyn_cast<AffineType>(pointType)) {
    return affineType.getCurve();
  } else if (auto jacobianType = dyn_cast<JacobianType>(pointType)) {
    return jacobianType.getCurve();
  } else if (auto xyzzType = dyn_cast<XYZZType>(pointType)) {
    return xyzzType.getCurve();
  } else {
    llvm_unreachable("Unsupported point-like type for curve extraction");
    return ShortWeierstrassAttr();
  }
}

}  // namespace mlir::zkir::elliptic_curve

// Generated definitions
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.cpp.inc"

#define GET_OP_CLASSES
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.cpp.inc"

namespace mlir::zkir::elliptic_curve {

class EllipticCurveOpAsmDialectInterface : public OpAsmDialectInterface {
 public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  // ex. !affine_curve-a3-b2-gx4-gy5_pf7_
  AliasResult getAlias(Type type, raw_ostream &os) const override {
    auto res = llvm::TypeSwitch<Type, AliasResult>(type)
                   .Case<AffineType>([&](auto &point) {
                     os << "affine_curve";
                     return AliasResult::FinalAlias;
                   })
                   .Case<JacobianType>([&](auto &point) {
                     os << "jacobian_curve";
                     return AliasResult::FinalAlias;
                   })
                   .Case<XYZZType>([&](auto &point) {
                     os << "xyzz_curve";
                     return AliasResult::FinalAlias;
                   })
                   .Default([&](Type) { return AliasResult::NoAlias; });
    return res;
  }
};

void EllipticCurveDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.cpp.inc"  // NOLINT(build/include)
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.cpp.inc"  // NOLINT(build/include)
      >();
  addOperations<
#define GET_OP_LIST
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.cpp.inc"  // NOLINT(build/include)
      >();

  addInterface<EllipticCurveOpAsmDialectInterface>();
}

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
    if (lhsType == rhsType && isa<JacobianType>(outputType)) {
      // affine, affine -> Jacobian
      return success();
    } else if (!isa<AffineType>(outputType) &&
               (lhsType == outputType || rhsType == outputType)) {
      // affine, Jacobian -> Jacobian
      // affine, XYZZ -> XYZZ
      return success();
    }
  } else if (lhsType == rhsType && rhsType == outputType) {
    // Jacobian, Jacobian -> Jacobian
    // XYZZ, XYZZ -> XYZZ
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
  if ((isa<AffineType>(inputType) && isa<JacobianType>(outputType)) ||
      inputType == outputType)
    return success();
  // TODO(ashjeong): check curves/fields are the same
  return emitError() << "wrong output type given input type";
}

LogicalResult ScalarMulOp::verify() {
  Type pointType = getPoint().getType();
  Type outputType = getOutput().getType();
  if ((isa<AffineType>(pointType) && isa<JacobianType>(outputType)) ||
      pointType == outputType)
    return success();
  // TODO(ashjeong): check curves/fields are the same
  return emitError() << "wrong output type given point type";
}

LogicalResult MSMOp::verify() {
  TensorType scalarsType = cast<TensorType>(getScalars().getType());
  TensorType pointsType = cast<TensorType>(getPoints().getType());
  Type pointType = pointsType.getElementType();
  if (scalarsType.getRank() != pointsType.getRank()) {
    return emitError() << "scalars and points must have the same rank";
  }
  Type outputType = getOutput().getType();
  if ((isa<AffineType>(pointType) || isa<JacobianType>(pointType)) &&
      !isa<JacobianType>(outputType)) {
    return emitError() << "affine or jacobian point inputs for msm must result "
                          "in jacobian output";
  } else if (isa<XYZZType>(pointType) && !isa<XYZZType>(outputType)) {
    return emitError()
           << "xyzz point inputs for msm must result in xyzz output";
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
