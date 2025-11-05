#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"

#include <cassert>

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"

// IWYU pragma: begin_keep
// Headers needed for EllipticCurveAttributes.cpp.inc
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "zkir/Dialect/Field/IR/FieldAttributes.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"
// Headers needed for EllipticCurveOps.cpp.inc
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"
#include "zkir/Utils/OpUtils.h"
// IWYU pragma: end_keep

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

// WARNING: Assumes Jacobian or XYZZ point types
Value createZeroPoint(ImplicitLocOpBuilder &b, Type pointType) {
  auto baseFieldType = getCurveFromPointLike(pointType).getBaseField();
  auto zeroBF = b.create<field::ConstantOp>(baseFieldType, 0);
  Value oneBF = field::isMontgomery(baseFieldType)
                    ? b.create<field::ToMontOp>(
                           baseFieldType,
                           b.create<field::ConstantOp>(
                               field::getStandardFormType(baseFieldType), 1))
                          .getResult()
                    : b.create<field::ConstantOp>(baseFieldType, 1);
  return isa<XYZZType>(pointType)
             ? b.create<PointOp>(pointType,
                                 ValueRange{oneBF, oneBF, zeroBF, zeroBF})
             : b.create<PointOp>(pointType, ValueRange{oneBF, oneBF, zeroBF});
}

} // namespace mlir::zkir::elliptic_curve

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
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.cpp.inc" // NOLINT(build/include)
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.cpp.inc" // NOLINT(build/include)
      >();
  addOperations<
#define GET_OP_LIST
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.cpp.inc" // NOLINT(build/include)
      >();

  addInterface<EllipticCurveOpAsmDialectInterface>();
}

} // namespace mlir::zkir::elliptic_curve
