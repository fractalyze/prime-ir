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
    auto res =
        llvm::TypeSwitch<Type, AliasResult>(type)
            .Case<AffineType>([&](auto &point) {
              os << "affine_curve";
              os << "-a" << point.getCurve().getA().getValue().getValue();
              os << "-b" << point.getCurve().getB().getValue().getValue();
              os << "-gx" << point.getCurve().getGx().getValue().getValue();
              os << "-gy" << point.getCurve().getGy().getValue().getValue();
              os << "_pf"
                 << point.getCurve().getA().getType().getModulus().getValue();
              return AliasResult::FinalAlias;
            })
            .Case<JacobianType>([&](auto &point) {
              os << "jacobian_curve";
              os << "-a" << point.getCurve().getA().getValue().getValue();
              os << "-b" << point.getCurve().getB().getValue().getValue();
              os << "-gx" << point.getCurve().getGx().getValue().getValue();
              os << "-gy" << point.getCurve().getGy().getValue().getValue();
              os << "_pf"
                 << point.getCurve().getA().getType().getModulus().getValue();
              return AliasResult::FinalAlias;
            })
            .Case<XYZZType>([&](auto &point) {
              os << "xyzz_curve";
              os << "-a" << point.getCurve().getA().getValue().getValue();
              os << "-b" << point.getCurve().getB().getValue().getValue();
              os << "-gx" << point.getCurve().getGx().getValue().getValue();
              os << "-gy" << point.getCurve().getGy().getValue().getValue();
              os << "_pf"
                 << point.getCurve().getA().getType().getModulus().getValue();
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

//////////// POINT INITIALIZATION ////////////

// NOTE(ashjeong): internally "z" is a stand-in for "zz" for XYZZ
ParseResult PointOp::parse(OpAsmParser &parser, OperationState &result) {
  APInt x, y, z, zzz;
  Type outputType;
  field::PrimeFieldType pfType;
  llvm::SMLoc loc;

  if (failed(parser.parseInteger(x)) || failed(parser.parseComma()) ||
      failed(parser.parseInteger(y)))
    return failure();

  auto parseXY = [&x, &y, &result](field::PrimeFieldType pfType,
                                   auto outputBitWidth) {
    x = x.zextOrTrunc(outputBitWidth);
    y = y.zextOrTrunc(outputBitWidth);
    result.addAttribute("x", field::PrimeFieldAttr::get(pfType, x));
    result.addAttribute("y", field::PrimeFieldAttr::get(pfType, y));
  };

  if (failed(parser.parseOptionalComma())) {
    // x, y = affine
    loc = parser.getCurrentLocation();
    if (failed(parser.parseColonType(outputType))) return failure();
    auto affine = dyn_cast<AffineType>(outputType);
    if (!affine) {
      return parser.emitError(loc, "type needs more than 2 values");
    }
    result.addTypes(affine);
    pfType = affine.getCurve().getA().getType();

    parseXY(pfType, pfType.getModulus().getValue().getBitWidth());
    return success();
  }

  if (failed(parser.parseInteger(z))) return failure();

  if (failed(parser.parseOptionalComma())) {
    // x, y, z = Jacobian
    loc = parser.getCurrentLocation();
    if (failed(parser.parseColonType(outputType))) return failure();
    auto jacobian = dyn_cast<JacobianType>(outputType);
    if (!jacobian) {
      return parser.emitError(loc, "type has wrong number of input values");
    }
    result.addTypes(jacobian);
    pfType = jacobian.getCurve().getA().getType();
    auto outputBitWidth = pfType.getModulus().getValue().getBitWidth();

    parseXY(pfType, outputBitWidth);
    z = z.zextOrTrunc(outputBitWidth);
    result.addAttribute("z", field::PrimeFieldAttr::get(pfType, z));
    return success();
  }

  // x, y, zz, zzz = XYZZ
  if (failed(parser.parseInteger(zzz))) return failure();
  loc = parser.getCurrentLocation();
  if (failed(parser.parseColonType(outputType))) return failure();

  auto xyzz = dyn_cast<XYZZType>(outputType);
  if (!xyzz) {
    return parser.emitError(loc, "type must take less than 4 values");
  }
  result.addTypes(xyzz);
  pfType = xyzz.getCurve().getA().getType();
  auto outputBitWidth = pfType.getModulus().getValue().getBitWidth();

  parseXY(pfType, outputBitWidth);
  z = z.zextOrTrunc(outputBitWidth);
  zzz = zzz.zextOrTrunc(outputBitWidth);
  result.addAttribute("zz", field::PrimeFieldAttr::get(pfType, z));
  result.addAttribute("zzz", field::PrimeFieldAttr::get(pfType, zzz));
  return success();
}

void PointOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getX());
  p << ", ";
  p.printAttributeWithoutType(getY());
  if (isa<JacobianType>(getOutput().getType())) {
    p << ", ";
    p.printAttributeWithoutType(getZ().value());
  } else if (isa<XYZZType>(getOutput().getType())) {
    p << ", ";
    p.printAttributeWithoutType(getZz().value());
    p << ", ";
    p.printAttributeWithoutType(getZzz().value());
  }
  p << " : ";
  p.printType(getOutput().getType());
}

/////////////// VERIFY OPS /////////////////

template <typename OpType>
LogicalResult verifyBinaryOp(OpType op) {
  auto lhsType = op.getLhs().getType();
  auto rhsType = op.getRhs().getType();
  auto outputType = op.getOutput().getType();
  if (dyn_cast<AffineType>(lhsType) || dyn_cast<AffineType>(rhsType)) {
    if (lhsType == rhsType && dyn_cast<JacobianType>(outputType)) {
      // affine, affine -> Jacobian
      return success();
    } else if (!dyn_cast<AffineType>(outputType) &&
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

template <typename OpType>
LogicalResult verifyUnaryOp(OpType op) {
  auto inputType = op.getInput().getType();
  auto outputType = op.getOutput().getType();
  if ((dyn_cast<AffineType>(inputType) && dyn_cast<JacobianType>(outputType)) ||
      inputType == outputType)
    return success();
  // TODO(ashjeong): check curves/fields are the same
  return op->emitError() << "wrong output type given input type";
}

LogicalResult DblOp::verify() { return verifyUnaryOp(*this); }

LogicalResult ScalarMulOp::verify() { return verifyUnaryOp(*this); }

/////////////// VERIFY POINT CONVERSIONS ////////////////

LogicalResult ConvertPointTypeOp::verify() {
  auto inputType = getInput().getType();
  auto outputType = getOutput().getType();
  if (inputType == outputType) return emitError() << "Converting on same types";
  // TODO(ashjeong): check curves are the same
  return success();
}

}  // namespace mlir::zkir::elliptic_curve
