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

ParseResult PointOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> opInfo;
  SmallVector<Type> types;
  SMLoc loc = parser.getCurrentLocation();
  OpAsmParser::UnresolvedOperand x, y, z;
  Type inputType, outputType;
  SmallVector<int32_t, 5> segmentSizes = {/*x=*/0, /*y=*/0, /*z=*/0, /*zz=*/0,
                                          /*zzz=*/0};

  if (failed(parser.parseOperand(x)) || failed(parser.parseComma()) ||
      failed(parser.parseOperand(y)))
    return failure();
  // affine
  opInfo = {x, y};
  segmentSizes[0] = 1;  // x
  segmentSizes[1] = 1;  // y
  if (succeeded(parser.parseOptionalComma()) &&
      succeeded(parser.parseOptionalOperand(z).value())) {
    opInfo.push_back(z);
    OpAsmParser::UnresolvedOperand zzz;
    if (succeeded(parser.parseOptionalComma()) &&
        succeeded(parser.parseOptionalOperand(zzz).value())) {
      // xyzz
      opInfo.push_back(zzz);
      segmentSizes[3] = 1;  // zz
      segmentSizes[4] = 1;  // zzz
    } else {
      // jacobian
      segmentSizes[2] = 1;  // z
    }
  }

  if (failed(parser.parseColonType(inputType)) || failed(parser.parseArrow()) ||
      failed(parser.parseType(outputType)))
    return failure();
  for (int i = 0; i < opInfo.size(); ++i) {
    types.push_back(inputType);
  }

  auto segmentSizesAttr = DenseIntElementsAttr::get(
      VectorType::get({static_cast<int64_t>(segmentSizes.size())},
                      parser.getBuilder().getI32Type()),
      segmentSizes);
  result.addAttribute("operand_segment_sizes", segmentSizesAttr);
  result.addTypes(outputType);
  return parser.resolveOperands(opInfo, types, loc, result.operands);
}

void PointOp::print(OpAsmPrinter &p) {
  p << ' ' << getOperands();
  p << " : ";
  p.printType(getCoords()[0].getType());
  p << " -> ";
  p.printType(getOutput().getType());
}

LogicalResult PointOp::verify() {
  Type outputType = getOutput().getType();
  uint8_t numCoords = getNumOperands();
  auto operands = getOperands();
  field::PrimeFieldType baseField =
      cast<field::PrimeFieldType>(operands[0].getType());

  if (isa<AffineType>(outputType) && numCoords != 2) {
    return emitError() << "Wrong number of coordinates for affine type";
  } else if (isa<JacobianType>(outputType) && numCoords != 3) {
    return emitError() << "Wrong number of coordinates for jacobian type";
  } else if (isa<XYZZType>(outputType) && numCoords != 4) {
    return emitError() << "Wrong number of coordinates for xyzz type";
  }

  for (int i = 1; i < operands.size(); ++i) {
    if (baseField != cast<field::PrimeFieldType>(operands[i].getType())) {
      return emitError() << "All coordinates are not of the same prime field";
    }
  }
  // TODO(ashjeong): check curve base field and coords types are the same
  return success();
}

/////////////// VERIFY OPS /////////////////

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

//////////////// VERIFY POINT CONVERSIONS ////////////////

LogicalResult ConvertPointTypeOp::verify() {
  Type inputType = getInput().getType();
  Type outputType = getOutput().getType();
  if (inputType == outputType) return emitError() << "Converting on same types";
  // TODO(ashjeong): check curves are the same
  return success();
}

}  // namespace mlir::zkir::elliptic_curve
