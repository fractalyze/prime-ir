#include "zkir/Dialect/Field/IR/FieldDialect.h"

#include <cassert>
#include <optional>

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "zkir/Dialect/Field/IR/FieldAttributes.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"

// Generated definitions
#include "zkir/Dialect/Field/IR/FieldDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "zkir/Dialect/Field/IR/FieldAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "zkir/Dialect/Field/IR/FieldTypes.cpp.inc"

#define GET_OP_CLASSES
#include "zkir/Dialect/Field/IR/FieldOps.cpp.inc"

namespace mlir::zkir::field {

class FieldOpAsmDialectInterface : public OpAsmDialectInterface {
 public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Type type, raw_ostream &os) const override {
    auto res = llvm::TypeSwitch<Type, AliasResult>(type)
                   .Case<PrimeFieldType>([&](auto &pfElemType) {
                     os << "pf";
                     os << pfElemType.getModulus().getValue();
                     return AliasResult::FinalAlias;
                   })
                   .Default([&](Type) { return AliasResult::NoAlias; });
    return res;
  }
};

void FieldDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "zkir/Dialect/Field/IR/FieldTypes.cpp.inc"  // NOLINT(build/include)
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "zkir/Dialect/Field/IR/FieldAttributes.cpp.inc"  // NOLINT(build/include)
      >();
  addOperations<
#define GET_OP_LIST
#include "zkir/Dialect/Field/IR/FieldOps.cpp.inc"  // NOLINT(build/include)
      >();

  addInterface<FieldOpAsmDialectInterface>();
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<APInt> parsedInt;
  Type parsedType;

  if (failed(parser.parseCommaSeparatedList(
          [&]() { return parser.parseInteger(parsedInt.emplace_back()); })) ||
      failed(parser.parseColonType(parsedType)))
    return failure();

  if (auto pfType = dyn_cast<PrimeFieldType>(parsedType)) {
    if (parsedInt.size() != 1) {
      parser.emitError(parser.getCurrentLocation(),
                       "prime field constant must have exactly one value");
      return failure();
    }

    auto modulus = pfType.getModulus().getValue();

    // TODO(batzor): Check if the modulus is a prime number.
    if (modulus.isNegative() || modulus.isZero()) {
      parser.emitError(parser.getCurrentLocation(), "modulus must be positive");
      return failure();
    }

    auto outputBitWidth = pfType.getModulus().getValue().getBitWidth();
    if (parsedInt[0].getActiveBits() > outputBitWidth)
      return parser.emitError(
          parser.getCurrentLocation(),
          "constant value is too large for the underlying type");

    // zero-extend or truncate to the correct bitwidth
    parsedInt[0] = parsedInt[0].zextOrTrunc(outputBitWidth);
    result.addAttribute("value", PrimeFieldAttr::get(pfType, parsedInt[0]));
    result.addTypes(pfType);
    return success();
  } else if (auto f2Type = dyn_cast<QuadraticExtFieldType>(parsedType)) {
    if (parsedInt.size() != 2) {
      parser.emitError(parser.getCurrentLocation(),
                       "quadratic extension field constant must have exactly "
                       "two values");
      return failure();
    }

    auto modulus = f2Type.getBaseField().getModulus().getValue();

    // TODO(batzor): Check if the modulus is a prime number.
    if (modulus.isNegative() || modulus.isZero()) {
      parser.emitError(parser.getCurrentLocation(), "modulus must be positive");
      return failure();
    }

    auto outputBitWidth =
        f2Type.getBaseField().getModulus().getValue().getBitWidth();
    for (const auto &value : parsedInt) {
      if (value.getActiveBits() > outputBitWidth)
        return parser.emitError(
            parser.getCurrentLocation(),
            "constant value is too large for the underlying type");
    }

    // zero-extend or truncate to the correct bitwidth
    parsedInt[0] = parsedInt[0].zextOrTrunc(outputBitWidth);
    parsedInt[1] = parsedInt[1].zextOrTrunc(outputBitWidth);
    result.addAttribute(
        "value", QuadraticExtFieldAttr::get(
                     parser.getContext(), f2Type,
                     PrimeFieldAttr::get(f2Type.getBaseField(), parsedInt[0]),
                     PrimeFieldAttr::get(f2Type.getBaseField(), parsedInt[1])));
    result.addTypes(f2Type);
    return success();
  }
  parser.emitError(parser.getCurrentLocation(),
                   "invalid constant type: expected prime or quadratic");
  return failure();
}

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getValue());
  p << " : ";
  p.printType(getOutput().getType());
}

template <typename OpType>
static LogicalResult disallowTensorOfExtField(OpType op) {
  // FIXME(batzor): In the prime field case, we rely on elementwise trait but in
  // the quadratic extension case, `linalg.generic` introduced by the
  // elementwise pass will be ill-formed due to the 1:N conversion.
  auto resultType = op.getResult().getType();
  if (isa<ShapedType>(resultType)) {
    auto elementType = cast<ShapedType>(resultType).getElementType();
    if (isa<QuadraticExtFieldType>(elementType)) {
      return op->emitOpError(
          "tensor operation is not supported for quadratic "
          "extension field type");
    }
  }
  return success();
}

LogicalResult NegateOp::verify() { return disallowTensorOfExtField(*this); }
LogicalResult AddOp::verify() { return disallowTensorOfExtField(*this); }
LogicalResult SubOp::verify() { return disallowTensorOfExtField(*this); }
LogicalResult MulOp::verify() { return disallowTensorOfExtField(*this); }
LogicalResult InverseOp::verify() { return disallowTensorOfExtField(*this); }
LogicalResult FromMontOp::verify() {
  Type resultType = getElementTypeOrSelf(this->getOutput().getType());
  bool isMontgomery = true;
  if (auto pfType = dyn_cast<PrimeFieldType>(resultType)) {
    isMontgomery = pfType.isMontgomery();
  } else if (auto f2Type = dyn_cast<QuadraticExtFieldType>(resultType)) {
    isMontgomery = f2Type.isMontgomery();
  }
  if (isMontgomery) {
    return emitOpError()
           << "FromMontOp result should be a standard type, but got "
           << resultType << ".";
  }
  return disallowTensorOfExtField(*this);
}
LogicalResult ToMontOp::verify() {
  Type resultType = getElementTypeOrSelf(this->getOutput().getType());
  bool isMontgomery = false;
  if (auto pfType = dyn_cast<PrimeFieldType>(resultType)) {
    isMontgomery = pfType.isMontgomery();
  } else if (auto f2Type = dyn_cast<QuadraticExtFieldType>(resultType)) {
    isMontgomery = f2Type.isMontgomery();
  }
  if (!isMontgomery) {
    return emitOpError()
           << "ToMontOp result should be a Montgomery type, but got "
           << resultType << ".";
  }
  return disallowTensorOfExtField(*this);
}

}  // namespace mlir::zkir::field
