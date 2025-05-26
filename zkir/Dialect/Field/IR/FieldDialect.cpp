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
  APInt parsedInt;
  Type parsedType;

  if (failed(parser.parseInteger(parsedInt)) ||
      failed(parser.parseColonType(parsedType)))
    return failure();

  if (parsedInt.isNegative()) {
    parser.emitError(parser.getCurrentLocation(),
                     "negative value is not allowed");
    return failure();
  }

  auto pfType = dyn_cast<PrimeFieldType>(parsedType);
  if (!pfType) {
    parser.emitError(parser.getCurrentLocation(),
                     "type must be of prime field type");
    return failure();
  }

  auto modulus = pfType.getModulus().getValue();

  // TODO(batzor): Check if the modulus is a prime number.
  if (modulus.isNegative() || modulus.isZero()) {
    parser.emitError(parser.getCurrentLocation(), "modulus must be positive");
    return failure();
  }

  auto outputBitWidth = pfType.getModulus().getValue().getBitWidth();
  if (parsedInt.getActiveBits() > outputBitWidth)
    return parser.emitError(
        parser.getCurrentLocation(),
        "constant value is too large for the underlying type");

  // zero-extend or truncate to the correct bitwidth
  parsedInt = parsedInt.zextOrTrunc(outputBitWidth);
  result.addAttribute(
      "value",
      IntegerAttr::get(IntegerType::get(parser.getContext(), outputBitWidth),
                       parsedInt));
  result.addTypes(pfType);
  return success();
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
LogicalResult FromMontOp::verify() { return disallowTensorOfExtField(*this); }
LogicalResult ToMontOp::verify() { return disallowTensorOfExtField(*this); }
LogicalResult MontMulOp::verify() { return disallowTensorOfExtField(*this); }

}  // namespace mlir::zkir::field
