#include "zkir/Dialect/Field/IR/FieldDialect.h"

#include <cassert>
#include <optional>

#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"       // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
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

namespace mlir {
namespace zkir {
namespace field {

class FieldOpAsmDialectInterface : public OpAsmDialectInterface {
 public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Type type, raw_ostream &os) const override {
    auto res = llvm::TypeSwitch<Type, AliasResult>(type)
                   .Case<PrimeFieldType>([&](auto &pfElemType) {
                     os << "PF";
                     os << "_";
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
  APInt parsedValue(0, 0, /*isSigned=*/true);
  Type parsedType;

  if (failed(parser.parseInteger(parsedValue))) {
    parser.emitError(parser.getCurrentLocation(),
                     "found invalid integer value");
    return failure();
  }

  if (parsedValue.isNegative()) {
    parser.emitError(parser.getCurrentLocation(),
                     "negative value is not allowed");
    return failure();
  }

  if (parser.parseColon() || parser.parseType(parsedType)) return failure();

  auto pfType = dyn_cast<PrimeFieldType>(parsedType);
  if (!pfType) return failure();

  auto modulus = pfType.getModulus().getValue();

  // TODO(batzor): Check if the modulus is a prime number.
  if (modulus.isNegative() || modulus.isZero()) {
    parser.emitError(parser.getCurrentLocation(), "modulus must be positive");
    return failure();
  }

  auto outputBitWidth = pfType.getModulus().getType().getIntOrFloatBitWidth();
  if (parsedValue.getActiveBits() > outputBitWidth)
    return parser.emitError(
        parser.getCurrentLocation(),
        "constant value is too large for the underlying type");

  auto intValue = IntegerAttr::get(pfType.getModulus().getType(),
                                   parsedValue.zextOrTrunc(outputBitWidth));
  result.addAttribute(
      "value", PrimeFieldAttr::get(parser.getContext(), pfType, intValue));
  result.addTypes(pfType);
  return success();
}

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  // getValue chain:
  // op's PrimeFieldAttr value
  //   -> PrimeFieldAttr's IntegerAttr value
  //   -> IntegerAttr's APInt value
  getValue().getValue().getValue().print(p.getStream(), /*isSigned=*/false);
  p << " : ";
  p.printType(getOutput().getType());
}

LogicalResult ConstantOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> loc,
    ConstantOpAdaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  returnTypes.push_back(adaptor.getValue().getType());
  return success();
}

}  // namespace field
}  // namespace zkir
}  // namespace mlir
