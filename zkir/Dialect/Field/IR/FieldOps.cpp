#include "zkir/Dialect/Field/IR/FieldOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "zkir/Dialect/ModArith/IR/ModArithAttributes.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"
#include "zkir/Utils/APIntUtils.h"

// IWYU pragma: begin_keep
// Headers needed for FieldCanonicalization.cpp.inc
#include "zkir/Dialect/TensorExt/IR/TensorExtOps.h"
// IWYU pragma: end_keep
namespace mlir::zkir::field {

PrimeFieldAttr getAttrAsStandardForm(PrimeFieldAttr attr) {
  assert(attr.getType().isMontgomery() &&
         "Expected Montgomery form for PrimeFieldAttr");

  auto standardType =
      PrimeFieldType::get(attr.getContext(), attr.getType().getModulus());
  APInt value = attr.getValue().getValue();
  APInt modulus = attr.getType().getModulus().getValue();
  auto modArithType = mod_arith::ModArithType::get(attr.getContext(),
                                                   attr.getType().getModulus());
  mod_arith::MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
  value = mulMod(value, montAttr.getRInv().getValue(), modulus);

  return PrimeFieldAttr::get(standardType, value);
}

PrimeFieldAttr getAttrAsMontgomeryForm(PrimeFieldAttr attr) {
  assert(!attr.getType().isMontgomery() &&
         "Expected standard form for PrimeFieldAttr");

  auto montType =
      PrimeFieldType::get(attr.getContext(), attr.getType().getModulus(), true);
  APInt value = attr.getValue().getValue();
  APInt modulus = attr.getType().getModulus().getValue();
  auto modArithType = mod_arith::ModArithType::get(attr.getContext(),
                                                   attr.getType().getModulus());
  mod_arith::MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
  value = mulMod(value, montAttr.getR().getValue(), modulus);

  return PrimeFieldAttr::get(montType, value);
}

Type getStandardFormType(Type type) {
  Type standardType = getElementTypeOrSelf(type);
  if (auto pfType = dyn_cast<PrimeFieldType>(standardType)) {
    if (pfType.isMontgomery()) {
      standardType =
          PrimeFieldType::get(type.getContext(), pfType.getModulus());
    }
  } else if (auto f2Type = dyn_cast<QuadraticExtFieldType>(standardType)) {
    if (f2Type.getBaseField().isMontgomery()) {
      auto pfType = PrimeFieldType::get(type.getContext(),
                                        f2Type.getBaseField().getModulus());
      standardType = QuadraticExtFieldType::get(
          type.getContext(), pfType, getAttrAsStandardForm(f2Type.getBeta()));
    }
  }
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    return MemRefType::get(memrefType.getShape(), standardType,
                           memrefType.getLayout(), memrefType.getMemorySpace());
  } else if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return shapedType.cloneWith(shapedType.getShape(), standardType);
  } else {
    return standardType;
  }
}

Type getMontgomeryFormType(Type type) {
  Type montType = getElementTypeOrSelf(type);
  if (auto pfType = dyn_cast<PrimeFieldType>(montType)) {
    if (!pfType.isMontgomery()) {
      montType =
          PrimeFieldType::get(type.getContext(), pfType.getModulus(), true);
    }
  } else if (auto f2Type = dyn_cast<QuadraticExtFieldType>(montType)) {
    if (!f2Type.isMontgomery()) {
      auto pfType = PrimeFieldType::get(
          type.getContext(), f2Type.getBaseField().getModulus(), true);
      montType = QuadraticExtFieldType::get(
          type.getContext(), pfType, getAttrAsMontgomeryForm(f2Type.getBeta()));
    }
  }
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    return MemRefType::get(memrefType.getShape(), montType,
                           memrefType.getLayout(), memrefType.getMemorySpace());
  } else if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return shapedType.cloneWith(shapedType.getShape(), montType);
  } else {
    return montType;
  }
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return adaptor.getValue();
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
static LogicalResult disallowShapedTypeOfExtField(OpType op) {
  // FIXME(batzor): In the prime field case, we rely on elementwise trait but in
  // the quadratic extension case, `linalg.generic` introduced by the
  // elementwise pass will be ill-formed due to the 1:N conversion.
  auto resultType = op.getResult().getType();
  if (isa<ShapedType>(resultType)) {
    auto elementType = cast<ShapedType>(resultType).getElementType();
    if (isa<QuadraticExtFieldType>(elementType)) {
      return op->emitOpError(
          "shaped type is not supported for quadratic "
          "extension field type");
    }
  }
  return success();
}

LogicalResult NegateOp::verify() { return disallowShapedTypeOfExtField(*this); }
LogicalResult AddOp::verify() { return disallowShapedTypeOfExtField(*this); }
LogicalResult SubOp::verify() { return disallowShapedTypeOfExtField(*this); }
LogicalResult MulOp::verify() { return disallowShapedTypeOfExtField(*this); }
LogicalResult PowOp::verify() { return disallowShapedTypeOfExtField(*this); }
LogicalResult InverseOp::verify() {
  return disallowShapedTypeOfExtField(*this);
}
LogicalResult FromMontOp::verify() {
  bool isMont = isMontgomery(this->getOutput().getType());
  if (isMont) {
    return emitOpError()
           << "FromMontOp result should be a standard type, but got "
           << getElementTypeOrSelf(this->getOutput().getType()) << ".";
  }
  return disallowShapedTypeOfExtField(*this);
}
LogicalResult ToMontOp::verify() {
  bool isMont = isMontgomery(this->getOutput().getType());
  if (!isMont) {
    return emitOpError()
           << "ToMontOp result should be a Montgomery type, but got "
           << getElementTypeOrSelf(this->getOutput().getType()) << ".";
  }
  return disallowShapedTypeOfExtField(*this);
}
LogicalResult ExtractOp::verify() {
  Type inputType = this->getInput().getType();
  TypeRange resultTypes = this->getOutput().getTypes();

  if (isa<ShapedType>(inputType) &&
      isa<QuadraticExtFieldType>(getElementTypeOrSelf(inputType))) {
    return emitOpError() << "shaped type is not supported for quadratic "
                            "extension field type";
  } else if (auto pfType = dyn_cast<PrimeFieldType>(inputType)) {
    // For prime field input, expect exactly one integer output
    if (resultTypes.size() != 1) {
      return emitOpError()
             << "expected one result type for prime field input, but got "
             << resultTypes.size();
    }
    auto intType = cast<IntegerType>(resultTypes[0]);
    if (intType.getWidth() != pfType.getModulus().getValue().getBitWidth()) {
      return emitOpError() << "result integer bitwidth " << intType.getWidth()
                           << " does not match prime field modulus bitwidth "
                           << pfType.getModulus().getValue().getBitWidth();
    }
  } else if (auto f2Type = dyn_cast<QuadraticExtFieldType>(inputType)) {
    if (isa<ShapedType>(inputType)) {
      return emitOpError() << "shaped type is not supported for quadratic "
                              "extension field type";
    }
    if (resultTypes.size() != 2) {
      return emitOpError() << "expected two result types for quadratic "
                              "extension field input, but got "
                           << resultTypes.size();
    }

    auto baseField = f2Type.getBaseField();
    unsigned modBitWidth = baseField.getModulus().getValue().getBitWidth();
    for (int i = 0; i < 2; i++) {
      auto intType = cast<IntegerType>((resultTypes[i]));
      if (intType.getWidth() != modBitWidth) {
        return emitOpError()
               << "result integer bitwidth " << intType.getWidth()
               << " does not match base field modulus bitwidth " << modBitWidth;
      }
    }
  }
  return success();
}
LogicalResult EncapsulateOp::verify() {
  Type resultType = (this->getOutput().getType());
  TypeRange inputTypes = this->getInput().getTypes();

  if (failed(disallowShapedTypeOfExtField(*this))) return failure();

  if (auto pfType = dyn_cast<PrimeFieldType>(resultType)) {
    if (inputTypes.size() != 1) {
      return emitOpError()
             << "expected one input for prime field output, but got "
             << inputTypes.size();
    }
    auto intType = cast<IntegerType>(inputTypes[0]);
    if (intType.getWidth() != pfType.getModulus().getValue().getBitWidth()) {
      return emitOpError() << "input integer bitwidth " << intType.getWidth()
                           << " does not match prime field modulus bitwidth "
                           << pfType.getModulus().getValue().getBitWidth();
    }
  } else if (auto f2Type = dyn_cast<QuadraticExtFieldType>(resultType)) {
    if (inputTypes.size() != 2) {
      return emitOpError() << "expected two input types for quadratic "
                              "extension field output, but got "
                           << inputTypes.size();
    }

    auto baseField = f2Type.getBaseField();
    unsigned modBitWidth = baseField.getModulus().getValue().getBitWidth();
    for (int i = 0; i < 2; i++) {
      auto intType = cast<IntegerType>((inputTypes[i]));
      if (intType.getWidth() != modBitWidth) {
        return emitOpError()
               << "input integer bitwidth " << intType.getWidth()
               << " does not match base field modulus bitwidth " << modBitWidth;
      }
    }
  }
  return success();
}

namespace {
#include "zkir/Dialect/Field/IR/FieldCanonicalization.cpp.inc"
}

void MulOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add<BitReverseMulBitReverse>(context);
}

}  // namespace mlir::zkir::field
