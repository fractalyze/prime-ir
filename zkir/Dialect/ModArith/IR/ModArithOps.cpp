#include "zkir/Dialect/ModArith/IR/ModArithOps.h"

namespace mlir::zkir::mod_arith {

Type getStandardFormType(Type type) {
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(type));
  if (modArithType.isMontgomery()) {
    auto standardType =
        ModArithType::get(type.getContext(), modArithType.getModulus());
    if (auto memrefType = dyn_cast<MemRefType>(type)) {
      return MemRefType::get(memrefType.getShape(), standardType,
                             memrefType.getLayout(),
                             memrefType.getMemorySpace());
    } else if (auto shapedType = dyn_cast<ShapedType>(type)) {
      return shapedType.cloneWith(shapedType.getShape(), standardType);
    } else {
      return standardType;
    }
  } else {
    return type;
  }
}

Type getMontgomeryFormType(Type type) {
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(type));
  if (!modArithType.isMontgomery()) {
    auto montType =
        ModArithType::get(type.getContext(), modArithType.getModulus(), true);
    if (auto memrefType = dyn_cast<MemRefType>(type)) {
      return MemRefType::get(memrefType.getShape(), montType,
                             memrefType.getLayout(),
                             memrefType.getMemorySpace());
    } else if (auto shapedType = dyn_cast<ShapedType>(type)) {
      return shapedType.cloneWith(shapedType.getShape(), montType);
    } else {
      return montType;
    }
  } else {
    return type;
  }
}

/// Ensures that the underlying integer type is wide enough for the modulus
template <typename OpType>
LogicalResult verifyModArithType(OpType op, ModArithType type) {
  APInt modulus = type.getModulus().getValue();
  unsigned bitWidth = modulus.getBitWidth();
  unsigned modWidth = modulus.getActiveBits();
  if (modWidth > bitWidth - 1)
    return op.emitOpError()
           << "underlying type's bitwidth must be 1 bit larger than "
           << "the modulus bitwidth, but got " << bitWidth
           << " while modulus requires width " << modWidth << ".";
  return success();
}

template <typename OpType>
LogicalResult verifySameWidth(OpType op, ModArithType modArithType,
                              IntegerType integerType) {
  unsigned bitWidth = modArithType.getModulus().getValue().getBitWidth();
  unsigned intWidth = integerType.getWidth();
  if (intWidth != bitWidth)
    return op.emitOpError()
           << "the result integer type should be of the same width as the "
           << "mod arith type width, but got " << intWidth
           << " while mod arith type width " << bitWidth << ".";
  return success();
}

LogicalResult NegateOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult ExtractOp::verify() {
  auto modArithType = getOperandModArithType(*this);
  auto integerType = getResultIntegerType(*this);
  auto result = verifySameWidth(*this, modArithType, integerType);
  if (result.failed()) return result;
  return verifyModArithType(*this, modArithType);
}

LogicalResult MontReduceOp::verify() {
  IntegerType integerType =
      cast<IntegerType>(getElementTypeOrSelf(this->getLow().getType()));
  ModArithType modArithType = getResultModArithType(*this);
  unsigned intWidth = integerType.getWidth();
  unsigned modWidth = modArithType.getModulus().getValue().getBitWidth();
  if (intWidth != modWidth)
    return emitOpError() << "Expected operand width to be " << modWidth
                         << ", but got " << intWidth << " instead.";
  return success();
}

LogicalResult ToMontOp::verify() {
  ModArithType resultType = getResultModArithType(*this);
  if (!resultType.isMontgomery())
    return emitOpError() << "ToMontOp result should be a Montgomery type, "
                         << "but got " << resultType << ".";
  return verifyModArithType(*this, resultType);
}

LogicalResult FromMontOp::verify() {
  ModArithType resultType = getResultModArithType(*this);
  if (resultType.isMontgomery())
    return emitOpError() << "FromMontOp result should be a standard type, "
                         << "but got " << resultType << ".";
  return verifyModArithType(*this, resultType);
}

LogicalResult MontMulOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult AddOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult DoubleOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult SubOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult MulOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult SquareOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult MontSquareOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

LogicalResult MacOp::verify() {
  return verifyModArithType(*this, getResultModArithType(*this));
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return adaptor.getValue();
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

  auto modArithType = dyn_cast<ModArithType>(parsedType);
  if (!modArithType) return failure();

  auto modulus = modArithType.getModulus().getValue();
  if (modulus.isNegative() || modulus.isZero()) {
    parser.emitError(parser.getCurrentLocation(), "modulus must be positive");
    return failure();
  }

  auto outputBitWidth = modArithType.getModulus().getValue().getBitWidth();
  if (parsedInt.getActiveBits() > outputBitWidth) {
    parser.emitError(parser.getCurrentLocation(),
                     "constant value is too large for the underlying type");
    return failure();
  }

  // zero-extend or truncate to the correct bitwidth
  parsedInt = parsedInt.zextOrTrunc(outputBitWidth).urem(modulus);
  result.addAttribute(
      "value",
      IntegerAttr::get(modArithType.getModulus().getType(), parsedInt));
  result.addTypes(parsedType);
  return success();
}

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getValue());
  p << " : ";
  p.printType(getOutput().getType());
}

}  // namespace mlir::zkir::mod_arith
