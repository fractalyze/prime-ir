#include "zkir/Dialect/Field/IR/FieldOps.h"

#include "zkir/Dialect/TensorExt/IR/TensorExtOps.h"
#include "zkir/Utils/APIntUtils.h"

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
  auto montAttr =
      mod_arith::MontgomeryAttr::get(attr.getContext(), modArithType);
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
  auto montAttr =
      mod_arith::MontgomeryAttr::get(attr.getContext(), modArithType);
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

namespace {
#include "zkir/Dialect/Field/IR/FieldCanonicalization.cpp.inc"
}

void MulOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add<BitReverseMulBitReverse>(context);
}

}  // namespace mlir::zkir::field
