#include "zkir/Dialect/ModArith/IR/ModArithOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"
#include "zkir/Utils/APIntUtils.h"

// IWYU pragma: begin_keep
// Headers needed for ModArithCanonicalization.cpp.inc
#include "mlir/IR/Matchers.h"
// IWYU pragma: end_keep

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

bool BitcastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  Type inputType = getElementTypeOrSelf(inputs.front());
  Type outputType = getElementTypeOrSelf(outputs.front());

  return getIntOrModArithBitWidth(inputType) ==
         getIntOrModArithBitWidth(outputType);
}

LogicalResult MontReduceOp::verify() {
  IntegerType integerType =
      cast<IntegerType>(getElementTypeOrSelf(getLow().getType()));
  ModArithType modArithType = getResultModArithType(*this);
  unsigned intWidth = integerType.getWidth();
  unsigned modWidth = modArithType.getStorageBitWidth();
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
  return success();
}

LogicalResult FromMontOp::verify() {
  ModArithType resultType = getResultModArithType(*this);
  if (resultType.isMontgomery())
    return emitOpError() << "FromMontOp result should be a standard type, "
                         << "but got " << resultType << ".";
  return success();
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return adaptor.getValue();
}

ConstantOp ConstantOp::materialize(OpBuilder &builder, Attribute value,
                                   Type type, Location loc) {
  return builder.create<ConstantOp>(loc, type, cast<TypedAttr>(value));
}

Operation *ModArithDialect::materializeConstant(OpBuilder &builder,
                                                Attribute value, Type type,
                                                Location loc) {
  if (auto boolAttr = dyn_cast<BoolAttr>(value)) {
    return builder.create<arith::ConstantOp>(loc, boolAttr);
  }
  return ConstantOp::materialize(builder, value, type, loc);
}

OpFoldResult BitcastOp::fold(FoldAdaptor adaptor) {
  if (isa_and_present<IntegerAttr>(adaptor.getInput()) ||
      isa_and_present<DenseIntElementsAttr>(adaptor.getInput())) {
    return adaptor.getInput();
  }
  return {};
}

OpFoldResult ToMontOp::fold(FoldAdaptor adaptor) {
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(getType()));
  MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
  APInt modulus = modArithType.getModulus().getValue();

  auto toMontConversion = [montAttr, modulus](APInt value) {
    return mulMod(value, montAttr.getR().getValue(), modulus);
  };

  if (auto input = dyn_cast_if_present<IntegerAttr>(adaptor.getInput())) {
    return IntegerAttr::get(input.getType(),
                            toMontConversion(input.getValue()));
  } else if (auto input = dyn_cast_if_present<DenseIntElementsAttr>(
                 adaptor.getInput())) {
    return input.mapValues(modArithType.getStorageType(), toMontConversion);
  }
  return {};
}

OpFoldResult FromMontOp::fold(FoldAdaptor adaptor) {
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(getType()));
  MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
  APInt modulus = modArithType.getModulus().getValue();

  auto fromMontConversion = [montAttr, modulus](APInt value) {
    return mulMod(value, montAttr.getRInv().getValue(), modulus);
  };

  if (auto input = dyn_cast_if_present<IntegerAttr>(adaptor.getInput())) {
    return IntegerAttr::get(input.getType(),
                            fromMontConversion(input.getValue()));
  } else if (auto input = dyn_cast_if_present<DenseIntElementsAttr>(
                 adaptor.getInput())) {
    return input.mapValues(modArithType.getStorageType(), fromMontConversion);
  }
  return {};
}

OpFoldResult CmpOp::fold(FoldAdaptor adaptor) {
  auto predicate = adaptor.getPredicate();
  if (auto lhs = dyn_cast_if_present<IntegerAttr>(adaptor.getLhs())) {
    if (auto rhs = dyn_cast_if_present<IntegerAttr>(adaptor.getRhs())) {
      APInt lhsValue = lhs.getValue();
      APInt rhsValue = rhs.getValue();
      switch (predicate) {
      case arith::CmpIPredicate::eq:
        return BoolAttr::get(getType().getContext(), lhsValue.eq(rhsValue));
      case arith::CmpIPredicate::ne:
        return BoolAttr::get(getType().getContext(), lhsValue.ne(rhsValue));
      case arith::CmpIPredicate::slt:
        return BoolAttr::get(getType().getContext(), lhsValue.slt(rhsValue));
      case arith::CmpIPredicate::sle:
        return BoolAttr::get(getType().getContext(), lhsValue.sle(rhsValue));
      case arith::CmpIPredicate::sgt:
        return BoolAttr::get(getType().getContext(), lhsValue.sgt(rhsValue));
      case arith::CmpIPredicate::sge:
        return BoolAttr::get(getType().getContext(), lhsValue.sge(rhsValue));
      case arith::CmpIPredicate::ult:
        return BoolAttr::get(getType().getContext(), lhsValue.ult(rhsValue));
      case arith::CmpIPredicate::ule:
        return BoolAttr::get(getType().getContext(), lhsValue.ule(rhsValue));
      case arith::CmpIPredicate::ugt:
        return BoolAttr::get(getType().getContext(), lhsValue.ugt(rhsValue));
      case arith::CmpIPredicate::uge:
        return BoolAttr::get(getType().getContext(), lhsValue.uge(rhsValue));
      default:
        llvm_unreachable("unknown cmpi predicate kind");
      }
    }
  }
  return {};
}

OpFoldResult NegateOp::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast_if_present<IntegerAttr>(adaptor.getInput())) {
    auto modArithType = cast<ModArithType>(getType());
    APInt modulus = modArithType.getModulus().getValue();
    APInt resultValue = modulus - input.getValue();
    return IntegerAttr::get(input.getType(), resultValue);
  }
  return {};
}

OpFoldResult DoubleOp::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast_if_present<IntegerAttr>(adaptor.getInput())) {
    auto modArithType = cast<ModArithType>(getType());
    APInt modulus = modArithType.getModulus().getValue();
    assert(modulus.getBitWidth() > modulus.getActiveBits());
    APInt resultValue = input.getValue().shl(1);
    if (resultValue.uge(modulus)) {
      resultValue -= modulus;
    }
    return IntegerAttr::get(input.getType(), resultValue);
  }
  return {};
}

OpFoldResult SquareOp::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast_if_present<IntegerAttr>(adaptor.getInput())) {
    auto modArithType = cast<ModArithType>(getType());
    APInt modulus = modArithType.getModulus().getValue();
    APInt resultValue = mulMod(input.getValue(), input.getValue(), modulus);
    if (modArithType.isMontgomery()) {
      MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
      resultValue = mulMod(resultValue, montAttr.getRInv().getValue(), modulus);
    }
    return IntegerAttr::get(input.getType(), resultValue);
  }
  return {};
}

OpFoldResult MontSquareOp::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast_if_present<IntegerAttr>(adaptor.getInput())) {
    auto modArithType = cast<ModArithType>(getType());
    APInt modulus = modArithType.getModulus().getValue();
    APInt resultValue = mulMod(input.getValue(), input.getValue(), modulus);
    MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
    resultValue = mulMod(resultValue, montAttr.getRInv().getValue(), modulus);
    return IntegerAttr::get(input.getType(), resultValue);
  }
  return {};
}

OpFoldResult InverseOp::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast_if_present<IntegerAttr>(adaptor.getInput())) {
    auto modArithType = cast<ModArithType>(getType());
    APInt modulus = modArithType.getModulus().getValue();
    APInt resultValue = multiplicativeInverse(input.getValue(), modulus);
    if (modArithType.isMontgomery()) {
      MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
      resultValue =
          mulMod(resultValue, montAttr.getRSquared().getValue(), modulus);
    }
    return IntegerAttr::get(input.getType(), resultValue);
  }
  return {};
}

OpFoldResult MontInverseOp::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast_if_present<IntegerAttr>(adaptor.getInput())) {
    auto modArithType = cast<ModArithType>(getType());
    MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();

    APInt modulus = modArithType.getModulus().getValue();
    APInt resultValue = multiplicativeInverse(input.getValue(), modulus);
    resultValue =
        mulMod(resultValue, montAttr.getRSquared().getValue(), modulus);
    return IntegerAttr::get(input.getType(), resultValue);
  }
  return {};
}

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  APInt rhsValue;
  if (!matchPattern(adaptor.getRhs(), m_ConstantInt(&rhsValue))) {
    return {};
  }

  if (rhsValue.isZero()) {
    return getLhs();
  }

  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(getType()));
  APInt modulus = modArithType.getModulus().getValue();
  auto addMod = [modulus](const APInt &a, const APInt &b) -> APInt {
    APInt sum = a + b;
    if (sum.uge(modulus)) {
      sum -= modulus;
    }
    return sum;
  };

  if (auto lhsConst = dyn_cast_if_present<IntegerAttr>(adaptor.getLhs())) {
    APInt lhsValue = lhsConst.getValue();
    APInt resultValue = addMod(lhsValue, rhsValue);
    return IntegerAttr::get(lhsConst.getType(), resultValue);
  }
  if (auto denseIntAttr =
          dyn_cast_if_present<DenseIntElementsAttr>(adaptor.getLhs())) {
    return denseIntAttr.mapValues(
        modArithType.getStorageType(),
        [addMod, rhsValue](APInt value) { return addMod(value, rhsValue); });
  }
  return {};
}

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  APInt rhsValue;
  if (!matchPattern(adaptor.getRhs(), m_ConstantInt(&rhsValue))) {
    return {};
  }

  // x - 0 -> x
  if (rhsValue.isZero()) {
    return getLhs();
  }

  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(getType()));
  APInt modulus = modArithType.getModulus().getValue();
  auto subMod = [modulus](const APInt &a, const APInt &b) -> APInt {
    auto diff = a + modulus - b;
    if (diff.uge(modulus)) {
      diff -= modulus;
    }
    return diff;
  };

  if (auto lhsConst = dyn_cast_if_present<IntegerAttr>(adaptor.getLhs())) {
    APInt lhsValue = lhsConst.getValue();
    APInt resultValue = subMod(lhsValue, rhsValue);
    return IntegerAttr::get(lhsConst.getType(), resultValue);
  }
  if (auto denseIntAttr =
          dyn_cast_if_present<DenseIntElementsAttr>(adaptor.getLhs())) {
    return denseIntAttr.mapValues(
        modArithType.getStorageType(),
        [subMod, rhsValue](APInt value) { return subMod(value, rhsValue); });
  }
  return {};
}

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(getType()));
  MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
  APInt modulus = modArithType.getModulus().getValue();
  APInt rhsValue;
  if (!matchPattern(adaptor.getRhs(), m_ConstantInt(&rhsValue))) {
    return {};
  }

  // x * 0 -> 0
  if (rhsValue.isZero()) {
    return getRhs();
  }

  // if in Montgomery domain, reduce the constant to standard form
  if (modArithType.isMontgomery()) {
    rhsValue = mulMod(rhsValue, montAttr.getRInv().getValue(), modulus);
  }

  // x * 1 -> x
  if (rhsValue.isOne()) {
    return getLhs();
  }

  if (auto lhsConst = dyn_cast_if_present<IntegerAttr>(adaptor.getLhs())) {
    APInt lhsValue = lhsConst.getValue();
    APInt resultValue = mulMod(lhsValue, rhsValue, modulus);
    return IntegerAttr::get(lhsConst.getType(), resultValue);
  }
  if (auto denseIntAttr =
          dyn_cast_if_present<DenseIntElementsAttr>(adaptor.getLhs())) {
    return denseIntAttr.mapValues(modArithType.getStorageType(),
                                  [rhsValue, modulus](APInt value) {
                                    return mulMod(value, rhsValue, modulus);
                                  });
  }
  return {};
}

OpFoldResult MontMulOp::fold(FoldAdaptor adaptor) {
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(getType()));
  MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
  APInt modulus = modArithType.getModulus().getValue();
  APInt rhsValue;
  if (!matchPattern(adaptor.getRhs(), m_ConstantInt(&rhsValue))) {
    return {};
  }

  // x * 0 -> 0
  if (rhsValue.isZero()) {
    return getRhs();
  }

  rhsValue = mulMod(rhsValue, montAttr.getRInv().getValue(), modulus);

  // x * 1 -> x
  if (rhsValue.isOne()) {
    return getLhs();
  }

  if (auto lhsConst = dyn_cast_if_present<IntegerAttr>(adaptor.getLhs())) {
    APInt lhsValue = lhsConst.getValue();
    APInt resultValue = mulMod(lhsValue, rhsValue, modulus);
    return IntegerAttr::get(lhsConst.getType(), resultValue);
  }
  if (auto denseIntAttr =
          dyn_cast_if_present<DenseIntElementsAttr>(adaptor.getLhs())) {
    return denseIntAttr.mapValues(modArithType.getStorageType(),
                                  [rhsValue, modulus](APInt value) {
                                    return mulMod(value, rhsValue, modulus);
                                  });
  }
  return {};
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  APInt parsedInt;
  Type parsedType;
  DenseElementsAttr valueAttr;

  if (parser.parseOptionalInteger(parsedInt).has_value()) {
    if (failed(parser.parseColonType(parsedType)))
      return failure();

    if (parsedInt.isNegative()) {
      parser.emitError(parser.getCurrentLocation(),
                       "negative value is not allowed");
      return failure();
    }

    auto modArithType = dyn_cast<ModArithType>(parsedType);
    if (!modArithType)
      return failure();

    APInt modulus = modArithType.getModulus().getValue();
    if (modulus.isNegative() || modulus.isZero()) {
      parser.emitError(parser.getCurrentLocation(), "modulus must be positive");
      return failure();
    }

    unsigned outputBitWidth = modArithType.getStorageBitWidth();
    if (parsedInt.getActiveBits() > outputBitWidth) {
      parser.emitError(parser.getCurrentLocation(),
                       "constant value is too large for the underlying type");
      return failure();
    }

    // zero-extend or truncate to the correct bitwidth
    parsedInt = parsedInt.zextOrTrunc(outputBitWidth).urem(modulus);
    result.addAttribute(
        "value", IntegerAttr::get(modArithType.getStorageType(), parsedInt));
    result.addTypes(parsedType);
    return success();
  }

  if (failed(parser.parseAttribute(valueAttr))) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected value to be a scalar or dense elements attr");
    return failure();
  }

  if (failed(parser.parseColonType(parsedType))) {
    return failure();
  }

  auto shapedType = dyn_cast<ShapedType>(parsedType);
  if (!shapedType) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected result type to be a shaped type");
    return failure();
  }

  // check if the shape of the value attribute is the same as the result type
  if (shapedType.getShape() != valueAttr.getType().getShape()) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected 'value' attribute to be a dense elements attr "
                     "with the same shape as the result type");
    return failure();
  }

  // check if the result type is mod arith like
  auto modArithType = dyn_cast<ModArithType>(shapedType.getElementType());
  if (!modArithType) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected result type to be a mod arith type");
    return failure();
  }

  // check if the element type of the value attribute is the same as the mod
  // arith storage type
  if (modArithType.getStorageType() != valueAttr.getType().getElementType()) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected 'value' attribute to be a dense elements attr "
                     "with the same element type as the mod arith storage");
    return failure();
  }

  result.addAttribute("value", valueAttr);
  result.addTypes(parsedType);
  return success();
}

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getValue());
  p << " : ";
  p.printType(getType());
}

namespace {
bool isNegativeOf(Attribute attr, Value val, uint32_t offset) {
  IntegerAttr intAttr = dyn_cast_if_present<IntegerAttr>(attr);
  if (auto denseIntAttr = dyn_cast_if_present<SplatElementsAttr>(attr)) {
    intAttr = denseIntAttr.getSplatValue<IntegerAttr>();
  }
  if (intAttr) {
    auto modArithType = cast<ModArithType>(getElementTypeOrSelf(val.getType()));
    APInt modulus = modArithType.getModulus().getValue();
    if (modArithType.isMontgomery()) {
      MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
      APInt montReduced =
          mulMod(intAttr.getValue(), montAttr.getRInv().getValue(), modulus);
      return montReduced == modulus - offset;
    } else {
      auto intAttr = cast<IntegerAttr>(attr);
      return intAttr.getValue() == modulus - offset;
    }
  }
  return false;
}

bool isEqualTo(Attribute attr, uint32_t offset) {
  IntegerAttr intAttr = dyn_cast_if_present<IntegerAttr>(attr);
  if (auto denseIntAttr = dyn_cast_if_present<SplatElementsAttr>(attr)) {
    intAttr = denseIntAttr.getSplatValue<IntegerAttr>();
  }
  if (intAttr) {
    return (intAttr.getValue() - offset).isZero();
  }
  return false;
}
} // namespace

namespace {
#include "zkir/Dialect/ModArith/IR/ModArithCanonicalization.cpp.inc"
}

void AddOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add<AddConstantTwice>(context);
  patterns.add<AddConstantToSubLhs>(context);
  patterns.add<AddConstantToSubRhs>(context);
  patterns.add<AddSelfIsDouble>(context);
  patterns.add<AddBothNegated>(context);
  patterns.add<AddAfterSub>(context);
  patterns.add<AddAfterNegLhs>(context);
  patterns.add<AddAfterNegRhs>(context);
  patterns.add<FactorMulAdd>(context);
}

void SubOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add<SubConstantFromAdd>(context);
  patterns.add<SubConstantTwiceLhs>(context);
  patterns.add<SubConstantTwiceRhs>(context);
  patterns.add<SubAddFromConstant>(context);
  patterns.add<SubSubFromConstantLhs>(context);
  patterns.add<SubSubFromConstantRhs>(context);
  patterns.add<SubSelfIsZero>(context);
  patterns.add<SubLhsAfterAdd>(context);
  patterns.add<SubRhsAfterAdd>(context);
  patterns.add<SubLhsAfterSub>(context);
  patterns.add<SubAfterNegLhs>(context);
  patterns.add<SubAfterNegRhs>(context);
  patterns.add<SubBothNegated>(context);
  patterns.add<SubAfterSquareBoth>(context);
  patterns.add<FactorMulSub>(context);
}

void MulOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add<MulByTwoIsDouble>(context);
  patterns.add<MulSelfIsSquare>(context);
  patterns.add<MulNegativeOneRhs>(context);
  patterns.add<MulNegativeTwoRhs>(context);
  patterns.add<MulNegativeThreeRhs>(context);
  patterns.add<MulNegativeFourRhs>(context);
  patterns.add<MulConstantTwice>(context);
  patterns.add<MulOfMulByConstant>(context);
  patterns.add<MulAddDistributeConstant>(context);
  patterns.add<MulSubDistributeConstantRhs>(context);
  patterns.add<MulSubDistributeConstantLhs>(context);
}

} // namespace mlir::zkir::mod_arith
