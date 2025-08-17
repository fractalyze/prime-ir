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
  if (result.failed())
    return result;
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

OpFoldResult ToMontOp::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast_if_present<IntegerAttr>(adaptor.getInput())) {
    auto modArithType = cast<ModArithType>(getOutput().getType());
    MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();

    APInt modulus = modArithType.getModulus().getValue();
    APInt resultValue =
        mulMod(input.getValue(), montAttr.getR().getValue(), modulus);
    return IntegerAttr::get(input.getType(), resultValue);
  }
  return {};
}

OpFoldResult FromMontOp::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast_if_present<IntegerAttr>(adaptor.getInput())) {
    auto modArithType = cast<ModArithType>(getOutput().getType());
    MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();

    APInt modulus = modArithType.getModulus().getValue();
    APInt resultValue =
        mulMod(input.getValue(), montAttr.getRInv().getValue(), modulus);
    return IntegerAttr::get(input.getType(), resultValue);
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
    auto modArithType = cast<ModArithType>(getOutput().getType());
    APInt modulus = modArithType.getModulus().getValue();
    APInt resultValue = modulus - input.getValue();
    return IntegerAttr::get(input.getType(), resultValue);
  }
  return {};
}

OpFoldResult DoubleOp::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast_if_present<IntegerAttr>(adaptor.getInput())) {
    auto modArithType = cast<ModArithType>(getOutput().getType());
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
    auto modArithType = cast<ModArithType>(getOutput().getType());
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
    auto modArithType = cast<ModArithType>(getOutput().getType());
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
    auto modArithType = cast<ModArithType>(getOutput().getType());
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
    auto modArithType = cast<ModArithType>(getOutput().getType());
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
  if (auto rhs = dyn_cast_if_present<IntegerAttr>(adaptor.getRhs())) {
    if (rhs.getValue().isZero()) {
      return getLhs();
    }
    if (auto lhs = dyn_cast_if_present<IntegerAttr>(adaptor.getLhs())) {
      auto modArithType = cast<ModArithType>(getOutput().getType());
      APInt modulus = modArithType.getModulus().getValue();
      APInt resultValue = lhs.getValue() + rhs.getValue();
      if (resultValue.uge(modulus)) {
        resultValue -= modulus;
      }
      return IntegerAttr::get(lhs.getType(), resultValue);
    }
  }
  return {};
}

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  if (auto rhs = dyn_cast_if_present<IntegerAttr>(adaptor.getRhs())) {
    if (rhs.getValue().isZero()) {
      return getLhs();
    }
    if (auto lhs = dyn_cast_if_present<IntegerAttr>(adaptor.getLhs())) {
      auto modArithType = cast<ModArithType>(getOutput().getType());
      APInt modulus = modArithType.getModulus().getValue();
      APInt rhsValue = rhs.getValue();
      APInt lhsValue = lhs.getValue();
      APInt resultValue;
      if (lhsValue.uge(rhsValue)) {
        resultValue = lhsValue - rhsValue;
      } else {
        resultValue = modulus - rhsValue + lhsValue;
      }
      return IntegerAttr::get(lhs.getType(), resultValue);
    }
  }
  return {};
}

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  if (auto rhs = dyn_cast_if_present<IntegerAttr>(adaptor.getRhs())) {
    auto modArithType = cast<ModArithType>(getOutput().getType());
    if (rhs.getValue().isZero()) {
      return getRhs();
    } else if (!modArithType.isMontgomery() && rhs.getValue().isOne()) {
      return getLhs();
    } else if (modArithType.isMontgomery()) {
      MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
      if (montAttr.getR().getValue().eq(rhs.getValue())) {
        return getLhs();
      }
    }
    if (auto lhs = dyn_cast_if_present<IntegerAttr>(adaptor.getLhs())) {
      APInt modulus = modArithType.getModulus().getValue();
      APInt resultValue = mulMod(lhs.getValue(), rhs.getValue(), modulus);
      if (modArithType.isMontgomery()) {
        MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
        resultValue =
            mulMod(resultValue, montAttr.getRInv().getValue(), modulus);
      }
      return IntegerAttr::get(lhs.getType(), resultValue);
    }
  }
  return {};
}

OpFoldResult MontMulOp::fold(FoldAdaptor adaptor) {
  if (auto rhs = dyn_cast_if_present<IntegerAttr>(adaptor.getRhs())) {
    auto modArithType = cast<ModArithType>(getOutput().getType());
    if (rhs.getValue().isZero()) {
      return getRhs();
    }
    MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
    if (montAttr.getR().getValue().eq(rhs.getValue())) {
      return getLhs();
    }
    if (auto lhs = dyn_cast_if_present<IntegerAttr>(adaptor.getLhs())) {
      APInt modulus = modArithType.getModulus().getValue();
      APInt resultValue = mulMod(lhs.getValue(), rhs.getValue(), modulus);
      resultValue = mulMod(resultValue, montAttr.getRInv().getValue(), modulus);
      return IntegerAttr::get(lhs.getType(), resultValue);
    }
  }
  return {};
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
  if (!modArithType)
    return failure();

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

namespace {
bool isNegativeOf(Attribute attr, Value val, uint32_t offset) {
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(val.getType()));
  APInt modulus = modArithType.getModulus().getValue();
  if (modArithType.isMontgomery()) {
    MontgomeryAttr montAttr = modArithType.getMontgomeryAttr();
    auto intAttr = cast<IntegerAttr>(attr);
    APInt montReduced =
        mulMod(intAttr.getValue(), montAttr.getRInv().getValue(), modulus);
    return montReduced == modulus - offset;
  } else {
    auto intAttr = cast<IntegerAttr>(attr);
    return intAttr.getValue() == modulus - offset;
  }
  return false;
}

bool isEqualTo(Attribute attr, uint32_t offset) {
  auto intAttr = cast<IntegerAttr>(attr);

  return (intAttr.getValue() - offset).isZero();
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
