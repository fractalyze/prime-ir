/* Copyright 2025 The ZKIR Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "zkir/Dialect/ModArith/IR/ModArithOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"
#include "zkir/Dialect/ModArith/IR/ModArithOperation.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"
#include "zkir/Utils/AssemblyFormatUtils.h"
#include "zkir/Utils/ConstantFolder.h"

// IWYU pragma: begin_keep
// Headers needed for ModArithCanonicalization.cpp.inc
#include "mlir/IR/Matchers.h"
// IWYU pragma: end_keep

namespace mlir::zkir::mod_arith {
namespace {

struct ConstantFolderConfig {
  using NativeInputType = APInt;
  using NativeOutputType = APInt;
  using ScalarAttr = IntegerAttr;
  using TensorAttr = DenseIntElementsAttr;
};

class UnaryModArithConstantFolder
    : public UnaryConstantFolder<ConstantFolderConfig>::Delegate {
public:
  explicit UnaryModArithConstantFolder(Type type)
      : modArithType(cast<ModArithType>(getElementTypeOrSelf(type))),
        modulus(modArithType.getModulus().getValue()) {}

  APInt getNativeInput(IntegerAttr attr) const final { return attr.getValue(); }

  OpFoldResult getScalarAttr(const APInt &value) const final {
    return IntegerAttr::get(modArithType.getStorageType(), value);
  }

  OpFoldResult getTensorAttr(ShapedType type,
                             ArrayRef<APInt> values) const final {
    return DenseIntElementsAttr::get(type.clone(modArithType.getStorageType()),
                                     values);
  }

  ModArithType modArithType;
  APInt modulus;
};

class AdditiveModArithConstantFolder
    : public AdditiveConstantFolder<ConstantFolderConfig> {
public:
  explicit AdditiveModArithConstantFolder(Type type)
      : modArithType(cast<ModArithType>(getElementTypeOrSelf(type))),
        modulus(modArithType.getModulus().getValue()) {}

  bool isZero(const APInt &value) const final { return value.isZero(); }

  APInt getNativeInput(IntegerAttr attr) const final { return attr.getValue(); }

  OpFoldResult getScalarAttr(const APInt &value) const final {
    return IntegerAttr::get(modArithType.getStorageType(), value);
  }

  OpFoldResult getTensorAttr(ShapedType type,
                             ArrayRef<APInt> values) const final {
    return DenseIntElementsAttr::get(type.clone(modArithType.getStorageType()),
                                     values);
  }

  ModArithType modArithType;
  APInt modulus;
};

class MultiplicativeModArithConstantFolder
    : public MultiplicativeConstantFolder<ConstantFolderConfig> {
public:
  explicit MultiplicativeModArithConstantFolder(Type type)
      : modArithType(cast<ModArithType>(getElementTypeOrSelf(type))),
        montAttr(modArithType.getMontgomeryAttr()),
        modulus(modArithType.getModulus().getValue()) {
    auto oneStd = APInt(modulus.getBitWidth(), 1);
    auto oneMont = montAttr.getR().getValue();
    one = modArithType.isMontgomery() ? oneMont : oneStd;
  }

  bool isZero(const APInt &value) const final { return value.isZero(); }
  bool isOne(const APInt &value) const final { return value == one; }

  APInt getNativeInput(IntegerAttr attr) const final { return attr.getValue(); }

  OpFoldResult getScalarAttr(const APInt &value) const final {
    return IntegerAttr::get(modArithType.getStorageType(), value);
  }

  OpFoldResult getTensorAttr(ShapedType type,
                             ArrayRef<APInt> values) const final {
    return DenseIntElementsAttr::get(type.clone(modArithType.getStorageType()),
                                     values);
  }

  ModArithType modArithType;
  MontgomeryAttr montAttr;
  APInt modulus;
  APInt one;
};

Type convertFormType(Type type, bool toMontgomery) {
  Type elementType = getElementTypeOrSelf(type);
  auto modArithType = dyn_cast<ModArithType>(elementType);
  if (!modArithType || modArithType.isMontgomery() == toMontgomery) {
    return type;
  }

  auto newElementType = ModArithType::get(
      type.getContext(), modArithType.getModulus(), toMontgomery);
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return shapedType.clone(newElementType);
  }
  return newElementType;
}

} // namespace

Type getStandardFormType(Type type) {
  return convertFormType(type, /*toMontgomery=*/false);
}

Type getMontgomeryFormType(Type type) {
  return convertFormType(type, /*toMontgomery=*/true);
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

// static
ConstantOp ConstantOp::materialize(OpBuilder &builder, Attribute value,
                                   Type type, Location loc) {
  if (!isa<ModArithType>(getElementTypeOrSelf(type))) {
    return nullptr;
  }

  if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
    return builder.create<ConstantOp>(loc, type, intAttr);
  } else if (auto denseElementsAttr = dyn_cast<DenseIntElementsAttr>(value)) {
    return builder.create<ConstantOp>(loc, type, denseElementsAttr);
  }
  return nullptr;
}

Operation *ModArithDialect::materializeConstant(OpBuilder &builder,
                                                Attribute value, Type type,
                                                Location loc) {
  if (auto boolAttr = dyn_cast<BoolAttr>(value)) {
    return builder.create<arith::ConstantOp>(loc, boolAttr);
  } else if (auto denseElementsAttr = dyn_cast<DenseIntElementsAttr>(value)) {
    if (!isa<ModArithType>(getElementTypeOrSelf(type))) {
      // This could be a folding result of CmpOp.
      return builder.create<arith::ConstantOp>(loc, denseElementsAttr);
    }
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

namespace {

class ToMontConstantFolder : public UnaryModArithConstantFolder {
public:
  using UnaryModArithConstantFolder::UnaryModArithConstantFolder;

  APInt operate(const APInt &value) const final {
    return ModArithOperation::fromUnchecked(value, modArithType).toMont();
  }
};

} // namespace

OpFoldResult ToMontOp::fold(FoldAdaptor adaptor) {
  ToMontConstantFolder folder(getType());
  return UnaryConstantFolder<ConstantFolderConfig>::fold(adaptor, &folder);
}

namespace {

class FromMontConstantFolder : public UnaryModArithConstantFolder {
public:
  using UnaryModArithConstantFolder::UnaryModArithConstantFolder;

  APInt operate(const APInt &value) const final {
    return ModArithOperation::fromUnchecked(value, modArithType).fromMont();
  }
};

} // namespace

OpFoldResult FromMontOp::fold(FoldAdaptor adaptor) {
  FromMontConstantFolder folder(getType());
  return UnaryConstantFolder<ConstantFolderConfig>::fold(adaptor, &folder);
}

namespace {

struct CmpConstantFolderConfig {
  using NativeInputType = APInt;
  using NativeOutputType = bool;
  using ScalarAttr = IntegerAttr;
  using TensorAttr = DenseIntElementsAttr;
};

class CmpConstantFolder
    : public BinaryConstantFolder<CmpConstantFolderConfig>::Delegate {
public:
  explicit CmpConstantFolder(CmpOp *op)
      : context(op->getType().getContext()), predicate(op->getPredicate()) {}

  APInt getNativeInput(IntegerAttr attr) const final { return attr.getValue(); }

  OpFoldResult getScalarAttr(const bool &value) const final {
    return BoolAttr::get(context, value);
  }

  OpFoldResult getTensorAttr(ShapedType type,
                             ArrayRef<bool> values) const final {
    return DenseIntElementsAttr::get(type.clone(IntegerType::get(context, 1)),
                                     values);
  }

  bool operate(const APInt &a, const APInt &b) const final {
    switch (predicate) {
    case arith::CmpIPredicate::eq:
      return a.eq(b);
    case arith::CmpIPredicate::ne:
      return a.ne(b);
    case arith::CmpIPredicate::slt:
      return a.slt(b);
    case arith::CmpIPredicate::sle:
      return a.sle(b);
    case arith::CmpIPredicate::sgt:
      return a.sgt(b);
    case arith::CmpIPredicate::sge:
      return a.sge(b);
    case arith::CmpIPredicate::ult:
      return a.ult(b);
    case arith::CmpIPredicate::ule:
      return a.ule(b);
    case arith::CmpIPredicate::ugt:
      return a.ugt(b);
    case arith::CmpIPredicate::uge:
      return a.uge(b);
    }
  }

private:
  MLIRContext *const context;
  arith::CmpIPredicate predicate;
};

} // namespace

OpFoldResult CmpOp::fold(FoldAdaptor adaptor) {
  CmpConstantFolder folder(this);
  return BinaryConstantFolder<CmpConstantFolderConfig>::fold(adaptor, &folder);
}

namespace {

class NegateConstantFolder : public UnaryModArithConstantFolder {
public:
  using UnaryModArithConstantFolder::UnaryModArithConstantFolder;

  APInt operate(const APInt &value) const final {
    return -ModArithOperation::fromUnchecked(value, modArithType);
  }
};

} // namespace

OpFoldResult NegateOp::fold(FoldAdaptor adaptor) {
  NegateConstantFolder folder(getType());
  return UnaryConstantFolder<ConstantFolderConfig>::fold(adaptor, &folder);
}

namespace {

class DoubleConstantFolder : public UnaryModArithConstantFolder {
public:
  using UnaryModArithConstantFolder::UnaryModArithConstantFolder;

  APInt operate(const APInt &value) const final {
    return ModArithOperation::fromUnchecked(value, modArithType).dbl();
  }
};

} // namespace

OpFoldResult DoubleOp::fold(FoldAdaptor adaptor) {
  DoubleConstantFolder folder(getType());
  return UnaryConstantFolder<ConstantFolderConfig>::fold(adaptor, &folder);
}

namespace {

class SquareConstantFolder : public UnaryModArithConstantFolder {
public:
  using UnaryModArithConstantFolder::UnaryModArithConstantFolder;

  APInt operate(const APInt &value) const final {
    return ModArithOperation::fromUnchecked(value, modArithType).square();
  }
};

} // namespace

OpFoldResult SquareOp::fold(FoldAdaptor adaptor) {
  SquareConstantFolder folder(getType());
  return UnaryConstantFolder<ConstantFolderConfig>::fold(adaptor, &folder);
}

namespace {

class MontSquareConstantFolder : public UnaryModArithConstantFolder {
public:
  using UnaryModArithConstantFolder::UnaryModArithConstantFolder;

  APInt operate(const APInt &value) const final {
    return ModArithOperation::fromUnchecked(value, modArithType).square();
  }
};

} // namespace

OpFoldResult MontSquareOp::fold(FoldAdaptor adaptor) {
  MontSquareConstantFolder folder(getType());
  return UnaryConstantFolder<ConstantFolderConfig>::fold(adaptor, &folder);
}

namespace {

class InverseConstantFolder : public UnaryModArithConstantFolder {
public:
  using UnaryModArithConstantFolder::UnaryModArithConstantFolder;

  APInt operate(const APInt &value) const final {
    return ModArithOperation::fromUnchecked(value, modArithType).inverse();
  }
};

} // namespace

OpFoldResult InverseOp::fold(FoldAdaptor adaptor) {
  InverseConstantFolder folder(getType());
  return UnaryConstantFolder<ConstantFolderConfig>::fold(adaptor, &folder);
}

namespace {

class MontInverseConstantFolder : public UnaryModArithConstantFolder {
public:
  using UnaryModArithConstantFolder::UnaryModArithConstantFolder;

  APInt operate(const APInt &value) const final {
    return ModArithOperation::fromUnchecked(value, modArithType).inverse();
  }
};

} // namespace

OpFoldResult MontInverseOp::fold(FoldAdaptor adaptor) {
  MontInverseConstantFolder folder(getType());
  return UnaryConstantFolder<ConstantFolderConfig>::fold(adaptor, &folder);
}

namespace {

class AddConstantFolder : public AdditiveModArithConstantFolder {
public:
  explicit AddConstantFolder(AddOp *op)
      : AdditiveModArithConstantFolder(op->getType()), op(op) {}

  OpFoldResult getLhs() const final { return op->getLhs(); }

  APInt operate(const APInt &a, const APInt &b) const final {
    return ModArithOperation::fromUnchecked(a, modArithType) +
           ModArithOperation::fromUnchecked(b, modArithType);
  }

private:
  AddOp *const op;
};

} // namespace

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  AddConstantFolder folder(this);
  return BinaryConstantFolder<ConstantFolderConfig>::fold(adaptor, &folder);
}

namespace {

class SubConstantFolder : public AdditiveModArithConstantFolder {
public:
  explicit SubConstantFolder(SubOp *op)
      : AdditiveModArithConstantFolder(op->getType()), op(op) {}

  OpFoldResult getLhs() const final { return op->getLhs(); }

  APInt operate(const APInt &a, const APInt &b) const final {
    return ModArithOperation::fromUnchecked(a, modArithType) -
           ModArithOperation::fromUnchecked(b, modArithType);
  }

private:
  SubOp *const op;
};

} // namespace

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  SubConstantFolder folder(this);
  return BinaryConstantFolder<ConstantFolderConfig>::fold(adaptor, &folder);
}

namespace {

class MulConstantFolder : public MultiplicativeModArithConstantFolder {
public:
  explicit MulConstantFolder(MulOp *op)
      : MultiplicativeModArithConstantFolder(op->getType()), op(op) {}

  OpFoldResult getLhs() const final { return op->getLhs(); }

  APInt operate(const APInt &a, const APInt &b) const final {
    return ModArithOperation::fromUnchecked(a, modArithType) *
           ModArithOperation::fromUnchecked(b, modArithType);
  }

private:
  MulOp *const op;
};

} // namespace

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  MulConstantFolder folder(this);
  return BinaryConstantFolder<ConstantFolderConfig>::fold(adaptor, &folder);
}

namespace {

class MontMulConstantFolder : public MultiplicativeModArithConstantFolder {
public:
  explicit MontMulConstantFolder(MontMulOp *op)
      : MultiplicativeModArithConstantFolder(op->getType()), op(op) {}

  OpFoldResult getLhs() const final { return op->getLhs(); }

  APInt operate(const APInt &a, const APInt &b) const final {
    return ModArithOperation::fromUnchecked(a, modArithType) *
           ModArithOperation::fromUnchecked(b, modArithType);
  }

private:
  MontMulOp *const op;
};

} // namespace

OpFoldResult MontMulOp::fold(FoldAdaptor adaptor) {
  MontMulConstantFolder folder(this);
  return BinaryConstantFolder<ConstantFolderConfig>::fold(adaptor, &folder);
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<APInt> parsedInts;
  Type parsedType;

  auto getModulusCallback = [&](APInt &modulus) -> ParseResult {
    if (auto modArithType =
            dyn_cast<ModArithType>(getElementTypeOrSelf(parsedType))) {
      modulus = modArithType.getModulus().getValue();
      return success();
    }

    return parser.emitError(parser.getCurrentLocation(),
                            "expected ModArithType for modulus");
  };

  auto parseResult = parseOptionalModularInteger(
      parser, parsedInts.emplace_back(), parsedType, getModulusCallback);
  if (parseResult.has_value()) {
    if (failed(parseResult.value())) {
      return failure();
    }
    result.addAttribute(
        "value",
        IntegerAttr::get(cast<ModArithType>(parsedType).getStorageType(),
                         parsedInts[0]));
    result.addTypes(parsedType);
    return success();
  }

  parsedInts.pop_back();
  if (failed(parseModularIntegerList(parser, parsedInts, parsedType,
                                     getModulusCallback))) {
    return failure();
  }

  auto shapedType = cast<ShapedType>(parsedType);
  auto denseElementsAttr = DenseIntElementsAttr::get(
      shapedType.clone(
          cast<ModArithType>(shapedType.getElementType()).getStorageType()),
      parsedInts);
  result.addAttribute("value", denseElementsAttr);
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
    auto valueOp =
        ModArithOperation::fromUnchecked(intAttr.getValue(), modArithType);
    ModArithType stdType = modArithType;
    if (modArithType.isMontgomery()) {
      stdType = cast<ModArithType>(getStandardFormType(modArithType));
      valueOp = ModArithOperation(valueOp.fromMont(), stdType);
    }
    ModArithOperation offsetOp(APInt(intAttr.getValue().getBitWidth(), offset),
                               stdType);
    return valueOp == -offsetOp;
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
