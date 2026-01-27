/* Copyright 2025 The PrimeIR Authors.

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

#include "prime_ir/Dialect/ModArith/IR/ModArithOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithDialect.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithOperation.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"
#include "prime_ir/Utils/AssemblyFormatUtils.h"
#include "prime_ir/Utils/ConstantFolder.h"

// IWYU pragma: begin_keep
// Headers needed for ModArithCanonicalization.cpp.inc
#include "mlir/IR/Matchers.h"
// IWYU pragma: end_keep

namespace mlir::prime_ir::mod_arith {
namespace {

struct ConstantFolderConfig {
  using NativeInputType = APInt;
  using NativeOutputType = APInt;
  using ScalarAttr = IntegerAttr;
  using TensorAttr = DenseIntElementsAttr;
};

template <typename BaseDelegate>
class ModArithFolderMixin : public BaseDelegate {
public:
  explicit ModArithFolderMixin(Type type)
      : modArithType(cast<ModArithType>(getElementTypeOrSelf(type))) {}

  APInt getNativeInput(IntegerAttr attr) const final { return attr.getValue(); }

  OpFoldResult getScalarAttr(const APInt &value) const final {
    return IntegerAttr::get(modArithType.getStorageType(), value);
  }

  OpFoldResult getTensorAttr(ShapedType type,
                             ArrayRef<APInt> values) const final {
    return DenseIntElementsAttr::get(type.clone(modArithType.getStorageType()),
                                     values);
  }

protected:
  ModArithType modArithType;
};

template <typename Func>
class GenericUnaryModArithConstantFolder
    : public ModArithFolderMixin<
          UnaryConstantFolder<ConstantFolderConfig>::Delegate> {
public:
  GenericUnaryModArithConstantFolder(Type outputType, Func fn,
                                     Type inputType = nullptr)
      : ModArithFolderMixin(outputType), fn(fn) {
    if (inputType)
      this->inputType = cast<ModArithType>(getElementTypeOrSelf(inputType));
    else
      this->inputType = this->modArithType;
  }

  APInt operate(const APInt &value) const final {
    return static_cast<APInt>(
        fn(ModArithOperation::fromUnchecked(value, inputType)));
  }

private:
  Func fn;
  ModArithType inputType;
};

template <typename Op, typename Func>
OpFoldResult foldUnaryOp(Op *op, typename Op::FoldAdaptor adaptor, Func fn,
                         Type inputType = nullptr) {
  GenericUnaryModArithConstantFolder<Func> folder(op->getType(), fn, inputType);
  return UnaryConstantFolder<ConstantFolderConfig>::fold(adaptor, &folder);
}

template <typename BaseDelegate, typename Op, typename Func>
class GenericBinaryModArithConstantFolder
    : public ModArithFolderMixin<BaseDelegate> {
public:
  GenericBinaryModArithConstantFolder(Op *op, Func fn)
      : ModArithFolderMixin<BaseDelegate>(op->getType()), op(op), fn(fn) {}

  OpFoldResult getLhs() const final { return op->getLhs(); }

  APInt operate(const APInt &a, const APInt &b) const final {
    return static_cast<APInt>(
        fn(ModArithOperation::fromUnchecked(a, this->modArithType),
           ModArithOperation::fromUnchecked(b, this->modArithType)));
  }

protected:
  Op *const op;
  Func fn;
};

template <typename Op, typename Func>
class GenericAdditiveModArithConstantFolder
    : public GenericBinaryModArithConstantFolder<
          AdditiveConstantFolderDelegate<ConstantFolderConfig>, Op, Func> {
public:
  using GenericBinaryModArithConstantFolder<
      AdditiveConstantFolderDelegate<ConstantFolderConfig>, Op,
      Func>::GenericBinaryModArithConstantFolder;

  bool isZero(const APInt &value) const final { return value.isZero(); }
};

template <typename Op, typename Func>
class GenericMultiplicativeModArithConstantFolder
    : public GenericBinaryModArithConstantFolder<
          MultiplicativeConstantFolderDelegate<ConstantFolderConfig>, Op,
          Func> {
public:
  GenericMultiplicativeModArithConstantFolder(Op *op, Func fn)
      : GenericBinaryModArithConstantFolder<
            MultiplicativeConstantFolderDelegate<ConstantFolderConfig>, Op,
            Func>(op, fn) {
    one = ModArithOperation(uint64_t{1}, this->modArithType);
  }

  bool isZero(const APInt &value) const final { return value.isZero(); }
  bool isOne(const APInt &value) const final { return value == one; }

private:
  APInt one;
};

template <typename Op, typename Func>
OpFoldResult foldAdditiveBinaryOp(Op *op, typename Op::FoldAdaptor adaptor,
                                  Func fn) {
  GenericAdditiveModArithConstantFolder<Op, Func> folder(op, fn);
  return BinaryConstantFolder<ConstantFolderConfig>::fold(adaptor, &folder);
}

template <typename Op, typename Func>
OpFoldResult
foldMultiplicativeBinaryOp(Op *op, typename Op::FoldAdaptor adaptor, Func fn) {
  GenericMultiplicativeModArithConstantFolder<Op, Func> folder(op, fn);
  return BinaryConstantFolder<ConstantFolderConfig>::fold(adaptor, &folder);
}

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

OpFoldResult ToMontOp::fold(FoldAdaptor adaptor) {
  auto stdType = getStandardFormType(getType());
  return foldUnaryOp(
      this, adaptor, [](const ModArithOperation &op) { return op.toMont(); },
      stdType);
}

OpFoldResult FromMontOp::fold(FoldAdaptor adaptor) {
  auto montType = getMontgomeryFormType(getType());
  return foldUnaryOp(
      this, adaptor, [](const ModArithOperation &op) { return op.fromMont(); },
      montType);
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
      : context(op->getType().getContext()), predicate(op->getPredicate()),
        modArithType(
            cast<ModArithType>(getElementTypeOrSelf(op->getLhs().getType()))) {}

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
    ModArithOperation aOp(a, modArithType);
    ModArithOperation bOp(b, modArithType);
    switch (predicate) {
    case arith::CmpIPredicate::eq:
      return aOp == bOp;
    case arith::CmpIPredicate::ne:
      return aOp != bOp;
    case arith::CmpIPredicate::ult:
      return aOp < bOp;
    case arith::CmpIPredicate::ule:
      return aOp <= bOp;
    case arith::CmpIPredicate::ugt:
      return aOp > bOp;
    case arith::CmpIPredicate::uge:
      return aOp >= bOp;
    default:
      llvm_unreachable("Unsupported comparison predicate");
    }
  }

private:
  MLIRContext *const context;
  arith::CmpIPredicate predicate;
  ModArithType modArithType;
};

} // namespace

OpFoldResult CmpOp::fold(FoldAdaptor adaptor) {
  CmpConstantFolder folder(this);
  return BinaryConstantFolder<CmpConstantFolderConfig>::fold(adaptor, &folder);
}

OpFoldResult NegateOp::fold(FoldAdaptor adaptor) {
  return foldUnaryOp(this, adaptor,
                     [](const ModArithOperation &op) { return -op; });
}

OpFoldResult DoubleOp::fold(FoldAdaptor adaptor) {
  return foldUnaryOp(this, adaptor,
                     [](const ModArithOperation &op) { return op.dbl(); });
}

OpFoldResult SquareOp::fold(FoldAdaptor adaptor) {
  return foldUnaryOp(this, adaptor,
                     [](const ModArithOperation &op) { return op.square(); });
}

OpFoldResult MontSquareOp::fold(FoldAdaptor adaptor) {
  return foldUnaryOp(this, adaptor,
                     [](const ModArithOperation &op) { return op.square(); });
}

OpFoldResult InverseOp::fold(FoldAdaptor adaptor) {
  return foldUnaryOp(this, adaptor,
                     [](const ModArithOperation &op) { return op.inverse(); });
}

OpFoldResult MontInverseOp::fold(FoldAdaptor adaptor) {
  return foldUnaryOp(this, adaptor,
                     [](const ModArithOperation &op) { return op.inverse(); });
}

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  return foldAdditiveBinaryOp(this, adaptor,
                              [](auto a, auto b) { return a + b; });
}

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  return foldAdditiveBinaryOp(this, adaptor,
                              [](auto a, auto b) { return a - b; });
}

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  return foldMultiplicativeBinaryOp(this, adaptor,
                                    [](auto a, auto b) { return a * b; });
}

OpFoldResult MontMulOp::fold(FoldAdaptor adaptor) {
  return foldMultiplicativeBinaryOp(this, adaptor,
                                    [](auto a, auto b) { return a * b; });
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
    auto modArithType = cast<ModArithType>(parsedType);
    auto valueAttr =
        IntegerAttr::get(modArithType.getStorageType(), parsedInts[0]);
    // Convert to Montgomery form if the type is Montgomery
    if (modArithType.isMontgomery()) {
      valueAttr = getAttrAsMontgomeryForm(modArithType.getModulus(), valueAttr);
    }
    result.addAttribute("value", valueAttr);
    result.addTypes(parsedType);
    return success();
  }

  parsedInts.pop_back();
  if (failed(parseModularIntegerList(parser, parsedInts, parsedType,
                                     getModulusCallback))) {
    return failure();
  }

  auto shapedType = cast<ShapedType>(parsedType);
  auto modArithType = cast<ModArithType>(shapedType.getElementType());
  DenseElementsAttr denseElementsAttr = DenseIntElementsAttr::get(
      shapedType.clone(modArithType.getStorageType()), parsedInts);
  // Convert to Montgomery form if the type is Montgomery
  if (modArithType.isMontgomery()) {
    denseElementsAttr =
        getAttrAsMontgomeryForm(modArithType.getModulus(), denseElementsAttr);
  }
  result.addAttribute("value", denseElementsAttr);
  result.addTypes(parsedType);
  return success();
}

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";

  // Print value in standard form for readability
  Type type = getType();
  auto modArithType = cast<ModArithType>(getElementTypeOrSelf(type));
  Attribute value = getValue();

  if (modArithType.isMontgomery()) {
    // Convert Montgomery form value to standard form for printing
    if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
      value = getAttrAsStandardForm(modArithType.getModulus(), intAttr);
    } else {
      value = getAttrAsStandardForm(modArithType.getModulus(),
                                    cast<DenseElementsAttr>(value));
    }
  }

  p.printAttributeWithoutType(value);
  p << " : ";
  p.printType(type);
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
      valueOp = valueOp.fromMont();
    }
    ModArithOperation offsetOp(APInt(intAttr.getValue().getBitWidth(), offset),
                               stdType);
    return valueOp == -offsetOp;
  }
  return false;
}

bool isEqualTo(Attribute attr, Value val, uint32_t offset) {
  IntegerAttr intAttr = dyn_cast_if_present<IntegerAttr>(attr);
  if (auto denseIntAttr = dyn_cast_if_present<SplatElementsAttr>(attr)) {
    intAttr = denseIntAttr.getSplatValue<IntegerAttr>();
  }
  if (intAttr) {
    auto modArithType = cast<ModArithType>(getElementTypeOrSelf(val.getType()));
    ModArithOperation valueOp(intAttr.getValue(), modArithType);
    ModArithType stdType = modArithType;
    if (modArithType.isMontgomery()) {
      stdType = cast<ModArithType>(getStandardFormType(modArithType));
      valueOp = valueOp.fromMont();
    }
    ModArithOperation offsetOp(APInt(intAttr.getValue().getBitWidth(), offset),
                               stdType);
    return valueOp == offsetOp;
  }
  return false;
}
} // namespace

namespace {
#include "prime_ir/Dialect/ModArith/IR/ModArithCanonicalization.cpp.inc"
}

#include "prime_ir/Utils/CanonicalizationPatterns.inc"

void AddOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
#define PRIME_IR_ADD_PATTERN(Name) patterns.add<ModArith##Name>(context);
  PRIME_IR_FIELD_ADD_PATTERN_LIST(PRIME_IR_ADD_PATTERN)
#undef PRIME_IR_ADD_PATTERN
}

void SubOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
#define PRIME_IR_SUB_PATTERN(Name) patterns.add<ModArith##Name>(context);
  PRIME_IR_FIELD_SUB_PATTERN_LIST(PRIME_IR_SUB_PATTERN)
#undef PRIME_IR_SUB_PATTERN
}

void MulOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
#define PRIME_IR_MUL_PATTERN(Name) patterns.add<ModArith##Name>(context);
  PRIME_IR_FIELD_MUL_PATTERN_LIST(PRIME_IR_MUL_PATTERN)
#undef PRIME_IR_MUL_PATTERN
}

} // namespace mlir::prime_ir::mod_arith
