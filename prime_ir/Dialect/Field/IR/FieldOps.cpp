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

#include "prime_ir/Dialect/Field/IR/FieldOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldOperation.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Utils/AssemblyFormatUtils.h"
#include "prime_ir/Utils/ConstantFolder.h"

// IWYU pragma: begin_keep
// Headers needed for FieldCanonicalization.cpp.inc
#include "mlir/IR/Matchers.h"
// IWYU pragma: end_keep
namespace mlir::prime_ir::field {

Type getStandardFormType(Type type) {
  Type standardType = getElementTypeOrSelf(type);
  if (auto pfType = dyn_cast<PrimeFieldType>(standardType)) {
    if (pfType.isMontgomery()) {
      standardType =
          PrimeFieldType::get(type.getContext(), pfType.getModulus());
    }
  } else if (auto extField =
                 dyn_cast<ExtensionFieldTypeInterface>(standardType)) {
    if (extField.isMontgomery()) {
      auto baseField = cast<PrimeFieldType>(extField.getBaseFieldType());
      auto pfType =
          PrimeFieldType::get(type.getContext(), baseField.getModulus());
      auto nonResidue = cast<IntegerAttr>(extField.getNonResidue());
      standardType = extField.cloneWith(
          pfType,
          mod_arith::getAttrAsStandardForm(baseField.getModulus(), nonResidue));
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
  } else if (auto extField = dyn_cast<ExtensionFieldTypeInterface>(montType)) {
    if (!extField.isMontgomery()) {
      auto baseField = cast<PrimeFieldType>(extField.getBaseFieldType());
      auto pfType =
          PrimeFieldType::get(type.getContext(), baseField.getModulus(), true);
      auto nonResidue = cast<IntegerAttr>(extField.getNonResidue());
      montType =
          extField.cloneWith(pfType, mod_arith::getAttrAsMontgomeryForm(
                                         baseField.getModulus(), nonResidue));
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

ParseResult parseFieldConstant(OpAsmParser &parser, OperationState &result) {
  // TODO(chokboole): support towers of extension fields
  SmallVector<APInt> parsedInts;
  Type parsedType;

  auto getModulusCallback = [&](APInt &modulus) -> ParseResult {
    auto elementType = getElementTypeOrSelf(parsedType);
    if (auto pfType = dyn_cast<PrimeFieldType>(elementType)) {
      modulus = pfType.getModulus().getValue();
      return success();
    } else if (auto extField =
                   dyn_cast<ExtensionFieldTypeInterface>(elementType)) {
      modulus = cast<PrimeFieldType>(extField.getBaseFieldType())
                    .getModulus()
                    .getValue();
      return success();
    }

    return parser.emitError(
        parser.getCurrentLocation(),
        "expected PrimeFieldType or ExtensionFieldTypeInterface");
  };

  auto parseResult = parseOptionalModularOrExtendedModularInteger(
      parser, parsedInts, parsedType, getModulusCallback);
  if (parseResult.has_value()) {
    if (failed(parseResult.value())) {
      return failure();
    }
    if (auto pfType = dyn_cast<PrimeFieldType>(parsedType)) {
      if (parsedInts.size() != 1) {
        return parser.emitError(parser.getCurrentLocation())
               << "prime field constant must have a single value, but got "
               << parsedInts.size();
      }
      result.addAttribute(
          "value", IntegerAttr::get(pfType.getStorageType(), parsedInts[0]));
    } else if (auto efType =
                   dyn_cast<ExtensionFieldTypeInterface>(parsedType)) {
      if (parsedInts.size() != efType.getDegreeOverPrime()) {
        return parser.emitError(parser.getCurrentLocation())
               << "extension field constant has " << parsedInts.size()
               << " coefficients, but expected " << efType.getDegreeOverPrime();
      }
      auto pfType = cast<PrimeFieldType>(efType.getBaseFieldType());
      result.addAttribute(
          "value", DenseElementsAttr::get(
                       RankedTensorType::get({efType.getDegreeOverPrime()},
                                             pfType.getStorageType()),
                       parsedInts));
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "unsupported type for constant: ")
             << parsedType;
    }
    result.addTypes(parsedType);
    return success();
  }

  if (failed(parseModularIntegerList(parser, parsedInts, parsedType,
                                     getModulusCallback))) {
    return failure();
  }

  auto shapedType = cast<ShapedType>(parsedType);
  auto elementType = getElementTypeOrSelf(parsedType);
  if (auto pfType = dyn_cast<PrimeFieldType>(elementType)) {
    auto denseElementsAttr = DenseIntElementsAttr::get(
        shapedType.clone(pfType.getStorageType()), parsedInts);
    result.addAttribute("value", denseElementsAttr);
    result.addTypes(parsedType);
    return success();
  }
  return parser.emitError(parser.getCurrentLocation(),
                          "dense attribute is only supported for shaped types "
                          "with a prime field element type");
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return adaptor.getValue();
}

// static
ConstantOp ConstantOp::materialize(OpBuilder &builder, Attribute value,
                                   Type type, Location loc) {
  if (!isa<PrimeFieldType>(getElementTypeOrSelf(type)) &&
      !isa<ExtensionFieldTypeInterface>(getElementTypeOrSelf(type))) {
    return nullptr;
  }

  if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
    return builder.create<ConstantOp>(loc, type, intAttr);
  } else if (auto denseElementsAttr = dyn_cast<DenseIntElementsAttr>(value)) {
    return builder.create<ConstantOp>(loc, type, denseElementsAttr);
  }
  return nullptr;
}

Operation *FieldDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  return ConstantOp::materialize(builder, value, type, loc);
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseFieldConstant(parser, result);
}

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getValue());
  p << " : ";
  p.printType(getType());
}

template <typename OpType>
static LogicalResult disallowShapedTypeOfExtField(OpType op) {
  // FIXME(batzor): In the prime field case, we rely on elementwise trait but in
  // the quadratic extension case, `linalg.generic` introduced by the
  // elementwise pass will be ill-formed due to the 1:N conversion.
  auto resultType = op.getType();
  if (isa<ShapedType>(resultType)) {
    auto elementType = cast<ShapedType>(resultType).getElementType();
    if (isa<QuadraticExtFieldType>(elementType)) {
      return op->emitOpError("shaped type is not supported for quadratic "
                             "extension field type");
    }
  }
  return success();
}

LogicalResult CmpOp::verify() {
  auto operandType = getElementTypeOrSelf(getLhs());
  if (isa<QuadraticExtFieldType>(operandType)) {
    arith::CmpIPredicate predicate = getPredicate();
    if (predicate == arith::CmpIPredicate::eq ||
        predicate == arith::CmpIPredicate::ne) {
      return success();
    } else {
      return emitOpError() << "only 'eq' and 'ne' comparisons are supported "
                              "for quadratic extension field type";
    }
  }
  return success();
}
LogicalResult FromMontOp::verify() {
  bool isMont = isMontgomery(getType());
  if (isMont) {
    return emitOpError()
           << "FromMontOp result should be a standard type, but got "
           << getElementTypeOrSelf(getType()) << ".";
  }
  return success();
}
LogicalResult ToMontOp::verify() {
  bool isMont = isMontgomery(getType());
  if (!isMont) {
    return emitOpError()
           << "ToMontOp result should be a Montgomery type, but got "
           << getElementTypeOrSelf(getType()) << ".";
  }
  return success();
}

bool BitcastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  Type inputType = getElementTypeOrSelf(inputs.front());
  Type outputType = getElementTypeOrSelf(outputs.front());

  return getIntOrPrimeFieldBitWidth(inputType) ==
         getIntOrPrimeFieldBitWidth(outputType);
}

LogicalResult ExtToCoeffsOp::verify() {
  Type inputType = getInput().getType();
  if (auto extField = dyn_cast<ExtensionFieldTypeInterface>(inputType)) {
    unsigned expected = extField.getDegreeOverBase();
    if (getOutput().size() == expected) {
      return success();
    }
    return emitOpError() << "expected " << expected
                         << " output types for extension field input, but got "
                         << getOutput().size();
  }
  return emitOpError() << "input type must be an extension field; got "
                       << inputType;
}

LogicalResult ExtFromCoeffsOp::verify() {
  Type outputType = getType();
  if (auto extField = dyn_cast<ExtensionFieldTypeInterface>(outputType)) {
    unsigned expected = extField.getDegreeOverBase();
    if (getInput().size() == expected) {
      return success();
    }
    return emitOpError() << "expected " << expected
                         << " input types for extension field output, but got "
                         << getInput().size();
  }
  return emitOpError() << "output type must be an extension field; got "
                       << outputType;
}

//===----------------------------------------------------------------------===//
// Prime Field Constant Folding
//===----------------------------------------------------------------------===//

namespace {

struct PrimeFieldConstantFolderConfig {
  using NativeInputType = APInt;
  using NativeOutputType = APInt;
  using ScalarAttr = IntegerAttr;
  using TensorAttr = DenseIntElementsAttr;
};

class UnaryPrimeFieldConstantFolder
    : public UnaryConstantFolder<PrimeFieldConstantFolderConfig>::Delegate {
public:
  explicit UnaryPrimeFieldConstantFolder(Type type)
      : primeFieldType(cast<PrimeFieldType>(getElementTypeOrSelf(type))) {}

  APInt getNativeInput(IntegerAttr attr) const final { return attr.getValue(); }

  OpFoldResult getScalarAttr(const APInt &value) const final {
    return IntegerAttr::get(primeFieldType.getStorageType(), value);
  }

  OpFoldResult getTensorAttr(ShapedType type,
                             ArrayRef<APInt> values) const final {
    return DenseIntElementsAttr::get(
        type.clone(primeFieldType.getStorageType()), values);
  }

protected:
  PrimeFieldType primeFieldType;
};

class PrimeFieldNegateConstantFolder : public UnaryPrimeFieldConstantFolder {
public:
  using UnaryPrimeFieldConstantFolder::UnaryPrimeFieldConstantFolder;

  APInt operate(const APInt &value) const final {
    return -PrimeFieldOperation::fromUnchecked(value, primeFieldType);
  }
};

class PrimeFieldDoubleConstantFolder : public UnaryPrimeFieldConstantFolder {
public:
  using UnaryPrimeFieldConstantFolder::UnaryPrimeFieldConstantFolder;

  APInt operate(const APInt &value) const final {
    return PrimeFieldOperation::fromUnchecked(value, primeFieldType).dbl();
  }
};

class PrimeFieldSquareConstantFolder : public UnaryPrimeFieldConstantFolder {
public:
  using UnaryPrimeFieldConstantFolder::UnaryPrimeFieldConstantFolder;

  APInt operate(const APInt &value) const final {
    return PrimeFieldOperation::fromUnchecked(value, primeFieldType).square();
  }
};

class PrimeFieldInverseConstantFolder : public UnaryPrimeFieldConstantFolder {
public:
  using UnaryPrimeFieldConstantFolder::UnaryPrimeFieldConstantFolder;

  APInt operate(const APInt &value) const final {
    return PrimeFieldOperation::fromUnchecked(value, primeFieldType).inverse();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Extension Field Constant Folding
//===----------------------------------------------------------------------===//

namespace {

// Config for extension field constant folder.
struct ExtensionFieldConstantFolderConfig {
  using NativeInputType = SmallVector<APInt>;
  using NativeOutputType = SmallVector<APInt>;
  using ScalarAttr = DenseIntElementsAttr;
};

// Base delegate for extension field unary operations.
class UnaryExtFieldConstantFolder
    : public ExtensionFieldUnaryConstantFolder<
          ExtensionFieldConstantFolderConfig>::Delegate {
public:
  explicit UnaryExtFieldConstantFolder(Type type)
      : extFieldType(cast<ExtensionFieldTypeInterface>(type)) {}

  SmallVector<APInt> getNativeInput(DenseIntElementsAttr attr) const final {
    auto values = attr.getValues<APInt>();
    return {values.begin(), values.end()};
  }

  OpFoldResult getScalarAttr(const SmallVector<APInt> &coeffs) const final {
    PrimeFieldType baseFieldType =
        cast<PrimeFieldType>(extFieldType.getBaseFieldType());
    auto tensorType = RankedTensorType::get(
        {static_cast<int64_t>(coeffs.size())}, baseFieldType.getStorageType());
    return DenseIntElementsAttr::get(tensorType, coeffs);
  }

protected:
  ExtensionFieldTypeInterface extFieldType;
};

// Negate: -[a₀, a₁, ...] = [-a₀, -a₁, ...]
template <size_t N>
class NegateConstantFolder : public UnaryExtFieldConstantFolder {
public:
  using UnaryExtFieldConstantFolder::UnaryExtFieldConstantFolder;

  std::optional<SmallVector<APInt>>
  operate(const SmallVector<APInt> &coeffs) const final {
    return (-ExtensionFieldOperation<N>(coeffs, extFieldType)).toAPInts();
  }
};

// Double: uses zk_dtypes for extension field doubling.
template <size_t N>
class DoubleConstantFolder : public UnaryExtFieldConstantFolder {
public:
  using UnaryExtFieldConstantFolder::UnaryExtFieldConstantFolder;

  std::optional<SmallVector<APInt>>
  operate(const SmallVector<APInt> &coeffs) const final {
    return ExtensionFieldOperation<N>(coeffs, extFieldType).Double().toAPInts();
  }
};

// Square: uses zk_dtypes for proper extension field squaring algorithm.
template <size_t N>
class SquareConstantFolder : public UnaryExtFieldConstantFolder {
public:
  using UnaryExtFieldConstantFolder::UnaryExtFieldConstantFolder;

  std::optional<SmallVector<APInt>>
  operate(const SmallVector<APInt> &coeffs) const final {
    return ExtensionFieldOperation<N>(coeffs, extFieldType).Square().toAPInts();
  }
};

// Inverse: uses zk_dtypes for Frobenius-based inversion.
template <size_t N>
class InverseConstantFolder : public UnaryExtFieldConstantFolder {
public:
  using UnaryExtFieldConstantFolder::UnaryExtFieldConstantFolder;

  std::optional<SmallVector<APInt>>
  operate(const SmallVector<APInt> &coeffs) const final {
    return ExtensionFieldOperation<N>(coeffs, extFieldType)
        .Inverse()
        .toAPInts();
  }
};

// Base delegate for extension field binary operations.
class BinaryExtFieldConstantFolder
    : public ExtensionFieldBinaryConstantFolder<
          ExtensionFieldConstantFolderConfig>::Delegate {
public:
  explicit BinaryExtFieldConstantFolder(Type type)
      : extFieldType(cast<ExtensionFieldTypeInterface>(type)),
        baseFieldType(cast<PrimeFieldType>(extFieldType.getBaseFieldType())) {}

  SmallVector<APInt> getNativeInput(DenseIntElementsAttr attr) const final {
    auto values = attr.getValues<APInt>();
    return {values.begin(), values.end()};
  }

  OpFoldResult getScalarAttr(const SmallVector<APInt> &coeffs) const final {
    auto tensorType = RankedTensorType::get(
        {static_cast<int64_t>(coeffs.size())}, baseFieldType.getStorageType());
    return DenseIntElementsAttr::get(tensorType, coeffs);
  }

protected:
  ExtensionFieldTypeInterface extFieldType;
  PrimeFieldType baseFieldType;
};

// Add: [a₀, a₁, ...] + [b₀, b₁, ...] = [a₀ + b₀, a₁ + b₁, ...]
template <size_t N>
class ExtFieldAddConstantFolder : public BinaryExtFieldConstantFolder {
public:
  using BinaryExtFieldConstantFolder::BinaryExtFieldConstantFolder;

  SmallVector<APInt> operate(const SmallVector<APInt> &lhs,
                             const SmallVector<APInt> &rhs) const final {
    return (ExtensionFieldOperation<N>(lhs, extFieldType) +
            ExtensionFieldOperation<N>(rhs, extFieldType))
        .toAPInts();
  }
};

// Sub: [a₀, a₁, ...] - [b₀, b₁, ...] = [a₀ - b₀, a₁ - b₁, ...]
template <size_t N>
class ExtFieldSubConstantFolder : public BinaryExtFieldConstantFolder {
public:
  using BinaryExtFieldConstantFolder::BinaryExtFieldConstantFolder;

  SmallVector<APInt> operate(const SmallVector<APInt> &lhs,
                             const SmallVector<APInt> &rhs) const final {
    return (ExtensionFieldOperation<N>(lhs, extFieldType) -
            ExtensionFieldOperation<N>(rhs, extFieldType))
        .toAPInts();
  }
};

// Mul: uses zk_dtypes for proper extension field multiplication algorithm.
template <size_t N>
class ExtFieldMulConstantFolder : public BinaryExtFieldConstantFolder {
public:
  using BinaryExtFieldConstantFolder::BinaryExtFieldConstantFolder;

  SmallVector<APInt> operate(const SmallVector<APInt> &lhs,
                             const SmallVector<APInt> &rhs) const final {
    return (ExtensionFieldOperation<N>(lhs, extFieldType) *
            ExtensionFieldOperation<N>(rhs, extFieldType))
        .toAPInts();
  }
};

// Helper to dispatch extension field folding by degree using fold expression.
// Eliminates repetitive if-else chains for degree 2, 3, 4.
template <template <typename> class ConstantFolderT,
          template <size_t> class FolderT, size_t N, typename Adaptor>
void tryFoldOne(Type type, size_t degree, Adaptor adaptor,
                OpFoldResult &result) {
  if (!result && degree == N) {
    FolderT<N> folder(type);
    result = ConstantFolderT<ExtensionFieldConstantFolderConfig>::fold(adaptor,
                                                                       &folder);
  }
}

template <template <typename> class ConstantFolderT,
          template <size_t> class FolderT, typename Adaptor, size_t... Ns>
OpFoldResult foldExtFieldImpl(Type type, size_t degree, Adaptor adaptor,
                              std::index_sequence<Ns...>) {
  OpFoldResult result;
  (tryFoldOne<ConstantFolderT, FolderT, Ns + kMinExtDegree>(type, degree,
                                                            adaptor, result),
   ...);
  return result;
}

template <template <size_t> class FolderT, typename Adaptor>
OpFoldResult foldExtField(Type type, Adaptor adaptor) {
  auto extFieldType = dyn_cast<ExtensionFieldTypeInterface>(type);
  if (!extFieldType)
    return {};
  return foldExtFieldImpl<ExtensionFieldUnaryConstantFolder, FolderT>(
      type, extFieldType.getDegreeOverBase(), adaptor,
      std::make_index_sequence<kNumExtDegrees>{});
}

template <template <size_t> class FolderT, typename Adaptor>
OpFoldResult foldExtFieldBinary(Type type, Adaptor adaptor) {
  auto extFieldType = dyn_cast<ExtensionFieldTypeInterface>(type);
  if (!extFieldType)
    return {};
  return foldExtFieldImpl<ExtensionFieldBinaryConstantFolder, FolderT>(
      type, extFieldType.getDegreeOverBase(), adaptor,
      std::make_index_sequence<kNumExtDegrees>{});
}

} // namespace

OpFoldResult NegateOp::fold(FoldAdaptor adaptor) {
  if (isa<PrimeFieldType>(getElementTypeOrSelf(getType()))) {
    PrimeFieldNegateConstantFolder folder(getType());
    return UnaryConstantFolder<PrimeFieldConstantFolderConfig>::fold(adaptor,
                                                                     &folder);
  }
  return foldExtField<NegateConstantFolder>(getType(), adaptor);
}

OpFoldResult DoubleOp::fold(FoldAdaptor adaptor) {
  if (isa<PrimeFieldType>(getElementTypeOrSelf(getType()))) {
    PrimeFieldDoubleConstantFolder folder(getType());
    return UnaryConstantFolder<PrimeFieldConstantFolderConfig>::fold(adaptor,
                                                                     &folder);
  }
  return foldExtField<DoubleConstantFolder>(getType(), adaptor);
}

OpFoldResult SquareOp::fold(FoldAdaptor adaptor) {
  if (isa<PrimeFieldType>(getElementTypeOrSelf(getType()))) {
    PrimeFieldSquareConstantFolder folder(getType());
    return UnaryConstantFolder<PrimeFieldConstantFolderConfig>::fold(adaptor,
                                                                     &folder);
  }
  return foldExtField<SquareConstantFolder>(getType(), adaptor);
}

OpFoldResult InverseOp::fold(FoldAdaptor adaptor) {
  if (isa<PrimeFieldType>(getElementTypeOrSelf(getType()))) {
    PrimeFieldInverseConstantFolder folder(getType());
    return UnaryConstantFolder<PrimeFieldConstantFolderConfig>::fold(adaptor,
                                                                     &folder);
  }
  return foldExtField<InverseConstantFolder>(getType(), adaptor);
}

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  return foldExtFieldBinary<ExtFieldAddConstantFolder>(getType(), adaptor);
}

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  return foldExtFieldBinary<ExtFieldSubConstantFolder>(getType(), adaptor);
}

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  return foldExtFieldBinary<ExtFieldMulConstantFolder>(getType(), adaptor);
}

//===----------------------------------------------------------------------===//
// CreateOp
//===----------------------------------------------------------------------===//

ParseResult CreateOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  if (parser.parseLSquare())
    return failure();

  if (parser.parseOptionalRSquare()) {
    do {
      OpAsmParser::UnresolvedOperand operand;
      if (parser.parseOperand(operand))
        return failure();
      operands.push_back(operand);
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRSquare())
      return failure();
  }

  Type resultType;
  if (parser.parseColonType(resultType))
    return failure();

  auto efType = dyn_cast<ExtensionFieldType>(resultType);
  if (!efType) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected extension field type");
  }

  Type coeffType = efType.getBaseField();
  if (parser.resolveOperands(operands, coeffType, result.operands))
    return failure();

  result.addTypes(resultType);
  return success();
}

void CreateOp::print(OpAsmPrinter &printer) {
  printer << " [";
  llvm::interleaveComma(getCoefficients(), printer,
                        [&](Value v) { printer << v; });
  printer << "] : " << getType();
}

LogicalResult CreateOp::verify() {
  auto efType = cast<ExtensionFieldType>(getType());
  unsigned degree = efType.getDegree();

  if (getCoefficients().size() != degree) {
    return emitOpError() << "expected " << degree
                         << " coefficients for extension field of degree "
                         << degree << ", but got " << getCoefficients().size();
  }

  PrimeFieldType baseField = efType.getBaseField();
  for (auto [idx, coeff] : llvm::enumerate(getCoefficients())) {
    auto coeffType = dyn_cast<PrimeFieldType>(coeff.getType());
    if (!coeffType || coeffType != baseField) {
      return emitOpError() << "coefficient " << idx << " has type "
                           << coeff.getType() << ", expected " << baseField;
    }
  }

  return success();
}

namespace {

bool isNegativeOf(Attribute attr, Value val, uint32_t offset) {
  IntegerAttr intAttr = dyn_cast_if_present<IntegerAttr>(attr);
  if (auto denseIntAttr = dyn_cast_if_present<SplatElementsAttr>(attr)) {
    intAttr = denseIntAttr.getSplatValue<IntegerAttr>();
  }
  if (intAttr) {
    auto primeFieldType =
        cast<PrimeFieldType>(getElementTypeOrSelf(val.getType()));
    PrimeFieldOperation valueOp(intAttr.getValue(), primeFieldType);
    PrimeFieldType stdType = primeFieldType;
    if (primeFieldType.isMontgomery()) {
      stdType = cast<PrimeFieldType>(getStandardFormType(primeFieldType));
      valueOp = valueOp.fromMont();
    }
    PrimeFieldOperation offsetOp(
        APInt(intAttr.getValue().getBitWidth(), offset), stdType);
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
    auto primeFieldType =
        cast<PrimeFieldType>(getElementTypeOrSelf(val.getType()));
    PrimeFieldOperation valueOp(intAttr.getValue(), primeFieldType);
    PrimeFieldType stdType = primeFieldType;
    if (primeFieldType.isMontgomery()) {
      stdType = cast<PrimeFieldType>(getStandardFormType(primeFieldType));
      valueOp = valueOp.fromMont();
    }
    PrimeFieldOperation offsetOp(
        APInt(intAttr.getValue().getBitWidth(), offset), stdType);
    return valueOp == offsetOp;
  }
  return false;
}

} // namespace

namespace {
#include "prime_ir/Dialect/Field/IR/FieldCanonicalization.cpp.inc"
}

#include "prime_ir/Utils/CanonicalizationPatterns.inc"

void AddOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
#define PRIME_IR_ADD_PATTERN(Name) patterns.add<Field##Name>(context);
  PRIME_IR_ADD_PATTERN_LIST(PRIME_IR_ADD_PATTERN)
#undef PRIME_IR_ADD_PATTERN
}

void SubOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
#define PRIME_IR_SUB_PATTERN(Name) patterns.add<Field##Name>(context);
  PRIME_IR_SUB_PATTERN_LIST(PRIME_IR_SUB_PATTERN)
#undef PRIME_IR_SUB_PATTERN
}

void MulOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
#define PRIME_IR_MUL_PATTERN(Name) patterns.add<Field##Name>(context);
  PRIME_IR_MUL_PATTERN_LIST(PRIME_IR_MUL_PATTERN)
#undef PRIME_IR_MUL_PATTERN
}

namespace {

struct ExtFromCoeffsOfExtToCoeffs
    : public mlir::OpRewritePattern<ExtFromCoeffsOp> {
  using OpRewritePattern<ExtFromCoeffsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtFromCoeffsOp op,
                                PatternRewriter &rewriter) const override {
    // Match: field.ext_from_coeffs(field.ext_to_coeffs(arg))
    if (op.getOperands().empty())
      return failure();

    auto extToCoeffsOp =
        op.getOperands().front().getDefiningOp<ExtToCoeffsOp>();
    if (!extToCoeffsOp)
      return failure();

    // The operands must be exactly the results of the ExtToCoeffsOp, in order.
    if (op.getOperands() != extToCoeffsOp->getResults())
      return failure();

    rewriter.replaceOp(op, extToCoeffsOp->getOperands());
    return success();
  }
};

struct ExtToCoeffsOfExtFromCoeffs
    : public mlir::OpRewritePattern<ExtToCoeffsOp> {
  using OpRewritePattern<ExtToCoeffsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtToCoeffsOp op,
                                PatternRewriter &rewriter) const override {
    // Match: field.ext_to_coeffs(field.ext_from_coeffs(arg...))
    auto extFromCoeffsOp = op.getOperand().getDefiningOp<ExtFromCoeffsOp>();
    if (!extFromCoeffsOp)
      return failure();

    rewriter.replaceOp(op, extFromCoeffsOp->getOperands());
    return success();
  }
};

} // namespace

void ExtFromCoeffsOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<ExtFromCoeffsOfExtToCoeffs>(context);
}

void ExtToCoeffsOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<ExtToCoeffsOfExtFromCoeffs>(context);
}

} // namespace mlir::prime_ir::field
