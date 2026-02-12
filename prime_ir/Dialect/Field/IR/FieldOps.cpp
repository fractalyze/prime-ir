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

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldOperation.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"
#include "prime_ir/Utils/AssemblyFormatUtils.h"
#include "prime_ir/Utils/BitcastOpUtils.h"
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
  } else if (auto extField = dyn_cast<ExtensionFieldType>(standardType)) {
    if (extField.isMontgomery()) {
      standardType =
          detail::convertExtFieldType<detail::MontDirection::FromMont>(
              extField);
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
  } else if (auto extField = dyn_cast<ExtensionFieldType>(montType)) {
    if (!extField.isMontgomery()) {
      montType =
          detail::convertExtFieldType<detail::MontDirection::ToMont>(extField);
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

namespace {

// Computes the attribute shape for an extension field type, reflecting the
// tower structure. For simple extensions like ef<2x PF>, returns {2}.
// For tower extensions like ef<3x ef<2x PF>>, returns {3, 2}.
SmallVector<int64_t> getExtFieldAttrShape(ExtensionFieldType efType) {
  SmallVector<int64_t> shape;
  Type current = efType;
  while (auto ef = dyn_cast<ExtensionFieldType>(current)) {
    shape.push_back(static_cast<int64_t>(ef.getDegree()));
    current = ef.getBaseField();
  }
  return shape;
}

} // namespace

// Helper to validate and create a field attribute using common utility logic.
// This reuses validateModularInteger to ensure consistent range checking.
FailureOr<Attribute> validateAndCreateFieldAttribute(OpAsmParser &parser,
                                                     Type type,
                                                     ArrayRef<APInt> values) {
  Type elementType = getElementTypeOrSelf(type);

  if (auto pfType = dyn_cast<PrimeFieldType>(elementType)) {
    if (values.size() != 1) {
      return parser.emitError(parser.getCurrentLocation())
             << "prime field constant must have a single value, but got "
             << values.size();
    }

    APInt val = values[0];
    APInt modulus = pfType.getModulus().getValue();

    // Delegate validation and bit-adjustment to the shared utility
    if (failed(prime_ir::validateModularInteger(parser, modulus, val))) {
      return failure();
    }

    return IntegerAttr::get(pfType.getStorageType(), val);
  } else if (auto efType = dyn_cast<ExtensionFieldType>(elementType)) {
    unsigned degree = efType.getDegreeOverPrime();
    if (values.size() != degree) {
      return parser.emitError(parser.getCurrentLocation())
             << "extension field constant has " << values.size()
             << " coefficients, but expected " << degree;
    }

    auto pfType = efType.getBasePrimeField();
    APInt modulus = pfType.getModulus().getValue();

    SmallVector<APInt> adjustedValues;
    adjustedValues.reserve(degree);

    for (const APInt &val : values) {
      APInt adjusted = val;
      // Delegate validation and bit-adjustment to the shared utility
      if (failed(prime_ir::validateModularInteger(parser, modulus, adjusted))) {
        return failure();
      }
      adjustedValues.push_back(adjusted);
    }

    auto towerShape = getExtFieldAttrShape(efType);
    return DenseElementsAttr::get(
        RankedTensorType::get(towerShape, pfType.getStorageType()),
        adjustedValues);
  }

  return parser.emitError(parser.getCurrentLocation(),
                          "unsupported type for constant creation: ")
         << type;
}

ParseResult parseFieldConstant(OpAsmParser &parser, OperationState &result) {
  SmallVector<APInt> parsedInts;
  Type parsedType;

  auto getModulusCallback = [&](APInt &modulus) -> ParseResult {
    auto elementType = getElementTypeOrSelf(parsedType);
    if (auto pfType = dyn_cast<PrimeFieldType>(elementType)) {
      modulus = pfType.getModulus().getValue();
      return success();
    } else if (auto bfType = dyn_cast<BinaryFieldType>(elementType)) {
      // Binary field: use 2^bitWidth as a "virtual modulus" for parsing
      // The value will be masked to valid range
      unsigned bitWidth = bfType.getBitWidth();
      modulus = APInt::getOneBitSet(bitWidth + 1, bitWidth);
      return success();
    } else if (auto extField = dyn_cast<ExtensionFieldType>(elementType)) {
      modulus = extField.getBasePrimeField().getModulus().getValue();
      return success();
    }

    return parser.emitError(
        parser.getCurrentLocation(),
        "expected PrimeFieldType, BinaryFieldType, or ExtensionFieldType");
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
      auto valueAttr = IntegerAttr::get(pfType.getStorageType(), parsedInts[0]);
      result.addAttribute("value", maybeToMontgomery(pfType, valueAttr));
    } else if (auto bfType = dyn_cast<BinaryFieldType>(parsedType)) {
      if (parsedInts.size() != 1) {
        return parser.emitError(parser.getCurrentLocation())
               << "binary field constant must have a single value, but got "
               << parsedInts.size();
      }
      // Mask to valid bit range
      APInt value = parsedInts[0].zextOrTrunc(bfType.getBitWidth());
      result.addAttribute("value",
                          IntegerAttr::get(bfType.getStorageType(), value));
    } else if (auto efType = dyn_cast<ExtensionFieldType>(parsedType)) {
      if (parsedInts.size() != efType.getDegreeOverPrime()) {
        return parser.emitError(parser.getCurrentLocation())
               << "extension field constant has " << parsedInts.size()
               << " coefficients, but expected " << efType.getDegreeOverPrime();
      }
      auto pfType = efType.getBasePrimeField();
      auto towerShape = getExtFieldAttrShape(efType);
      auto denseAttr = DenseElementsAttr::get(
          RankedTensorType::get(towerShape, pfType.getStorageType()),
          parsedInts);
      result.addAttribute("value", maybeToMontgomery(efType, denseAttr));
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "unsupported type for constant: ")
             << parsedType;
    }
    result.addTypes(parsedType);
    return success();
  }

  // Shape validation callback for extension field tensors.
  // For tensor<Nx!EF{d}>, the parsed shape should be [N, d] (tensor dims +
  // coefficient dim).
  auto shapeValidationCallback =
      [&](ArrayRef<int64_t> typeShape,
          ArrayRef<int64_t> parsedShape) -> ParseResult {
    auto elementType = getElementTypeOrSelf(parsedType);
    if (auto efType = dyn_cast<ExtensionFieldType>(elementType)) {
      // Accept both flat shape [tensorDims..., degreeOverPrime] and
      // tower shape [tensorDims..., towerSignature...].
      auto towerDims = getExtFieldAttrShape(efType);
      SmallVector<int64_t> expectedTower(typeShape);
      expectedTower.append(towerDims.begin(), towerDims.end());

      SmallVector<int64_t> expectedFlat(typeShape);
      expectedFlat.push_back(static_cast<int64_t>(efType.getDegreeOverPrime()));

      bool matchesTower = expectedTower.size() == parsedShape.size() &&
                          std::equal(expectedTower.begin(), expectedTower.end(),
                                     parsedShape.begin());
      bool matchesFlat = expectedFlat.size() == parsedShape.size() &&
                         std::equal(expectedFlat.begin(), expectedFlat.end(),
                                    parsedShape.begin());

      if (!matchesTower && !matchesFlat) {
        return parser.emitError(parser.getCurrentLocation(),
                                "extension field tensor constant shape [")
               << llvm::make_range(parsedShape.begin(), parsedShape.end())
               << "] does not match expected shape [tensor dims: "
               << llvm::make_range(typeShape.begin(), typeShape.end())
               << ", tower dims: "
               << llvm::make_range(towerDims.begin(), towerDims.end()) << "]";
      }
      return success();
    }
    // For prime fields, exact shape match.
    if (typeShape.size() != parsedShape.size() ||
        !std::equal(typeShape.begin(), typeShape.end(), parsedShape.begin())) {
      return parser.emitError(parser.getCurrentLocation(),
                              "tensor constant shape [")
             << llvm::make_range(parsedShape.begin(), parsedShape.end())
             << "] does not match type shape ["
             << llvm::make_range(typeShape.begin(), typeShape.end()) << "]";
    }
    return success();
  };

  if (failed(parseModularIntegerList(parser, parsedInts, parsedType,
                                     getModulusCallback,
                                     shapeValidationCallback))) {
    return failure();
  }

  auto shapedType = cast<ShapedType>(parsedType);
  auto elementType = getElementTypeOrSelf(parsedType);
  if (auto pfType = dyn_cast<PrimeFieldType>(elementType)) {
    auto denseElementsAttr = DenseIntElementsAttr::get(
        shapedType.clone(pfType.getStorageType()), parsedInts);
    result.addAttribute("value", maybeToMontgomery(pfType, denseElementsAttr));
    result.addTypes(parsedType);
    return success();
  }
  if (auto bfType = dyn_cast<BinaryFieldType>(elementType)) {
    // Adjust each APInt to the correct bitwidth for binary field storage
    SmallVector<APInt> adjustedInts;
    adjustedInts.reserve(parsedInts.size());
    for (const APInt &val : parsedInts) {
      adjustedInts.push_back(val.zextOrTrunc(bfType.getBitWidth()));
    }
    auto denseElementsAttr = DenseIntElementsAttr::get(
        shapedType.clone(bfType.getStorageType()), adjustedInts);
    result.addAttribute("value", denseElementsAttr);
    result.addTypes(parsedType);
    return success();
  }
  if (auto efType = dyn_cast<ExtensionFieldType>(elementType)) {
    auto pfType = efType.getBasePrimeField();
    // For tensor<Nx!EF{d}>, the attribute shape is [N, towerDims...].
    auto towerDims = getExtFieldAttrShape(efType);
    SmallVector<int64_t> attrShape(shapedType.getShape());
    attrShape.append(towerDims.begin(), towerDims.end());
    auto attrType = RankedTensorType::get(attrShape, pfType.getStorageType());
    auto denseElementsAttr = DenseIntElementsAttr::get(attrType, parsedInts);
    result.addAttribute("value", denseElementsAttr);
    result.addTypes(parsedType);
    return success();
  }
  return parser.emitError(parser.getCurrentLocation(),
                          "unsupported element type for dense constant");
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return adaptor.getValue();
}

// static
ConstantOp ConstantOp::materialize(OpBuilder &builder, Attribute value,
                                   Type type, Location loc) {
  auto elementType = getElementTypeOrSelf(type);
  if (!isa<PrimeFieldType, BinaryFieldType, ExtensionFieldType>(elementType)) {
    return nullptr;
  }

  if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
    return builder.create<ConstantOp>(loc, type, intAttr);
  }
  return builder.create<ConstantOp>(loc, type,
                                    cast<DenseIntElementsAttr>(value));
}

Operation *FieldDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  if (auto boolAttr = dyn_cast<BoolAttr>(value)) {
    return builder.create<arith::ConstantOp>(loc, boolAttr);
  } else if (auto denseElementsAttr = dyn_cast<DenseIntElementsAttr>(value)) {
    if (!isa<PrimeFieldType, BinaryFieldType, ExtensionFieldType>(
            getElementTypeOrSelf(type))) {
      // This could be a folding result of CmpOp.
      return builder.create<arith::ConstantOp>(loc, denseElementsAttr);
    }
  }
  return ConstantOp::materialize(builder, value, type, loc);
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseFieldConstant(parser, result);
}

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";

  Type type = getType();
  Type elementType = getElementTypeOrSelf(type);
  Attribute value = maybeToStandard(elementType, getValue());

  p.printAttributeWithoutType(value);
  p << " : ";
  p.printType(type);
}

LogicalResult CmpOp::verify() {
  auto operandType = getElementTypeOrSelf(getLhs());
  if (isa<ExtensionFieldType>(operandType)) {
    arith::CmpIPredicate predicate = getPredicate();
    if (predicate == arith::CmpIPredicate::eq ||
        predicate == arith::CmpIPredicate::ne) {
      return success();
    } else {
      return emitOpError() << "only 'eq' and 'ne' comparisons are supported "
                              "for extension field type";
    }
  }
  return success();
}

namespace {

struct CmpConstantFolderConfig {
  using NativeInputType = APInt;
  using NativeOutputType = bool;
  using ScalarAttr = IntegerAttr;
  using TensorAttr = DenseIntElementsAttr;
};

class PrimeFieldCmpConstantFolder
    : public BinaryConstantFolder<CmpConstantFolderConfig>::Delegate {
public:
  explicit PrimeFieldCmpConstantFolder(CmpOp *op)
      : context(op->getType().getContext()), predicate(op->getPredicate()),
        pfType(cast<PrimeFieldType>(getElementTypeOrSelf(op->getLhs()))) {}

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
    PrimeFieldOperation aOp(a, pfType);
    PrimeFieldOperation bOp(b, pfType);
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
  const arith::CmpIPredicate predicate;
  const PrimeFieldType pfType;
};

struct ExtFieldCmpConstantFolderConfig {
  using NativeInputType = SmallVector<APInt>;
  using NativeOutputType = bool;
  using ScalarAttr = DenseIntElementsAttr;
  using TensorAttr = DenseIntElementsAttr;
};

class ExtensionFieldCmpConstantFolder
    : public BinaryConstantFolder<ExtFieldCmpConstantFolderConfig>::Delegate {
public:
  explicit ExtensionFieldCmpConstantFolder(CmpOp *op)
      : context(op->getType().getContext()), predicate(op->getPredicate()),
        efType(cast<ExtensionFieldType>(getElementTypeOrSelf(op->getLhs()))) {}

  SmallVector<APInt> getNativeInput(DenseIntElementsAttr attr) const final {
    auto values = attr.getValues<APInt>();
    return {values.begin(), values.end()};
  }

  OpFoldResult getScalarAttr(const bool &value) const final {
    return BoolAttr::get(context, value);
  }

  OpFoldResult getTensorAttr(ShapedType type,
                             ArrayRef<bool> values) const final {
    return DenseIntElementsAttr::get(type.clone(IntegerType::get(context, 1)),
                                     values);
  }

  bool operate(const SmallVector<APInt> &a,
               const SmallVector<APInt> &b) const final {
    assert(a.size() == b.size());
    FieldOperation aOp = FieldOperation::fromUnchecked(a, efType);
    FieldOperation bOp = FieldOperation::fromUnchecked(b, efType);
    // For extension fields, only eq and ne are supported
    if (predicate == arith::CmpIPredicate::eq) {
      return aOp == bOp;
    } else if (predicate == arith::CmpIPredicate::ne) {
      return aOp != bOp;
    }
    llvm_unreachable("Unsupported comparison predicate");
  }

private:
  MLIRContext *const context;
  const arith::CmpIPredicate predicate;
  const ExtensionFieldType efType;
};

} // namespace

OpFoldResult CmpOp::fold(FoldAdaptor adaptor) {
  Type type = getLhs().getType();
  Type elemType = getElementTypeOrSelf(type);
  if (isa<PrimeFieldType>(elemType)) {
    PrimeFieldCmpConstantFolder folder(this);
    return BinaryConstantFolder<CmpConstantFolderConfig>::fold(adaptor,
                                                               &folder);
  }
  if (isa<ExtensionFieldType>(elemType)) {
    // TODO(chokobole): Tensor of extension field constant folding is not yet
    // supported.
    if (isa<ShapedType>(type)) {
      return {};
    }
    ExtensionFieldCmpConstantFolder folder(this);
    return BinaryConstantFolder<ExtFieldCmpConstantFolderConfig>::fold(adaptor,
                                                                       &folder);
  }
  return {};
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

bool isFieldLikeType(Type type) {
  return isa<PrimeFieldType, ExtensionFieldType, mod_arith::ModArithType,
             IntegerType>(type);
}

namespace {

// Helper function to compute the total number of prime field elements
// represented by a shaped type. IntegerType is treated as a single element
// (after mod-arith-to-arith lowering). For tower extensions, uses
// getDegreeOverPrime() to get the total degree.
std::optional<int64_t> getTotalPrimeFieldElements(ShapedType shapedType) {
  if (!shapedType.hasStaticShape()) {
    return std::nullopt;
  }

  Type elementType = shapedType.getElementType();
  int64_t numElements = shapedType.getNumElements();

  if (isa<PrimeFieldType, mod_arith::ModArithType, IntegerType>(elementType)) {
    return numElements;
  }
  if (auto efType = dyn_cast<ExtensionFieldType>(elementType)) {
    // Use getDegreeOverPrime() to handle tower extensions correctly
    return numElements * efType.getDegreeOverPrime();
  }
  return std::nullopt;
}

} // namespace

bool isTensorReinterpretBitcast(Type inputType, Type outputType) {
  auto inputShaped = dyn_cast<ShapedType>(inputType);
  auto outputShaped = dyn_cast<ShapedType>(outputType);

  if (!inputShaped || !outputShaped) {
    return false;
  }

  Type inputElementType = inputShaped.getElementType();
  Type outputElementType = outputShaped.getElementType();

  // Both element types must be field-like types (prime, extension, or
  // mod_arith)
  if (!isFieldLikeType(inputElementType) ||
      !isFieldLikeType(outputElementType)) {
    return false;
  }

  // At least one side must be an extension field for tensor reinterpret
  return isa<ExtensionFieldType>(inputElementType) ||
         isa<ExtensionFieldType>(outputElementType);
}

bool BitcastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  Type inputType = inputs.front();
  Type outputType = outputs.front();

  Type inputElementType = getElementTypeOrSelf(inputType);
  Type outputElementType = getElementTypeOrSelf(outputType);

  // Integer to integer is not allowed (use arith.bitcast), check this first
  if (areBothIntegerTypes(inputElementType, outputElementType)) {
    return false;
  }

  // Same type is allowed for verification (will be folded away by folder)
  if (isSameTypeBitcast(inputType, outputType)) {
    return true;
  }

  // Case 1: Tensor reinterpret bitcast (extension field <->
  // prime/mod_arith/integer field)
  if (isTensorReinterpretBitcast(inputType, outputType)) {
    auto inputShaped = cast<ShapedType>(inputType);
    auto outputShaped = cast<ShapedType>(outputType);

    // Calculate bitwidth for each element type
    unsigned inputBitWidth;
    if (auto efType = dyn_cast<ExtensionFieldType>(inputElementType)) {
      inputBitWidth = efType.getStorageBitWidth();
    } else if (auto maType =
                   dyn_cast<mod_arith::ModArithType>(inputElementType)) {
      inputBitWidth = mod_arith::getIntOrModArithBitWidth(maType);
    } else {
      inputBitWidth = getIntOrPrimeFieldBitWidth(inputElementType);
    }

    unsigned outputBitWidth;
    if (auto efType = dyn_cast<ExtensionFieldType>(outputElementType)) {
      outputBitWidth = efType.getStorageBitWidth();
    } else if (auto maType =
                   dyn_cast<mod_arith::ModArithType>(outputElementType)) {
      outputBitWidth = mod_arith::getIntOrModArithBitWidth(maType);
    } else {
      outputBitWidth = getIntOrPrimeFieldBitWidth(outputElementType);
    }

    // The total number of prime field elements must match
    auto inputTotal = getTotalPrimeFieldElements(inputShaped);
    auto outputTotal = getTotalPrimeFieldElements(outputShaped);
    if (!inputTotal || !outputTotal || *inputTotal != *outputTotal) {
      return false;
    }

    // Total bitwidth must match
    return inputBitWidth * inputShaped.getNumElements() ==
           outputBitWidth * outputShaped.getNumElements();
  }

  // Check shape compatibility
  if (failed(verifyCompatibleShape(inputType, outputType))) {
    return false;
  }

  // Case 2: prime field <-> prime field bitcast
  if (auto inputPF = dyn_cast<PrimeFieldType>(inputElementType)) {
    if (auto outputPF = dyn_cast<PrimeFieldType>(outputElementType)) {
      // Allow if bitwidths match (modulus can be different)
      return getIntOrPrimeFieldBitWidth(inputElementType) ==
             getIntOrPrimeFieldBitWidth(outputElementType);
    }
  }

  // Case 3: extension field <-> extension field bitcast
  // This handles both simple extensions (EF over PF) and tower extensions
  // (EF over EF over ... over PF)
  if (auto inputEF = dyn_cast<ExtensionFieldType>(inputElementType)) {
    if (auto outputEF = dyn_cast<ExtensionFieldType>(outputElementType)) {
      // Total degree over prime must match (handles tower extensions)
      if (inputEF.getDegreeOverPrime() != outputEF.getDegreeOverPrime()) {
        return false;
      }
      // Allow if base prime field bitwidths match
      return getIntOrPrimeFieldBitWidth(inputEF.getBasePrimeField()) ==
             getIntOrPrimeFieldBitWidth(outputEF.getBasePrimeField());
    }
  }

  // Case 3: Binary field <-> integer bitcast
  if (auto inputBF = dyn_cast<BinaryFieldType>(inputElementType)) {
    if (auto outputInt = dyn_cast<IntegerType>(outputElementType)) {
      // Allow if shapes match and bitwidths match
      if (auto inputShaped = dyn_cast<ShapedType>(inputType)) {
        auto outputShaped = dyn_cast<ShapedType>(outputType);
        if (!outputShaped ||
            inputShaped.getShape() != outputShaped.getShape()) {
          return false;
        }
      }
      return inputBF.getBitWidth() == outputInt.getWidth();
    }
  }
  if (auto inputInt = dyn_cast<IntegerType>(inputElementType)) {
    if (auto outputBF = dyn_cast<BinaryFieldType>(outputElementType)) {
      // Allow if shapes match and bitwidths match
      if (auto inputShaped = dyn_cast<ShapedType>(inputType)) {
        auto outputShaped = dyn_cast<ShapedType>(outputType);
        if (!outputShaped ||
            inputShaped.getShape() != outputShaped.getShape()) {
          return false;
        }
      }
      return inputInt.getWidth() == outputBF.getBitWidth();
    }
  }

  // Case 4: prime field <- > integer bitcast
  if (!isa<PrimeFieldType, IntegerType>(inputElementType) ||
      !isa<PrimeFieldType, IntegerType>(outputElementType)) {
    return false;
  }

  // Check bitwidth compatibility
  return getIntOrPrimeFieldBitWidth(inputElementType) ==
         getIntOrPrimeFieldBitWidth(outputElementType);
}

LogicalResult BitcastOp::verify() {
  Type inputType = getInput().getType();
  Type outputType = getOutput().getType();

  if (areCastCompatible(TypeRange{inputType}, TypeRange{outputType})) {
    return success();
  }

  // Provide detailed error messages for all failure cases

  Type inputElementType = getElementTypeOrSelf(inputType);
  Type outputElementType = getElementTypeOrSelf(outputType);

  if (areBothIntegerTypes(inputElementType, outputElementType)) {
    return emitOpError()
           << "integer to integer bitcast should use arith.bitcast";
  }

  // Case 1: Tensor reinterpret bitcast
  if (isTensorReinterpretBitcast(inputType, outputType)) {
    auto inputShaped = cast<ShapedType>(inputType);
    auto outputShaped = cast<ShapedType>(outputType);

    auto inputTotal = getTotalPrimeFieldElements(inputShaped);
    auto outputTotal = getTotalPrimeFieldElements(outputShaped);
    if (!inputTotal || !outputTotal) {
      return emitOpError()
             << "tensor reinterpret bitcast requires static shapes";
    }
    if (*inputTotal != *outputTotal) {
      return emitOpError() << "tensor reinterpret bitcast requires matching "
                              "total prime field "
                              "element count; input has "
                           << *inputTotal << " elements but output has "
                           << *outputTotal;
    }

    // Check bitwidth (handles tower extensions via getBasePrimeField and
    // getDegreeOverPrime)
    Type inputElementType = inputShaped.getElementType();
    Type outputElementType = outputShaped.getElementType();
    unsigned inputBitWidth;
    if (auto efType = dyn_cast<ExtensionFieldType>(inputElementType)) {
      inputBitWidth = efType.getStorageBitWidth();
    } else {
      inputBitWidth = getIntOrPrimeFieldBitWidth(inputElementType);
    }
    unsigned outputBitWidth;
    if (auto efType = dyn_cast<ExtensionFieldType>(outputElementType)) {
      outputBitWidth = efType.getStorageBitWidth();
    } else {
      outputBitWidth = getIntOrPrimeFieldBitWidth(outputElementType);
    }
    return emitOpError()
           << "tensor reinterpret bitcast requires matching total bitwidth; "
              "input has "
           << (inputBitWidth * inputShaped.getNumElements())
           << " bits but output has "
           << (outputBitWidth * outputShaped.getNumElements()) << " bits";
  }

  // Check shape compatibility for non-tensor-reinterpret cases
  if (failed(verifyCompatibleShape(inputType, outputType))) {
    return emitOpError()
           << "input and output shapes are incompatible for bitcast";
  }

  // Case 2: Prime field bitcast
  if (isa<PrimeFieldType>(inputElementType) &&
      isa<PrimeFieldType>(outputElementType)) {
    unsigned inputBitWidth = getIntOrPrimeFieldBitWidth(inputElementType);
    unsigned outputBitWidth = getIntOrPrimeFieldBitWidth(outputElementType);
    return emitOpError()
           << "prime field bitcast requires matching bitwidths, but input has "
           << inputBitWidth << " bits and output has " << outputBitWidth
           << " bits";
  }

  // Case 3: Extension field bitcast (including tower extensions)
  if (auto inputEF = dyn_cast<ExtensionFieldType>(inputElementType)) {
    if (auto outputEF = dyn_cast<ExtensionFieldType>(outputElementType)) {
      if (inputEF.getDegreeOverPrime() != outputEF.getDegreeOverPrime()) {
        return emitOpError()
               << "extension field bitcast requires matching total degrees "
                  "over prime, but input has degree "
               << inputEF.getDegreeOverPrime() << " and output has degree "
               << outputEF.getDegreeOverPrime();
      }
      unsigned inputBitWidth =
          getIntOrPrimeFieldBitWidth(inputEF.getBasePrimeField());
      unsigned outputBitWidth =
          getIntOrPrimeFieldBitWidth(outputEF.getBasePrimeField());
      return emitOpError() << "extension field bitcast requires matching base "
                              "prime field bitwidths, but input has "
                           << inputBitWidth << " bits and output has "
                           << outputBitWidth << " bits";
    }
  }

  // Case 4: Scalar/element-wise bitcast (prime field <-> integer)
  if (!isa<PrimeFieldType, IntegerType>(inputElementType) ||
      !isa<PrimeFieldType, IntegerType>(outputElementType)) {
    return emitOpError()
           << "bitcast requires field or integer types, but got input type '"
           << inputType << "' and output type '" << outputType << "'";
  }

  unsigned inputBitWidth = getIntOrPrimeFieldBitWidth(inputElementType);
  unsigned outputBitWidth = getIntOrPrimeFieldBitWidth(outputElementType);
  if (inputBitWidth != outputBitWidth) {
    return emitOpError()
           << "bitcast requires matching bitwidths, but input has "
           << inputBitWidth << " bits and output has " << outputBitWidth
           << " bits";
  }

  return emitOpError()
         << "internal error: operands are cast-incompatible for unknown reason "
         << "(input: " << inputType << ", output: " << outputType << ")";
}

OpFoldResult BitcastOp::fold(FoldAdaptor adaptor) {
  return foldBitcast(*this, adaptor);
}

LogicalResult BitcastOp::canonicalize(BitcastOp op, PatternRewriter &rewriter) {
  return canonicalizeBitcast(op, rewriter);
}

LogicalResult ExtToCoeffsOp::verify() {
  Type inputType = getInput().getType();
  if (auto extField = dyn_cast<ExtensionFieldType>(inputType)) {
    unsigned expected = extField.getDegree();
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
  auto efType = dyn_cast<ExtensionFieldType>(outputType);
  if (!efType) {
    return emitOpError() << "output type must be an extension field; got "
                         << outputType;
  }

  unsigned expected = efType.getDegree();
  if (getInput().size() != expected) {
    return emitOpError() << "expected " << expected
                         << " input types for extension field output, but got "
                         << getInput().size();
  }

  // Validate PrimeFieldType coefficients match base field
  Type baseFieldType = efType.getBaseField();
  auto pfType = dyn_cast<PrimeFieldType>(baseFieldType);
  if (!pfType) {
    // TODO(junbeomlee): Add coefficient type validation for tower extensions
    return success();
  }

  for (auto [idx, coeff] : llvm::enumerate(getInput())) {
    auto coeffPfType = dyn_cast<PrimeFieldType>(coeff.getType());
    if (coeffPfType && coeffPfType != pfType) {
      return emitOpError() << "coefficient " << idx << " has type "
                           << coeff.getType() << ", expected " << pfType;
    }
  }

  return success();
}

namespace {

//===----------------------------------------------------------------------===//
// Mixins & Configs
//===----------------------------------------------------------------------===//

struct PrimeFieldConstantFolderConfig {
  using NativeInputType = APInt;
  using NativeOutputType = APInt;
  using ScalarAttr = IntegerAttr;
  using TensorAttr = DenseIntElementsAttr;
};

struct ExtensionFieldConstantFolderConfig {
  using NativeInputType = SmallVector<APInt>;
  using NativeOutputType = SmallVector<APInt>;
  using ScalarAttr = DenseIntElementsAttr;
  using TensorAttr = DenseIntElementsAttr;
};

struct BinaryFieldConstantFolderConfig {
  using NativeInputType = APInt;
  using NativeOutputType = APInt;
  using ScalarAttr = IntegerAttr;
  using TensorAttr = DenseIntElementsAttr;
};

template <typename BaseDelegate>
class PrimeFieldFolderMixin : public BaseDelegate {
public:
  explicit PrimeFieldFolderMixin(Type type)
      : pfType(cast<PrimeFieldType>(getElementTypeOrSelf(type))) {}

  APInt getNativeInput(IntegerAttr attr) const final { return attr.getValue(); }

  OpFoldResult getScalarAttr(const APInt &value) const final {
    return IntegerAttr::get(pfType.getStorageType(), value);
  }

  OpFoldResult getTensorAttr(ShapedType type,
                             ArrayRef<APInt> values) const final {
    return DenseIntElementsAttr::get(type.clone(pfType.getStorageType()),
                                     values);
  }

protected:
  const PrimeFieldType pfType;
};

template <typename BaseDelegate>
class ExtensionFieldFolderMixin : public BaseDelegate {
public:
  explicit ExtensionFieldFolderMixin(Type type)
      : type(type),
        efType(cast<ExtensionFieldType>(getElementTypeOrSelf(type))) {}

  SmallVector<APInt> getNativeInput(DenseIntElementsAttr attr) const final {
    auto values = attr.getValues<APInt>();
    return {values.begin(), values.end()};
  }

  OpFoldResult getScalarAttr(const SmallVector<APInt> &coeffs) const final {
    PrimeFieldType baseFieldType = efType.getBasePrimeField();
    auto towerShape = getExtFieldAttrShape(efType);
    auto tensorType =
        RankedTensorType::get(towerShape, baseFieldType.getStorageType());
    return DenseIntElementsAttr::get(tensorType, coeffs);
  }

  OpFoldResult getTensorAttr(ShapedType type,
                             ArrayRef<SmallVector<APInt>> values) const final {
    // Flatten all coefficient vectors into a single vector
    SmallVector<APInt> flattenedValues;
    for (const auto &coeffs : values) {
      flattenedValues.append(coeffs.begin(), coeffs.end());
    }
    // Create result attribute with shape [tensor_dims..., towerDims...]
    PrimeFieldType pfType = efType.getBasePrimeField();
    auto towerDims = getExtFieldAttrShape(efType);
    SmallVector<int64_t> attrShape(type.getShape());
    attrShape.append(towerDims.begin(), towerDims.end());
    auto attrType = RankedTensorType::get(attrShape, pfType.getStorageType());
    return DenseIntElementsAttr::get(attrType, flattenedValues);
  }

protected:
  const Type type;
  const ExtensionFieldType efType;
};

template <typename BaseDelegate>
class BinaryFieldFolderMixin : public BaseDelegate {
public:
  explicit BinaryFieldFolderMixin(Type type)
      : bfType(cast<BinaryFieldType>(getElementTypeOrSelf(type))) {}

  APInt getNativeInput(IntegerAttr attr) const final { return attr.getValue(); }

  OpFoldResult getScalarAttr(const APInt &value) const final {
    return IntegerAttr::get(bfType.getStorageType(), value);
  }

  OpFoldResult getTensorAttr(ShapedType type,
                             ArrayRef<APInt> values) const final {
    return DenseIntElementsAttr::get(type.clone(bfType.getStorageType()),
                                     values);
  }

protected:
  const BinaryFieldType bfType;
};

//===----------------------------------------------------------------------===//
// Generic Folders (Prime & Extension Unified Logic)
//===----------------------------------------------------------------------===//

template <typename Func>
class GenericUnaryPrimeFieldFolder
    : public PrimeFieldFolderMixin<
          UnaryConstantFolder<PrimeFieldConstantFolderConfig>::Delegate> {
public:
  GenericUnaryPrimeFieldFolder(Type type, Func fn, Type inputType = nullptr)
      : PrimeFieldFolderMixin(type), fn(fn),
        inputPfType(inputType
                        ? cast<PrimeFieldType>(getElementTypeOrSelf(inputType))
                        : this->pfType) {}

  APInt operate(const APInt &value) const final {
    return static_cast<APInt>(
        fn(FieldOperation::fromUnchecked(value, inputPfType)));
  }

private:
  const Func fn;
  const PrimeFieldType inputPfType;
};

template <typename Func>
class GenericUnaryExtFieldFolder
    : public ExtensionFieldFolderMixin<
          UnaryConstantFolder<ExtensionFieldConstantFolderConfig>::Delegate> {
public:
  GenericUnaryExtFieldFolder(Type type, Func fn, Type inputType = nullptr)
      : ExtensionFieldFolderMixin(type), fn(fn),
        inputEfType(inputType ? cast<ExtensionFieldType>(
                                    getElementTypeOrSelf(inputType))
                              : this->efType) {}

  SmallVector<APInt> operate(const SmallVector<APInt> &coeffs) const final {
    return static_cast<SmallVector<APInt>>(
        fn(FieldOperation::fromUnchecked(coeffs, inputEfType)));
  }

  // Override foldScalar to dispatch to foldTensor when dealing with tensors.
  // This is needed because for extension fields, ScalarAttr == TensorAttr ==
  // DenseIntElementsAttr, so the generic fold() always calls foldScalar.
  OpFoldResult foldScalar(DenseIntElementsAttr attr) const override {
    if (isa<ShapedType>(this->type)) {
      return foldTensor(attr);
    }
    // Actual scalar folding: use default implementation
    return this->getScalarAttr(operate(this->getNativeInput(attr)));
  }

  // Override foldTensor to handle extension field tensor constants properly.
  OpFoldResult foldTensor(DenseIntElementsAttr attr) const override {
    unsigned degree = this->efType.getDegreeOverPrime();
    auto inputValues = attr.getValues<APInt>();
    size_t numElements = inputValues.size();

    if (numElements % degree != 0) {
      return {};
    }

    SmallVector<SmallVector<APInt>> results;
    results.reserve(numElements / degree);

    for (size_t i = 0; i < numElements; i += degree) {
      SmallVector<APInt> coeffs(inputValues.begin() + i,
                                inputValues.begin() + i + degree);
      results.push_back(operate(coeffs));
    }

    auto shapedType = dyn_cast<ShapedType>(this->type);
    if (!shapedType) {
      return {};
    }
    return this->getTensorAttr(shapedType, results);
  }

private:
  const Func fn;
  const ExtensionFieldType inputEfType;
};

template <typename Func>
class GenericUnaryBinaryFieldFolder
    : public BinaryFieldFolderMixin<
          UnaryConstantFolder<BinaryFieldConstantFolderConfig>::Delegate> {
public:
  GenericUnaryBinaryFieldFolder(Type type, Func fn)
      : BinaryFieldFolderMixin(type), fn(fn) {}

  APInt operate(const APInt &value) const final {
    return static_cast<APInt>(
        fn(FieldOperation::fromUnchecked(value, this->bfType)));
  }

private:
  const Func fn;
};

template <typename BaseDelegate, typename Op, typename Func>
class GenericBinaryPrimeFieldConstantFolder
    : public PrimeFieldFolderMixin<BaseDelegate> {
public:
  GenericBinaryPrimeFieldConstantFolder(Op *op, Func fn)
      : PrimeFieldFolderMixin<BaseDelegate>(op->getType()), op(op), fn(fn) {}

  OpFoldResult getLhs() const final { return op->getLhs(); }

  APInt operate(const APInt &a, const APInt &b) const final {
    return static_cast<APInt>(
        fn(FieldOperation::fromUnchecked(a, this->pfType),
           FieldOperation::fromUnchecked(b, this->pfType)));
  }

protected:
  Op *const op;
  const Func fn;
};

template <typename BaseDelegate, typename Op, typename Func>
class GenericBinaryExtFieldConstantFolder
    : public ExtensionFieldFolderMixin<BaseDelegate> {
public:
  GenericBinaryExtFieldConstantFolder(Op *op, Func fn)
      : ExtensionFieldFolderMixin<BaseDelegate>(op->getType()), op(op), fn(fn) {
  }

  OpFoldResult getLhs() const final { return op->getLhs(); }

  SmallVector<APInt> operate(const SmallVector<APInt> &a,
                             const SmallVector<APInt> &b) const final {
    return static_cast<SmallVector<APInt>>(
        fn(FieldOperation::fromUnchecked(a, this->efType),
           FieldOperation::fromUnchecked(b, this->efType)));
  }

  // Override foldScalar to dispatch to foldTensor when dealing with tensors.
  // This is needed because for extension fields, ScalarAttr == TensorAttr ==
  // DenseIntElementsAttr, so the generic fold() always calls foldScalar.
  OpFoldResult foldScalar(DenseIntElementsAttr lhs,
                          DenseIntElementsAttr rhs) const override {
    if (isa<ShapedType>(this->type)) {
      return foldTensor(lhs, rhs);
    }
    // Actual scalar folding: use default implementation
    return this->getScalarAttr(
        operate(this->getNativeInput(lhs), this->getNativeInput(rhs)));
  }

  // Override single-arg foldScalar to dispatch to foldTensor when dealing with
  // tensors. This handles special case optimizations (x+0, x*1, etc.)
  OpFoldResult foldScalar(DenseIntElementsAttr rhs) const override {
    if (isa<ShapedType>(this->type)) {
      return foldTensor(rhs);
    }
    // Actual scalar folding: delegate to base class
    return BaseDelegate::foldScalar(rhs);
  }

  // Override foldTensor to handle extension field tensor constants properly.
  OpFoldResult foldTensor(DenseIntElementsAttr lhsAttr,
                          DenseIntElementsAttr rhsAttr) const override {
    unsigned degree = this->efType.getDegreeOverPrime();
    auto lhsValues = lhsAttr.getValues<APInt>();
    auto rhsValues = rhsAttr.getValues<APInt>();
    size_t numElements = lhsValues.size();

    if (numElements != rhsValues.size() || numElements % degree != 0) {
      return {};
    }

    SmallVector<SmallVector<APInt>> results;
    results.reserve(numElements / degree);

    for (size_t i = 0; i < numElements; i += degree) {
      SmallVector<APInt> lhsCoeffs(lhsValues.begin() + i,
                                   lhsValues.begin() + i + degree);
      SmallVector<APInt> rhsCoeffs(rhsValues.begin() + i,
                                   rhsValues.begin() + i + degree);
      results.push_back(operate(lhsCoeffs, rhsCoeffs));
    }

    auto shapedType = dyn_cast<ShapedType>(this->type);
    if (!shapedType) {
      return {};
    }
    return this->getTensorAttr(shapedType, results);
  }

  // Override single-arg foldTensor for special case optimizations.
  // Subclasses (Additive/Multiplicative) will override with specific logic.
  OpFoldResult foldTensor(DenseIntElementsAttr rhs) const override {
    return {};
  }

protected:
  Op *const op;
  const Func fn;
};

template <typename BaseDelegate, typename Op, typename Func>
class GenericBinaryBinaryFieldConstantFolder
    : public BinaryFieldFolderMixin<BaseDelegate> {
public:
  GenericBinaryBinaryFieldConstantFolder(Op *op, Func fn)
      : BinaryFieldFolderMixin<BaseDelegate>(op->getType()), op(op), fn(fn) {}

  OpFoldResult getLhs() const final { return op->getLhs(); }

  APInt operate(const APInt &a, const APInt &b) const final {
    return static_cast<APInt>(
        fn(FieldOperation::fromUnchecked(a, this->bfType),
           FieldOperation::fromUnchecked(b, this->bfType)));
  }

protected:
  Op *const op;
  const Func fn;
};

//===----------------------------------------------------------------------===//
// Specific Implementations (Additive / Multiplicative)
//===----------------------------------------------------------------------===//

template <typename Op, typename Func>
class PrimeAdditiveFolder
    : public GenericBinaryPrimeFieldConstantFolder<
          AdditiveConstantFolderDelegate<PrimeFieldConstantFolderConfig>, Op,
          Func> {
public:
  using GenericBinaryPrimeFieldConstantFolder<
      AdditiveConstantFolderDelegate<PrimeFieldConstantFolderConfig>, Op,
      Func>::GenericBinaryPrimeFieldConstantFolder;

  bool isZero(const APInt &value) const final { return value.isZero(); }
};

template <typename Op, typename Func>
class PrimeMultiplicativeFolder
    : public GenericBinaryPrimeFieldConstantFolder<
          MultiplicativeConstantFolderDelegate<PrimeFieldConstantFolderConfig>,
          Op, Func> {
public:
  PrimeMultiplicativeFolder(Op *op, Func fn)
      : GenericBinaryPrimeFieldConstantFolder<
            MultiplicativeConstantFolderDelegate<
                PrimeFieldConstantFolderConfig>,
            Op, Func>(op, fn),
        one(FieldOperation(uint64_t{1}, this->pfType)) {}

  bool isZero(const APInt &value) const final { return value.isZero(); }
  bool isOne(const APInt &value) const final { return value == one; }

private:
  const APInt one;
};

template <typename Op, typename Func>
class ExtAdditiveFolder
    : public GenericBinaryExtFieldConstantFolder<
          AdditiveConstantFolderDelegate<ExtensionFieldConstantFolderConfig>,
          Op, Func> {
public:
  using GenericBinaryExtFieldConstantFolder<
      AdditiveConstantFolderDelegate<ExtensionFieldConstantFolderConfig>, Op,
      Func>::GenericBinaryExtFieldConstantFolder;
  // Make base class foldTensor overloads visible
  using GenericBinaryExtFieldConstantFolder<
      AdditiveConstantFolderDelegate<ExtensionFieldConstantFolderConfig>, Op,
      Func>::foldTensor;

  bool isZero(const SmallVector<APInt> &value) const final {
    return llvm::all_of(value, [](const APInt &v) { return v.isZero(); });
  }

  // Fold additive identity: x + 0 = x
  OpFoldResult foldTensor(DenseIntElementsAttr rhs) const override {
    auto rhsValues = rhs.getValues<APInt>();
    // If all elements are zero, return lhs (x + 0 = x)
    // NOLINTNEXTLINE(whitespace/newline)
    if (llvm::all_of(rhsValues, [](const APInt &v) { return v.isZero(); })) {
      return this->op->getOperand(0);
    }
    return {};
  }
};

template <typename Op, typename Func>
class ExtMultiplicativeFolder : public GenericBinaryExtFieldConstantFolder<
                                    MultiplicativeConstantFolderDelegate<
                                        ExtensionFieldConstantFolderConfig>,
                                    Op, Func> {
public:
  // Make base class foldTensor overloads visible
  using GenericBinaryExtFieldConstantFolder<
      MultiplicativeConstantFolderDelegate<ExtensionFieldConstantFolderConfig>,
      Op, Func>::foldTensor;

  ExtMultiplicativeFolder(Op *op, Func fn)
      : GenericBinaryExtFieldConstantFolder<
            MultiplicativeConstantFolderDelegate<
                ExtensionFieldConstantFolderConfig>,
            Op, Func>(op, fn),
        one(static_cast<SmallVector<APInt>>(
            FieldOperation(uint64_t{1}, this->efType))) {}

  bool isZero(const SmallVector<APInt> &value) const final {
    return llvm::all_of(value, [](const APInt &v) { return v.isZero(); });
  }
  bool isOne(const SmallVector<APInt> &value) const final {
    return value == one;
  }

  // Fold multiplicative identities: x * 0 = 0, x * 1 = x
  OpFoldResult foldTensor(DenseIntElementsAttr rhs) const override {
    auto rhsValues = rhs.getValues<APInt>();
    unsigned degree = this->efType.getDegreeOverPrime();
    size_t numElements = rhsValues.size();

    if (numElements % degree != 0) {
      return {};
    }

    // Check if all extension field elements are zero or one
    bool allZeros = true;
    bool allOnes = true;

    for (size_t i = 0; i < numElements; i += degree) {
      SmallVector<APInt> coeffs(rhsValues.begin() + i,
                                rhsValues.begin() + i + degree);
      if (!isZero(coeffs)) {
        allZeros = false;
      }
      if (!isOne(coeffs)) {
        allOnes = false;
      }
      if (!allZeros && !allOnes) {
        break; // Early exit if neither condition holds
      }
    }

    // x * 0 = 0, return rhs
    if (allZeros) {
      return this->op->getOperand(1);
    }
    // x * 1 = x, return lhs
    if (allOnes) {
      return this->op->getOperand(0);
    }

    return {};
  }

private:
  const SmallVector<APInt> one;
};

template <typename Op, typename Func>
class BinaryFieldAdditiveFolder
    : public GenericBinaryBinaryFieldConstantFolder<
          AdditiveConstantFolderDelegate<BinaryFieldConstantFolderConfig>, Op,
          Func> {
public:
  using GenericBinaryBinaryFieldConstantFolder<
      AdditiveConstantFolderDelegate<BinaryFieldConstantFolderConfig>, Op,
      Func>::GenericBinaryBinaryFieldConstantFolder;

  bool isZero(const APInt &value) const final { return value.isZero(); }
};

template <typename Op, typename Func>
class BinaryFieldMultiplicativeFolder
    : public GenericBinaryBinaryFieldConstantFolder<
          MultiplicativeConstantFolderDelegate<BinaryFieldConstantFolderConfig>,
          Op, Func> {
public:
  BinaryFieldMultiplicativeFolder(Op *op, Func fn)
      : GenericBinaryBinaryFieldConstantFolder<
            MultiplicativeConstantFolderDelegate<
                BinaryFieldConstantFolderConfig>,
            Op, Func>(op, fn),
        one(static_cast<APInt>(FieldOperation(uint64_t{1}, this->bfType))) {}

  bool isZero(const APInt &value) const final { return value.isZero(); }
  bool isOne(const APInt &value) const final { return value == one; }

private:
  const APInt one;
};

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

template <typename Op, typename Func>
OpFoldResult foldUnaryOp(Op *op, typename Op::FoldAdaptor adaptor, Func fn,
                         Type inputType = nullptr) {
  Type type = op->getType();
  Type elemType = getElementTypeOrSelf(type);

  return llvm::TypeSwitch<Type, OpFoldResult>(elemType)
      .Case<PrimeFieldType>([&](auto) {
        GenericUnaryPrimeFieldFolder<Func> folder(type, fn, inputType);
        return UnaryConstantFolder<PrimeFieldConstantFolderConfig>::fold(
            adaptor, &folder);
      })
      .template Case<BinaryFieldType>([&](auto) {
        GenericUnaryBinaryFieldFolder<Func> folder(type, fn);
        return UnaryConstantFolder<BinaryFieldConstantFolderConfig>::fold(
            adaptor, &folder);
      })
      .template Case<ExtensionFieldType>([&](auto) {
        GenericUnaryExtFieldFolder<Func> folder(type, fn, inputType);
        return UnaryConstantFolder<ExtensionFieldConstantFolderConfig>::fold(
            adaptor, &folder);
      })
      // NOLINTNEXTLINE(whitespace/newline)
      .Default([](auto) { return OpFoldResult{}; });
}

template <typename Op, typename Func>
OpFoldResult foldAdditiveBinaryOp(Op *op, typename Op::FoldAdaptor adaptor,
                                  Func fn) {
  Type type = op->getType();
  Type elemType = getElementTypeOrSelf(type);

  return llvm::TypeSwitch<Type, OpFoldResult>(elemType)
      .Case<PrimeFieldType>([&](auto) {
        PrimeAdditiveFolder<Op, Func> folder(op, fn);
        return BinaryConstantFolder<PrimeFieldConstantFolderConfig>::fold(
            adaptor, &folder);
      })
      .template Case<BinaryFieldType>([&](auto) {
        BinaryFieldAdditiveFolder<Op, Func> folder(op, fn);
        return BinaryConstantFolder<BinaryFieldConstantFolderConfig>::fold(
            adaptor, &folder);
      })
      .template Case<ExtensionFieldType>([&](auto) {
        ExtAdditiveFolder<Op, Func> folder(op, fn);
        return BinaryConstantFolder<ExtensionFieldConstantFolderConfig>::fold(
            adaptor, &folder);
      })
      // NOLINTNEXTLINE(whitespace/newline)
      .Default([](auto) { return OpFoldResult{}; });
}

template <typename Op, typename Func>
OpFoldResult
foldMultiplicativeBinaryOp(Op *op, typename Op::FoldAdaptor adaptor, Func fn) {
  Type type = op->getType();
  Type elemType = getElementTypeOrSelf(type);

  return llvm::TypeSwitch<Type, OpFoldResult>(elemType)
      .Case<PrimeFieldType>([&](auto) {
        PrimeMultiplicativeFolder<Op, Func> folder(op, fn);
        return BinaryConstantFolder<PrimeFieldConstantFolderConfig>::fold(
            adaptor, &folder);
      })
      .template Case<BinaryFieldType>([&](auto) {
        BinaryFieldMultiplicativeFolder<Op, Func> folder(op, fn);
        return BinaryConstantFolder<BinaryFieldConstantFolderConfig>::fold(
            adaptor, &folder);
      })
      .template Case<ExtensionFieldType>([&](auto) {
        ExtMultiplicativeFolder<Op, Func> folder(op, fn);
        return BinaryConstantFolder<ExtensionFieldConstantFolderConfig>::fold(
            adaptor, &folder);
      })
      // NOLINTNEXTLINE(whitespace/newline)
      .Default([](auto) { return OpFoldResult{}; });
}

} // namespace

OpFoldResult NegateOp::fold(FoldAdaptor adaptor) {
  return foldUnaryOp(this, adaptor,
                     [](const FieldOperation &op) { return -op; });
}

OpFoldResult DoubleOp::fold(FoldAdaptor adaptor) {
  return foldUnaryOp(this, adaptor,
                     [](const FieldOperation &op) { return op.dbl(); });
}

OpFoldResult SquareOp::fold(FoldAdaptor adaptor) {
  return foldUnaryOp(this, adaptor,
                     [](const FieldOperation &op) { return op.square(); });
}

OpFoldResult InverseOp::fold(FoldAdaptor adaptor) {
  return foldUnaryOp(this, adaptor,
                     [](const FieldOperation &op) { return op.inverse(); });
}

OpFoldResult FromMontOp::fold(FoldAdaptor adaptor) {
  // from_mont(to_mont(x)) -> x
  if (auto toMont = getInput().getDefiningOp<ToMontOp>()) {
    return toMont.getInput();
  }
  auto montType = getMontgomeryFormType(getType());
  return foldUnaryOp(
      this, adaptor, [](const FieldOperation &op) { return op.fromMont(); },
      montType);
}

OpFoldResult ToMontOp::fold(FoldAdaptor adaptor) {
  // to_mont(from_mont(x)) -> x
  if (auto fromMont = getInput().getDefiningOp<FromMontOp>()) {
    return fromMont.getInput();
  }
  auto stdType = getStandardFormType(getType());
  return foldUnaryOp(
      this, adaptor, [](const FieldOperation &op) { return op.toMont(); },
      stdType);
}

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  return foldAdditiveBinaryOp(
      this, adaptor,
      [](const FieldOperation &a, const FieldOperation &b) { return a + b; });
}

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  return foldAdditiveBinaryOp(
      this, adaptor,
      [](const FieldOperation &a, const FieldOperation &b) { return a - b; });
}

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  return foldMultiplicativeBinaryOp(
      this, adaptor,
      [](const FieldOperation &a, const FieldOperation &b) { return a * b; });
}

namespace {

// Helper to compare a constant attribute with an integer offset.
// Handles both standard and Montgomery forms.
template <typename Predicate>
bool compareWithOffset(Attribute attr, Value val, uint32_t offset,
                       Predicate pred) {
  Type elementType = getElementTypeOrSelf(val.getType());

  // Extract typed attr, handling splat for prime field tensors
  TypedAttr typedAttr;
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    typedAttr = intAttr;
  } else if (auto splatAttr = dyn_cast<SplatElementsAttr>(attr)) {
    typedAttr = splatAttr.getSplatValue<IntegerAttr>();
  } else if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(attr)) {
    typedAttr = denseAttr;
  }
  if (!typedAttr) {
    return false;
  }

  // Use fromUnchecked because the attribute is already stored in the correct
  // representation (Montgomery form for Montgomery types).
  FieldOperation valueOp =
      FieldOperation::fromUnchecked(typedAttr, elementType);
  Type stdType = elementType;
  if (isMontgomery(elementType)) {
    stdType = getStandardFormType(elementType);
    valueOp = valueOp.fromMont();
  }
  FieldOperation offsetOp(static_cast<uint64_t>(offset), stdType);
  return pred(valueOp, offsetOp);
}

bool isNegativeOf(Attribute attr, Value val, uint32_t offset) {
  return compareWithOffset(
      attr, val, offset,
      [](const FieldOperation &a, const FieldOperation &b) { return a == -b; });
}

bool isEqualTo(Attribute attr, Value val, uint32_t offset) {
  return compareWithOffset(
      attr, val, offset,
      [](const FieldOperation &a, const FieldOperation &b) { return a == b; });
}

} // namespace

namespace {
#include "prime_ir/Dialect/Field/IR/FieldCanonicalization.cpp.inc"
}

#include "prime_ir/Utils/CanonicalizationPatterns.inc"

void AddOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
#define PRIME_IR_ADD_PATTERN(Name) patterns.add<Field##Name>(context);
  PRIME_IR_FIELD_ADD_PATTERN_LIST(PRIME_IR_ADD_PATTERN)
#undef PRIME_IR_ADD_PATTERN
}

void SubOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
#define PRIME_IR_SUB_PATTERN(Name) patterns.add<Field##Name>(context);
  PRIME_IR_FIELD_SUB_PATTERN_LIST(PRIME_IR_SUB_PATTERN)
#undef PRIME_IR_SUB_PATTERN
}

void MulOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
#define PRIME_IR_MUL_PATTERN(Name) patterns.add<Field##Name>(context);
  PRIME_IR_FIELD_MUL_PATTERN_LIST(PRIME_IR_MUL_PATTERN)
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
