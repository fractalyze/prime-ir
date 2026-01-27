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
      auto baseField = cast<PrimeFieldType>(extField.getBaseField());
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
  } else if (auto extField = dyn_cast<ExtensionFieldType>(montType)) {
    if (!extField.isMontgomery()) {
      auto baseField = cast<PrimeFieldType>(extField.getBaseField());
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

    return DenseElementsAttr::get(
        RankedTensorType::get({degree}, pfType.getStorageType()),
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
    } else if (auto extField = dyn_cast<ExtensionFieldType>(elementType)) {
      modulus = extField.getBasePrimeField().getModulus().getValue();
      return success();
    }

    return parser.emitError(parser.getCurrentLocation(),
                            "expected PrimeFieldType or ExtensionFieldType");
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
    } else if (auto efType = dyn_cast<ExtensionFieldType>(parsedType)) {
      if (parsedInts.size() != efType.getDegreeOverPrime()) {
        return parser.emitError(parser.getCurrentLocation())
               << "extension field constant has " << parsedInts.size()
               << " coefficients, but expected " << efType.getDegreeOverPrime();
      }
      auto pfType = efType.getBasePrimeField();
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

  // Shape validation callback for extension field tensors.
  // For tensor<Nx!EF{d}>, the parsed shape should be [N, d] (tensor dims +
  // coefficient dim).
  auto shapeValidationCallback =
      [&](ArrayRef<int64_t> typeShape,
          ArrayRef<int64_t> parsedShape) -> ParseResult {
    auto elementType = getElementTypeOrSelf(parsedType);
    if (auto efType = dyn_cast<ExtensionFieldType>(elementType)) {
      unsigned degree = efType.getDegreeOverPrime();
      SmallVector<int64_t> expectedShape(typeShape);
      expectedShape.push_back(static_cast<int64_t>(degree));
      if (expectedShape.size() != parsedShape.size() ||
          !std::equal(expectedShape.begin(), expectedShape.end(),
                      parsedShape.begin())) {
        return parser.emitError(parser.getCurrentLocation(),
                                "extension field tensor constant shape [")
               << llvm::make_range(parsedShape.begin(), parsedShape.end())
               << "] does not match expected shape [tensor dims: "
               << llvm::make_range(typeShape.begin(), typeShape.end())
               << ", degree: " << degree << "]";
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
    result.addAttribute("value", denseElementsAttr);
    result.addTypes(parsedType);
    return success();
  }
  if (auto efType = dyn_cast<ExtensionFieldType>(elementType)) {
    auto pfType = efType.getBasePrimeField();
    // For tensor<Nx!EF{d}>, the attribute shape is [N, d].
    SmallVector<int64_t> attrShape(shapedType.getShape());
    attrShape.push_back(static_cast<int64_t>(efType.getDegreeOverPrime()));
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
  if (!isa<PrimeFieldType>(getElementTypeOrSelf(type)) &&
      !isa<ExtensionFieldType>(getElementTypeOrSelf(type))) {
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
  if (auto boolAttr = dyn_cast<BoolAttr>(value)) {
    return builder.create<arith::ConstantOp>(loc, boolAttr);
  } else if (auto denseElementsAttr = dyn_cast<DenseIntElementsAttr>(value)) {
    if (!isa<PrimeFieldType>(getElementTypeOrSelf(type)) &&
        !isa<ExtensionFieldType>(getElementTypeOrSelf(type))) {
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
  p.printAttributeWithoutType(getValue());
  p << " : ";
  p.printType(getType());
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
  arith::CmpIPredicate predicate;
  PrimeFieldType pfType;
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
  arith::CmpIPredicate predicate;
  ExtensionFieldType efType;
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
      inputBitWidth = getIntOrPrimeFieldBitWidth(efType.getBasePrimeField()) *
                      efType.getDegreeOverPrime();
    } else if (auto maType =
                   dyn_cast<mod_arith::ModArithType>(inputElementType)) {
      inputBitWidth = mod_arith::getIntOrModArithBitWidth(maType);
    } else {
      inputBitWidth = getIntOrPrimeFieldBitWidth(inputElementType);
    }

    unsigned outputBitWidth;
    if (auto efType = dyn_cast<ExtensionFieldType>(outputElementType)) {
      outputBitWidth = getIntOrPrimeFieldBitWidth(efType.getBasePrimeField()) *
                       efType.getDegreeOverPrime();
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
      inputBitWidth = getIntOrPrimeFieldBitWidth(efType.getBasePrimeField()) *
                      efType.getDegreeOverPrime();
    } else {
      inputBitWidth = getIntOrPrimeFieldBitWidth(inputElementType);
    }
    unsigned outputBitWidth;
    if (auto efType = dyn_cast<ExtensionFieldType>(outputElementType)) {
      outputBitWidth = getIntOrPrimeFieldBitWidth(efType.getBasePrimeField()) *
                       efType.getDegreeOverPrime();
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
  PrimeFieldType pfType;
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
    auto tensorType = RankedTensorType::get(
        {static_cast<int64_t>(coeffs.size())}, baseFieldType.getStorageType());
    return DenseIntElementsAttr::get(tensorType, coeffs);
  }

  OpFoldResult getTensorAttr(ShapedType type,
                             ArrayRef<SmallVector<APInt>> values) const final {
    // Flatten all coefficient vectors into a single vector
    SmallVector<APInt> flattenedValues;
    for (const auto &coeffs : values) {
      flattenedValues.append(coeffs.begin(), coeffs.end());
    }
    // Create result attribute with shape [tensor_dims..., degree]
    PrimeFieldType pfType = efType.getBasePrimeField();
    unsigned degree = efType.getDegreeOverPrime();
    SmallVector<int64_t> attrShape(type.getShape());
    attrShape.push_back(static_cast<int64_t>(degree));
    auto attrType = RankedTensorType::get(attrShape, pfType.getStorageType());
    return DenseIntElementsAttr::get(attrType, flattenedValues);
  }

protected:
  Type type;
  ExtensionFieldType efType;
};

//===----------------------------------------------------------------------===//
// Generic Folders (Prime & Extension Unified Logic)
//===----------------------------------------------------------------------===//

template <typename Func>
class GenericUnaryPrimeFieldFolder
    : public PrimeFieldFolderMixin<
          UnaryConstantFolder<PrimeFieldConstantFolderConfig>::Delegate> {
public:
  GenericUnaryPrimeFieldFolder(Type type, Func fn)
      : PrimeFieldFolderMixin(type), fn(fn) {}

  APInt operate(const APInt &value) const final {
    return static_cast<APInt>(
        fn(FieldOperation::fromUnchecked(value, this->pfType)));
  }

private:
  Func fn;
};

template <typename Func>
class GenericUnaryExtFieldFolder
    : public ExtensionFieldFolderMixin<
          UnaryConstantFolder<ExtensionFieldConstantFolderConfig>::Delegate> {
public:
  GenericUnaryExtFieldFolder(Type type, Func fn)
      : ExtensionFieldFolderMixin(type), fn(fn) {}

  SmallVector<APInt> operate(const SmallVector<APInt> &coeffs) const final {
    return static_cast<SmallVector<APInt>>(
        fn(FieldOperation::fromUnchecked(coeffs, this->efType)));
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
  Func fn;
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
  Func fn;
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
  Func fn;
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
            Op, Func>(op, fn) {
    one = FieldOperation(uint64_t{1}, this->pfType);
  }
  bool isZero(const APInt &value) const final { return value.isZero(); }
  bool isOne(const APInt &value) const final { return value == one; }

private:
  APInt one;
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
            Op, Func>(op, fn) {
    one = static_cast<SmallVector<APInt>>(
        FieldOperation(uint64_t{1}, this->efType));
  }

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
  SmallVector<APInt> one;
};

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

template <typename Op, typename Func>
OpFoldResult foldUnaryOp(Op *op, typename Op::FoldAdaptor adaptor, Func fn) {
  Type type = op->getType();
  Type elemType = getElementTypeOrSelf(type);

  if (isa<PrimeFieldType>(elemType)) {
    GenericUnaryPrimeFieldFolder<Func> folder(type, fn);
    return UnaryConstantFolder<PrimeFieldConstantFolderConfig>::fold(adaptor,
                                                                     &folder);
  }
  if (isa<ExtensionFieldType>(elemType)) {
    GenericUnaryExtFieldFolder<Func> folder(type, fn);
    return UnaryConstantFolder<ExtensionFieldConstantFolderConfig>::fold(
        adaptor, &folder);
  }
  return {};
}

template <typename Op, typename Func>
OpFoldResult foldAdditiveBinaryOp(Op *op, typename Op::FoldAdaptor adaptor,
                                  Func fn) {
  Type type = op->getType();
  Type elemType = getElementTypeOrSelf(type);

  if (isa<PrimeFieldType>(elemType)) {
    PrimeAdditiveFolder<Op, Func> folder(op, fn);
    return BinaryConstantFolder<PrimeFieldConstantFolderConfig>::fold(adaptor,
                                                                      &folder);
  }
  if (isa<ExtensionFieldType>(elemType)) {
    ExtAdditiveFolder<Op, Func> folder(op, fn);
    return BinaryConstantFolder<ExtensionFieldConstantFolderConfig>::fold(
        adaptor, &folder);
  }
  return {};
}

template <typename Op, typename Func>
OpFoldResult
foldMultiplicativeBinaryOp(Op *op, typename Op::FoldAdaptor adaptor, Func fn) {
  Type type = op->getType();
  Type elemType = getElementTypeOrSelf(type);

  if (isa<PrimeFieldType>(elemType)) {
    PrimeMultiplicativeFolder<Op, Func> folder(op, fn);
    return BinaryConstantFolder<PrimeFieldConstantFolderConfig>::fold(adaptor,
                                                                      &folder);
  }
  if (isa<ExtensionFieldType>(elemType)) {
    ExtMultiplicativeFolder<Op, Func> folder(op, fn);
    return BinaryConstantFolder<ExtensionFieldConstantFolderConfig>::fold(
        adaptor, &folder);
  }
  return {};
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

bool isNegativeOf(Attribute attr, Value val, uint32_t offset) {
  IntegerAttr intAttr = dyn_cast_if_present<IntegerAttr>(attr);
  if (auto denseIntAttr = dyn_cast_if_present<SplatElementsAttr>(attr)) {
    intAttr = denseIntAttr.getSplatValue<IntegerAttr>();
  }
  if (intAttr) {
    auto primeFieldType =
        dyn_cast<PrimeFieldType>(getElementTypeOrSelf(val.getType()));
    if (!primeFieldType) {
      return false;
    }
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
        dyn_cast<PrimeFieldType>(getElementTypeOrSelf(val.getType()));
    if (!primeFieldType) {
      return false;
    }
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
