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

    auto pfType = cast<PrimeFieldType>(efType.getBaseField());
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
  // TODO(chokboole): support towers of extension fields
  SmallVector<APInt> parsedInts;
  Type parsedType;

  auto getModulusCallback = [&](APInt &modulus) -> ParseResult {
    auto elementType = getElementTypeOrSelf(parsedType);
    if (auto pfType = dyn_cast<PrimeFieldType>(elementType)) {
      modulus = pfType.getModulus().getValue();
      return success();
    } else if (auto extField = dyn_cast<ExtensionFieldType>(elementType)) {
      modulus =
          cast<PrimeFieldType>(extField.getBaseField()).getModulus().getValue();
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
      auto valueAttr = IntegerAttr::get(pfType.getStorageType(), parsedInts[0]);
      result.addAttribute("value", maybeToMontgomery(pfType, valueAttr));
    } else if (auto efType = dyn_cast<ExtensionFieldType>(parsedType)) {
      if (parsedInts.size() != efType.getDegreeOverPrime()) {
        return parser.emitError(parser.getCurrentLocation())
               << "extension field constant has " << parsedInts.size()
               << " coefficients, but expected " << efType.getDegreeOverPrime();
      }
      auto baseFieldType = cast<PrimeFieldType>(efType.getBaseField());
      auto denseAttr = DenseElementsAttr::get(
          RankedTensorType::get({efType.getDegreeOverPrime()},
                                baseFieldType.getStorageType()),
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

  if (failed(parseModularIntegerList(parser, parsedInts, parsedType,
                                     getModulusCallback))) {
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
      !isa<ExtensionFieldType>(getElementTypeOrSelf(type))) {
    return nullptr;
  }

  if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
    return builder.create<ConstantOp>(loc, type, intAttr);
  } else {
    return builder.create<ConstantOp>(loc, type,
                                      cast<DenseIntElementsAttr>(value));
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
  Type elemType = getElementTypeOrSelf(getLhs());
  if (isa<PrimeFieldType>(elemType)) {
    PrimeFieldCmpConstantFolder folder(this);
    return BinaryConstantFolder<CmpConstantFolderConfig>::fold(adaptor,
                                                               &folder);
  }
  if (isa<ExtensionFieldType>(elemType)) {
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

// Helper function to get the modulus from a field-like type.
// Returns the modulus APInt for PrimeFieldType, ExtensionFieldType (base
// field), or ModArithType. Returns std::nullopt for IntegerType (which doesn't
// carry modulus information after mod-arith-to-arith lowering).
std::optional<APInt> getFieldModulus(Type elementType) {
  if (auto pfType = dyn_cast<PrimeFieldType>(elementType)) {
    return pfType.getModulus().getValue();
  }
  if (auto efType = dyn_cast<ExtensionFieldType>(elementType)) {
    return efType.getBaseField().getModulus().getValue();
  }
  if (auto maType = dyn_cast<mod_arith::ModArithType>(elementType)) {
    return maType.getModulus().getValue();
  }
  // IntegerType (and other types) don't carry modulus information
  return std::nullopt;
}

// Helper function to compute the total number of prime field elements
// represented by a shaped type. IntegerType is treated as a single element
// (after mod-arith-to-arith lowering).
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
    return numElements * efType.getDegree();
  }
  return std::nullopt;
}

// Check if two prime field types have the same modulus.
bool haveSameModulus(PrimeFieldType inputPF, PrimeFieldType outputPF) {
  return inputPF.getModulus() == outputPF.getModulus();
}

// Check if two extension field types have compatible structure (same degree,
// same base field modulus, and same non-residue in standard form).
bool haveCompatibleStructure(ExtensionFieldType inputEF,
                             ExtensionFieldType outputEF) {
  // Must have the same degree
  if (inputEF.getDegree() != outputEF.getDegree()) {
    return false;
  }

  // Check base field modulus
  auto inputBaseField = cast<PrimeFieldType>(inputEF.getBaseField());
  auto outputBaseField = cast<PrimeFieldType>(outputEF.getBaseField());
  if (!haveSameModulus(inputBaseField, outputBaseField)) {
    return false;
  }

  // Compare non-residue values in standard form
  auto inputNonResidue = cast<IntegerAttr>(inputEF.getNonResidue());
  auto outputNonResidue = cast<IntegerAttr>(outputEF.getNonResidue());

  // Convert both non-residues to standard form for comparison
  IntegerAttr modulus = inputBaseField.getModulus();
  IntegerAttr inputNRStd =
      inputEF.isMontgomery()
          ? mod_arith::getAttrAsStandardForm(modulus, inputNonResidue)
          : inputNonResidue;
  IntegerAttr outputNRStd =
      outputEF.isMontgomery()
          ? mod_arith::getAttrAsStandardForm(modulus, outputNonResidue)
          : outputNonResidue;

  return inputNRStd.getValue() == outputNRStd.getValue();
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

  // Case 1: Tensor reinterpret bitcast (extension field <->
  // prime/mod_arith/integer field)
  if (isTensorReinterpretBitcast(inputType, outputType)) {
    auto inputShaped = cast<ShapedType>(inputType);
    auto outputShaped = cast<ShapedType>(outputType);

    Type inputElementType = inputShaped.getElementType();
    Type outputElementType = outputShaped.getElementType();

    // The base field modulus must match (works with PrimeFieldType,
    // ExtensionFieldType, and ModArithType). IntegerType doesn't carry modulus
    // info (returns std::nullopt), so we skip the check if one side is integer.
    auto inputModulus = getFieldModulus(inputElementType);
    auto outputModulus = getFieldModulus(outputElementType);
    // If both sides have modulus info, they must match
    if (inputModulus && outputModulus && *inputModulus != *outputModulus) {
      return false;
    }

    // The total number of prime field elements must match
    auto inputTotal = getTotalPrimeFieldElements(inputShaped);
    auto outputTotal = getTotalPrimeFieldElements(outputShaped);
    return inputTotal && outputTotal && *inputTotal == *outputTotal;
  }

  // Case 2: Montgomery form bitcast (same field type, different Montgomery
  // flag)
  Type inputElementType = getElementTypeOrSelf(inputType);
  Type outputElementType = getElementTypeOrSelf(outputType);

  // Check for prime field bitcast (Montgomery form conversion)
  if (auto inputPF = dyn_cast<PrimeFieldType>(inputElementType)) {
    if (auto outputPF = dyn_cast<PrimeFieldType>(outputElementType)) {
      // Same type is not allowed (no-op)
      if (inputPF == outputPF) {
        return false;
      }
      // Allow if shapes match and they have the same modulus
      if (auto inputShaped = dyn_cast<ShapedType>(inputType)) {
        auto outputShaped = dyn_cast<ShapedType>(outputType);
        if (!outputShaped ||
            inputShaped.getShape() != outputShaped.getShape()) {
          return false;
        }
      }
      return haveSameModulus(inputPF, outputPF);
    }
  }

  // Check for extension field bitcast (Montgomery form conversion)
  if (auto inputEF = dyn_cast<ExtensionFieldType>(inputElementType)) {
    if (auto outputEF = dyn_cast<ExtensionFieldType>(outputElementType)) {
      // Same type is not allowed (no-op)
      if (inputEF == outputEF) {
        return false;
      }
      // Allow if shapes match and they have compatible structure
      if (auto inputShaped = dyn_cast<ShapedType>(inputType)) {
        auto outputShaped = dyn_cast<ShapedType>(outputType);
        if (!outputShaped ||
            inputShaped.getShape() != outputShaped.getShape()) {
          return false;
        }
      }
      return haveCompatibleStructure(inputEF, outputEF);
    }
  }

  // Case 3: Original scalar/element-wise bitcast (prime field <-> integer)
  // Must be prime field or integer types
  if (!isa<PrimeFieldType, IntegerType>(inputElementType) ||
      !isa<PrimeFieldType, IntegerType>(outputElementType)) {
    return false;
  }

  // Integer to integer is not allowed (use arith.bitcast)
  if (isa<IntegerType>(inputElementType) &&
      isa<IntegerType>(outputElementType)) {
    return false;
  }

  // Check bitwidth compatibility
  return getIntOrPrimeFieldBitWidth(inputElementType) ==
         getIntOrPrimeFieldBitWidth(outputElementType);
}

LogicalResult BitcastOp::verify() {
  Type inputType = getInput().getType();
  Type outputType = getOutput().getType();

  if (areCastCompatible(inputType, outputType)) {
    return success();
  }

  // Provide detailed error messages
  if (isTensorReinterpretBitcast(inputType, outputType)) {
    auto inputShaped = cast<ShapedType>(inputType);
    auto outputShaped = cast<ShapedType>(outputType);

    Type inputElementType = inputShaped.getElementType();
    Type outputElementType = outputShaped.getElementType();

    auto inputModulus = getFieldModulus(inputElementType);
    auto outputModulus = getFieldModulus(outputElementType);
    // Only report modulus mismatch if both sides have modulus info
    if (inputModulus && outputModulus && *inputModulus != *outputModulus) {
      return emitOpError()
             << "base field moduli must match; input element type "
             << inputElementType << " and output element type "
             << outputElementType << " have different moduli";
    }

    auto inputTotal = getTotalPrimeFieldElements(inputShaped);
    auto outputTotal = getTotalPrimeFieldElements(outputShaped);
    if (!inputTotal || !outputTotal) {
      return emitOpError() << "tensor shapes must be static";
    }
    if (*inputTotal != *outputTotal) {
      return emitOpError()
             << "total prime field element count must match; input has "
             << *inputTotal << " elements but output has " << *outputTotal;
    }
  }

  Type inputElementType = getElementTypeOrSelf(inputType);
  Type outputElementType = getElementTypeOrSelf(outputType);

  if (isa<IntegerType>(inputElementType) &&
      isa<IntegerType>(outputElementType)) {
    return emitOpError()
           << "integer to integer bitcast should use arith.bitcast";
  }

  return emitOpError() << "input type '" << inputType << "' and output type '"
                       << outputType << "' are not compatible for bitcast";
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
      : efType(cast<ExtensionFieldType>(getElementTypeOrSelf(type))) {}

  SmallVector<APInt> getNativeInput(DenseIntElementsAttr attr) const final {
    auto values = attr.getValues<APInt>();
    return {values.begin(), values.end()};
  }

  OpFoldResult getScalarAttr(const SmallVector<APInt> &coeffs) const final {
    PrimeFieldType baseFieldType = cast<PrimeFieldType>(efType.getBaseField());
    auto tensorType = RankedTensorType::get(
        {static_cast<int64_t>(coeffs.size())}, baseFieldType.getStorageType());
    return DenseIntElementsAttr::get(tensorType, coeffs);
  }

  OpFoldResult getTensorAttr(ShapedType type,
                             ArrayRef<SmallVector<APInt>> values) const final {
    llvm_unreachable("not implemented");
  }

protected:
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

  bool isZero(const SmallVector<APInt> &value) const final {
    return llvm::all_of(value, [](const APInt &v) { return v.isZero(); });
  }
};

template <typename Op, typename Func>
class ExtMultiplicativeFolder : public GenericBinaryExtFieldConstantFolder<
                                    MultiplicativeConstantFolderDelegate<
                                        ExtensionFieldConstantFolderConfig>,
                                    Op, Func> {
public:
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
  Type elemType = getElementTypeOrSelf(op->getType());

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
  Type elemType = getElementTypeOrSelf(op->getType());

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
