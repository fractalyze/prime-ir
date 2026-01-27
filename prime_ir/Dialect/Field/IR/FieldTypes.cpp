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

#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "prime_ir/Dialect/Field/IR/FieldOperation.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/TowerFieldConfig.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithAttributes.h"
#include "prime_ir/Utils/AssemblyFormatUtils.h"

namespace mlir::prime_ir::field {

#include "prime_ir/Dialect/Field/IR/FieldTypesInterfaces.cpp.inc"

bool isMontgomery(Type type) {
  Type elementType = getElementTypeOrSelf(type);
  if (auto fieldType = dyn_cast<FieldTypeInterface>(elementType)) {
    return fieldType.isMontgomery();
  }
  return false;
}

unsigned getIntOrPrimeFieldBitWidth(Type type) {
  assert(llvm::isa<PrimeFieldType>(type) || llvm::isa<IntegerType>(type));
  if (auto pfType = dyn_cast<PrimeFieldType>(type)) {
    return pfType.getStorageBitWidth();
  }
  return cast<IntegerType>(type).getWidth();
}

mod_arith::ModArithType convertPrimeFieldType(PrimeFieldType type) {
  IntegerAttr modulus = type.getModulus();
  bool isMontgomery = type.isMontgomery();
  return mod_arith::ModArithType::get(type.getContext(), modulus, isMontgomery);
}

ParseResult parseColonFieldType(AsmParser &parser, Type &type) {
  if (failed(parser.parseColonType(type)))
    return failure();

  if (isa<PrimeFieldType>(type)) {
    return success();
  } else if (isa<ExtensionFieldType>(type)) {
    return success();
  }
  return parser.emitError(parser.getCurrentLocation(),
                          "expected prime field or extension field type");
}

ParseResult validateAttribute(AsmParser &parser, Type type, Attribute attr,
                              std::string_view attrName) {
  if (auto pfType = dyn_cast<field::PrimeFieldType>(type)) {
    if (!isa<IntegerAttr>(attr)) {
      return parser.emitError(parser.getCurrentLocation(),
                              "expected integer attribute for " +
                                  std::string(attrName));
    }
    return success();
  }
  if (!isa<DenseIntElementsAttr>(attr)) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected dense int elements attribute for " +
                                std::string(attrName));
  }
  return success();
}

Attribute maybeToMontgomery(Type type, Attribute attr) {
  if (auto pfType = dyn_cast<field::PrimeFieldType>(type)) {
    auto intAttr = cast<IntegerAttr>(attr);
    if (pfType.isMontgomery()) {
      return mod_arith::getAttrAsMontgomeryForm(pfType.getModulus(), intAttr);
    } else {
      return intAttr;
    }
  }

  auto efType = cast<field::ExtensionFieldType>(type);
  auto denseElementsAttr = cast<DenseElementsAttr>(attr);
  if (efType.isMontgomery()) {
    // Use the underlying prime field's modulus for Montgomery conversion
    return mod_arith::getAttrAsMontgomeryForm(
        efType.getBasePrimeField().getModulus(), denseElementsAttr);
  } else {
    return denseElementsAttr;
  }
}

Value createFieldConstant(Type fieldType, ImplicitLocOpBuilder &builder,
                          uint64_t value) {
  TypedAttr attr;
  auto constantLike = cast<ConstantLikeInterface>(fieldType);
  if (auto efType = dyn_cast<field::ExtensionFieldType>(fieldType)) {
    SmallVector<APInt> coeffs(
        efType.getDegree(),
        APInt(getIntOrPrimeFieldBitWidth(efType.getBaseField()), 0));
    coeffs[0] = APInt(getIntOrPrimeFieldBitWidth(efType.getBaseField()), value);
    attr = constantLike.createConstantAttrFromValues(coeffs);
  } else {
    attr = constantLike.createConstantAttrFromValues(
        ArrayRef<APInt>{APInt(getIntOrPrimeFieldBitWidth(fieldType), value)});
  }
  return builder.create<ConstantOp>(fieldType, attr)->getResult(0);
}

Attribute maybeToStandard(Type type, Attribute attr) {
  if (auto pfType = dyn_cast<field::PrimeFieldType>(type)) {
    auto intAttr = cast<IntegerAttr>(attr);
    if (pfType.isMontgomery()) {
      return mod_arith::getAttrAsStandardForm(pfType.getModulus(), intAttr);
    } else {
      return intAttr;
    }
  }

  auto efType = cast<field::ExtensionFieldType>(type);
  auto denseElementsAttr = cast<DenseElementsAttr>(attr);
  if (efType.isMontgomery()) {
    // Use the underlying prime field's modulus for Montgomery conversion
    return mod_arith::getAttrAsStandardForm(
        efType.getBasePrimeField().getModulus(), denseElementsAttr);
  } else {
    return denseElementsAttr;
  }
}

//===----------------------------------------------------------------------===//
// PrimeFieldType
//===----------------------------------------------------------------------===//

// static
Type PrimeFieldType::parse(AsmParser &parser) {
  return parseModulus<PrimeFieldType>(parser);
}

void PrimeFieldType::print(AsmPrinter &printer) const {
  printModulus(printer, getModulus().getValue(), getStorageType(),
               isMontgomery());
}

llvm::TypeSize PrimeFieldType::getTypeSizeInBits(
    DataLayout const &, llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return llvm::TypeSize::getFixed(getStorageBitWidth());
}

uint64_t PrimeFieldType::getABIAlignment(
    DataLayout const &dataLayout,
    llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return dataLayout.getTypeABIAlignment(getStorageType());
}

bool PrimeFieldType::isMontgomery() const { return getIsMontgomery(); }

TypedAttr PrimeFieldType::createConstantAttr(int64_t c) const {
  PrimeFieldOperation pfOp(c, *this);
  return pfOp.getIntegerAttr();
}

TypedAttr
PrimeFieldType::createConstantAttrFromValues(ArrayRef<APInt> values) const {
  assert(values.size() == 1);
  PrimeFieldOperation pfOp(values[0], *this);
  return pfOp.getIntegerAttr();
}

ShapedType PrimeFieldType::overrideShapedType(ShapedType type) const {
  return type.clone(getStorageType());
}

//===----------------------------------------------------------------------===//
// ExtensionFieldType utilities
//===----------------------------------------------------------------------===//

namespace ext_field_utils {
namespace {

// Create constant attribute using tower-aware dispatch.
// Uses getFlatDenseIntElementsAttr() which works for both tower and non-tower.
TypedAttr createConstantAttrImpl(ArrayRef<APInt> coeffs,
                                 ExtensionFieldType efType) {
  assert(coeffs.size() == efType.getDegreeOverPrime());
  SmallVector<APInt> coeffsVec(coeffs.begin(), coeffs.end());
  auto sig = getTowerSignature(efType);
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define CREATE_CONSTANT_ATTR(unused_sig, TypeName)    \
  TypeName efOp(coeffsVec, efType);                   \
  return efOp.getFlatDenseIntElementsAttr();
  DISPATCH_TOWER_BY_SIGNATURE(sig, CREATE_CONSTANT_ATTR, ExtensionFieldOperation,
                              Op)
#undef CREATE_CONSTANT_ATTR
}

template <unsigned kDegreeOverBase>
Value buildStructFromCoeffs(ImplicitLocOpBuilder &builder, Type structType,
                            llvm::ArrayRef<Value> coeffs) {
  return prime_ir::SimpleStructBuilder<kDegreeOverBase>::initialized(
      builder, builder.getLoc(), structType, coeffs);
}

template <unsigned kDegreeOverBase>
llvm::SmallVector<Value> extractCoeffsFromStruct(ImplicitLocOpBuilder &builder,
                                                 Value structValue) {
  prime_ir::SimpleStructBuilder<kDegreeOverBase> extFieldStruct(structValue);
  return extFieldStruct.getValues(builder, builder.getLoc());
}

// Compile-time dispatch helpers for ExtensionFieldType
template <unsigned... Degrees>
Value dispatchBuildStructFromCoeffs(
    unsigned degree, ImplicitLocOpBuilder &builder, Type structType,
    ArrayRef<Value> coeffs, std::integer_sequence<unsigned, Degrees...>) {
  Value result;
  (void)((Degrees == degree ? (result = buildStructFromCoeffs<Degrees>(
                                   builder, structType, coeffs),
                               true)
                            : false) ||
         ...);
  assert(result && "unsupported extension field degree");
  return result;
}

template <unsigned... Degrees>
SmallVector<Value>
dispatchExtractCoeffsFromStruct(unsigned degree, ImplicitLocOpBuilder &builder,
                                Value structValue,
                                std::integer_sequence<unsigned, Degrees...>) {
  SmallVector<Value> result;
  (void)((Degrees == degree ? (result = extractCoeffsFromStruct<Degrees>(
                                   builder, structValue),
                               true)
                            : false) ||
         ...);
  assert(!result.empty() && "unsupported extension field degree");
  return result;
}

// Generate sequence from kMinExtDegree to kMaxExtDegree (2, 3, 4)
template <size_t Start, size_t... Is>
constexpr auto makeExtDegreeSequence(std::index_sequence<Is...>) {
  return std::integer_sequence<unsigned,
                               static_cast<unsigned>(Start + Is)...>{};
}

constexpr auto kExtDegreeSequence = makeExtDegreeSequence<kMinExtDegree>(
    std::make_index_sequence<kNumExtDegrees>{});

} // namespace
} // namespace ext_field_utils

//===----------------------------------------------------------------------===//
// ExtensionFieldType
//===----------------------------------------------------------------------===//

LogicalResult
ExtensionFieldType::verify(function_ref<InFlightDiagnostic()> emitError,
                           unsigned degree, Type baseField,
                           Attribute nonResidue) {
  if (degree < 2 || degree > kMaxExtDegree) {
    return emitError() << "extension field degree must be between 2 and "
                       << kMaxExtDegree << ", got " << degree;
  }

  // Base field must be either a prime field or an extension field
  if (!isa<PrimeFieldType>(baseField) && !isa<ExtensionFieldType>(baseField)) {
    return emitError() << "base field must be a prime field or extension field";
  }

  // For towers, skip non-residue validation (would need extension field
  // arithmetic). The non-residue should be an element of the base field.
  if (isa<ExtensionFieldType>(baseField)) {
    // TODO(chokobole): Add proper non-residue validation for tower extensions.
    // For now, we trust that the user provides a valid non-residue.
    return success();
  }

  // For direct extensions over prime fields, validate that nonResidue is
  // actually a non-residue: nonResidue^((p - 1) / n) ≢ 1 (mod p)
  auto pfType = cast<PrimeFieldType>(baseField);
  auto nrOp =
      PrimeFieldOperation::fromUnchecked(cast<IntegerAttr>(nonResidue), pfType);
  APInt p = pfType.getModulus().getValue();
  APInt exp = (p - 1).udiv(APInt(p.getBitWidth(), degree));
  if (nrOp.power(exp).isOne()) {
    return emitError() << "nonResidue must satisfy nonResidue^((p - 1) / "
                       << degree << ") != 1 (mod p)";
  }

  return success();
}

// static
Type ExtensionFieldType::parse(AsmParser &parser) {
  if (failed(parser.parseLess())) {
    return nullptr;
  }

  // Parse "Nx" format (e.g., "2x", "3x", "4x")
  unsigned degree;
  if (failed(parser.parseInteger(degree))) {
    return nullptr;
  }

  // Validate degree early
  if (degree < 2 || degree > kMaxExtDegree) {
    parser.emitError(parser.getCurrentLocation(),
                     "extension field degree must be between 2 and ")
        << kMaxExtDegree;
    return nullptr;
  }

  if (failed(parser.parseKeyword("x"))) {
    return nullptr;
  }

  // Parse base field type (can be prime or extension field)
  Type baseFieldType;
  if (failed(parser.parseType(baseFieldType))) {
    return nullptr;
  }
  if (!isa<PrimeFieldType>(baseFieldType) &&
      !isa<ExtensionFieldType>(baseFieldType)) {
    parser.emitError(parser.getCurrentLocation(),
                     "base field must be a prime field or extension field");
    return nullptr;
  }

  // Parse non-residue
  if (failed(parser.parseComma())) {
    return nullptr;
  }
  IntegerAttr nonResidue;
  if (failed(parser.parseAttribute(nonResidue))) {
    return nullptr;
  }

  // Convert non-residue to Montgomery form if base field is in Montgomery form
  if (auto pfType = dyn_cast<PrimeFieldType>(baseFieldType)) {
    if (pfType.getIsMontgomery()) {
      nonResidue =
          mod_arith::getAttrAsMontgomeryForm(pfType.getModulus(), nonResidue);
    }
  } else if (auto efType = dyn_cast<ExtensionFieldType>(baseFieldType)) {
    if (efType.isMontgomery()) {
      // For tower extensions, convert the scalar non-residue to Montgomery form
      // using the underlying prime field's modulus. The scalar will later be
      // embedded as [value, 0, 0, ...] in the base extension field.
      nonResidue = mod_arith::getAttrAsMontgomeryForm(
          efType.getBasePrimeField().getModulus(), nonResidue);
    }
  }

  if (failed(parser.parseGreater())) {
    return nullptr;
  }
  return ExtensionFieldType::get(parser.getContext(), degree, baseFieldType,
                                 nonResidue);
}

void ExtensionFieldType::print(AsmPrinter &printer) const {
  Type baseField = getBaseField();
  auto nonResidue = getNonResidue();

  // Convert non-residue from Montgomery form for printing
  if (auto pfType = dyn_cast<PrimeFieldType>(baseField)) {
    if (pfType.getIsMontgomery()) {
      nonResidue = mod_arith::getAttrAsStandardForm(
          pfType.getModulus(), cast<IntegerAttr>(nonResidue));
    }
  } else if (auto efType = dyn_cast<ExtensionFieldType>(baseField)) {
    if (efType.isMontgomery()) {
      // For tower extensions, convert the scalar non-residue from Montgomery
      // form using the underlying prime field's modulus.
      nonResidue = mod_arith::getAttrAsStandardForm(
          efType.getBasePrimeField().getModulus(),
          cast<IntegerAttr>(nonResidue));
    }
  }

  printer << "<" << getDegree() << "x" << baseField << ", " << nonResidue
          << ">";
}

llvm::TypeSize ExtensionFieldType::getTypeSizeInBits(
    DataLayout const &dataLayout,
    llvm::ArrayRef<DataLayoutEntryInterface> params) const {
  Type baseField = getBaseField();
  if (auto pfType = dyn_cast<PrimeFieldType>(baseField)) {
    return llvm::TypeSize::getFixed(pfType.getStorageBitWidth() * getDegree());
  }
  // For tower: recursively compute base field size
  auto efType = cast<ExtensionFieldType>(baseField);
  auto baseSize = efType.getTypeSizeInBits(dataLayout, params);
  return llvm::TypeSize::getFixed(baseSize.getFixedValue() * getDegree());
}

uint64_t ExtensionFieldType::getABIAlignment(
    DataLayout const &dataLayout,
    llvm::ArrayRef<DataLayoutEntryInterface> params) const {
  Type baseField = getBaseField();
  if (auto pfType = dyn_cast<PrimeFieldType>(baseField)) {
    return dataLayout.getTypeABIAlignment(pfType.getStorageType());
  }
  // For tower: use alignment of the underlying prime field
  return dataLayout.getTypeABIAlignment(getBasePrimeField().getStorageType());
}

bool ExtensionFieldType::isMontgomery() const {
  return getBasePrimeField().getIsMontgomery();
}

TypedAttr ExtensionFieldType::createConstantAttr(int64_t c) const {
  APInt baseCoeff = APInt(getIntOrPrimeFieldBitWidth(getBaseField()), c);
  return createConstantAttrFromValues(ArrayRef<APInt>{baseCoeff});
}

TypedAttr
ExtensionFieldType::createConstantAttrFromValues(ArrayRef<APInt> coeffs) const {
  return ext_field_utils::createConstantAttrImpl(coeffs, *this);
}

ShapedType ExtensionFieldType::overrideShapedType(ShapedType type) const {
  return type;
}

unsigned ExtensionFieldType::getDegreeOverPrime() const {
  Type baseField = getBaseField();
  if (isa<PrimeFieldType>(baseField)) {
    return getDegree();
  }
  // For tower: multiply degrees
  auto efBase = cast<ExtensionFieldType>(baseField);
  return getDegree() * efBase.getDegreeOverPrime();
}

PrimeFieldType ExtensionFieldType::getBasePrimeField() const {
  Type baseField = getBaseField();
  if (auto pfType = dyn_cast<PrimeFieldType>(baseField)) {
    return pfType;
  }
  // Recursively find the prime field at the base of the tower
  return cast<ExtensionFieldType>(baseField).getBasePrimeField();
}

bool ExtensionFieldType::isTower() const {
  return isa<ExtensionFieldType>(getBaseField());
}

unsigned ExtensionFieldType::getTowerDepth() const {
  Type baseField = getBaseField();
  if (isa<PrimeFieldType>(baseField)) {
    return 1;
  }
  return 1 + cast<ExtensionFieldType>(baseField).getTowerDepth();
}

Type ExtensionFieldType::cloneWith(Type baseField, Attribute element) const {
  return ExtensionFieldType::get(getContext(), getDegree(), baseField,
                                 cast<IntegerAttr>(element));
}

Value ExtensionFieldType::buildStructFromCoeffs(
    ImplicitLocOpBuilder &builder, Type structType,
    llvm::ArrayRef<Value> coeffs) const {
  return ext_field_utils::dispatchBuildStructFromCoeffs(
      getDegree(), builder, structType, coeffs,
      ext_field_utils::kExtDegreeSequence);
}

llvm::SmallVector<Value>
ExtensionFieldType::extractCoeffsFromStruct(ImplicitLocOpBuilder &builder,
                                            Value structValue) const {
  return ext_field_utils::dispatchExtractCoeffsFromStruct(
      getDegree(), builder, structValue, ext_field_utils::kExtDegreeSequence);
}

} // namespace mlir::prime_ir::field
