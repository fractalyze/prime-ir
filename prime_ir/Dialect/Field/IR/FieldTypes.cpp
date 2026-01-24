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
    return mod_arith::getAttrAsMontgomeryForm(
        cast<field::PrimeFieldType>(efType.getBaseField()).getModulus(),
        denseElementsAttr);
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
    return mod_arith::getAttrAsStandardForm(
        cast<field::PrimeFieldType>(efType.getBaseField()).getModulus(),
        denseElementsAttr);
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

template <unsigned kDegreeOverPrime>
TypedAttr createConstantAttr(ArrayRef<APInt> coeffs,
                             ExtensionFieldType efType) {
  assert(coeffs.size() == kDegreeOverPrime);
  SmallVector<APInt> coeffsVec(coeffs.begin(), coeffs.end());
  ExtensionFieldOperation<kDegreeOverPrime> efOp(coeffsVec, efType);
  return efOp.getDenseIntElementsAttr();
}

template <unsigned kDegreeOverPrime>
Value buildStructFromCoeffs(ImplicitLocOpBuilder &builder, Type structType,
                            llvm::ArrayRef<Value> coeffs) {
  return prime_ir::SimpleStructBuilder<kDegreeOverPrime>::initialized(
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
TypedAttr
dispatchCreateConstantAttr(unsigned degree, ArrayRef<APInt> coeffs,
                           ExtensionFieldType efType,
                           std::integer_sequence<unsigned, Degrees...>) {
  TypedAttr result;
  (void)((Degrees == degree
              ? (result = createConstantAttr<Degrees>(coeffs, efType), true)
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
                           unsigned degree, PrimeFieldType baseField,
                           IntegerAttr nonResidue) {
  if (degree < 2 || degree > kMaxExtDegree) {
    return emitError() << "extension field degree must be between 2 and "
                       << kMaxExtDegree << ", got " << degree;
  }

  // Validate that nonResidue is actually a non-residue:
  // nonResidue^((p - 1) / n) ≢ 1 (mod p)
  // TODO(junbeomlee): Use order of baseField instead of modulus for towers.
  auto nrOp = PrimeFieldOperation::fromUnchecked(nonResidue, baseField);
  APInt p = baseField.getModulus().getValue();
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

  // Parse base field type
  Type baseFieldType;
  if (failed(parser.parseType(baseFieldType))) {
    return nullptr;
  }
  if (!isa<PrimeFieldType>(baseFieldType)) {
    parser.emitError(parser.getCurrentLocation(),
                     "base field must be a prime field");
    return nullptr;
  }
  auto pfType = cast<PrimeFieldType>(baseFieldType);

  // Parse non-residue
  if (failed(parser.parseComma())) {
    return nullptr;
  }
  IntegerAttr nonResidue;
  if (failed(parser.parseAttribute(nonResidue))) {
    return nullptr;
  }
  if (pfType.getIsMontgomery()) {
    nonResidue =
        mod_arith::getAttrAsMontgomeryForm(pfType.getModulus(), nonResidue);
  }

  if (failed(parser.parseGreater())) {
    return nullptr;
  }
  return ExtensionFieldType::get(parser.getContext(), degree, pfType,
                                 nonResidue);
}

void ExtensionFieldType::print(AsmPrinter &printer) const {
  auto pfType = getBaseField();
  auto nonResidue = getNonResidue();
  if (pfType.getIsMontgomery()) {
    nonResidue =
        mod_arith::getAttrAsStandardForm(pfType.getModulus(), nonResidue);
  }
  printer << "<" << getDegree() << "x" << pfType << ", " << nonResidue << ">";
}

llvm::TypeSize ExtensionFieldType::getTypeSizeInBits(
    DataLayout const &, llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return llvm::TypeSize::getFixed(getBaseField().getStorageBitWidth() *
                                  getDegree());
}

uint64_t ExtensionFieldType::getABIAlignment(
    DataLayout const &dataLayout,
    llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return dataLayout.getTypeABIAlignment(getBaseField().getStorageType());
}

bool ExtensionFieldType::isMontgomery() const {
  return getBaseField().getIsMontgomery();
}

TypedAttr ExtensionFieldType::createConstantAttr(int64_t c) const {
  APInt baseCoeff = APInt(getIntOrPrimeFieldBitWidth(getBaseField()), c);
  return createConstantAttrFromValues(ArrayRef<APInt>{baseCoeff});
}

TypedAttr
ExtensionFieldType::createConstantAttrFromValues(ArrayRef<APInt> coeffs) const {
  return ext_field_utils::dispatchCreateConstantAttr(
      getDegree(), coeffs, *this, ext_field_utils::kExtDegreeSequence);
}

ShapedType ExtensionFieldType::overrideShapedType(ShapedType type) const {
  return type;
}

unsigned ExtensionFieldType::getDegreeOverPrime() const { return getDegree(); }

Type ExtensionFieldType::cloneWith(Type baseField, Attribute element) const {
  return ExtensionFieldType::get(getContext(), getDegree(),
                                 cast<PrimeFieldType>(baseField),
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
