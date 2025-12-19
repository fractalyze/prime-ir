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

#include "zkir/Dialect/Field/IR/FieldTypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkir/Dialect/ModArith/IR/ModArithAttributes.h"
#include "zkir/Utils/AssemblyFormatUtils.h"

namespace mlir::zkir::field {

#include "zkir/Dialect/Field/IR/FieldTypesInterfaces.cpp.inc"

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
  } else if (isa<ExtensionFieldTypeInterface>(type)) {
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

  auto efType = cast<field::ExtensionFieldTypeInterface>(type);
  auto denseElementsAttr = cast<DenseElementsAttr>(attr);
  if (efType.isMontgomery()) {
    return mod_arith::getAttrAsMontgomeryForm(
        cast<field::PrimeFieldType>(efType.getBaseFieldType()).getModulus(),
        denseElementsAttr);
  } else {
    return denseElementsAttr;
  }
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

  auto efType = cast<field::ExtensionFieldTypeInterface>(type);
  auto denseElementsAttr = cast<DenseElementsAttr>(attr);
  if (efType.isMontgomery()) {
    return mod_arith::getAttrAsStandardForm(
        cast<field::PrimeFieldType>(efType.getBaseFieldType()).getModulus(),
        denseElementsAttr);
  } else {
    return denseElementsAttr;
  }
}

//===----------------------------------------------------------------------===//
// FieldTypeInterface utilities
//===----------------------------------------------------------------------===//

namespace field_utils {
namespace {

template <typename T>
bool isMontgomery(const T *field) {
  if constexpr (std::is_same_v<T, PrimeFieldType>) {
    return field->getIsMontgomery();
  } else {
    return field->getBaseField().getIsMontgomery();
  }
}

template <typename T>
Value createZeroConstant(const T *field, ImplicitLocOpBuilder &builder) {
  return builder.create<ConstantOp>(*field, 0);
}

template <typename T>
Value createOneConstant(const T *field, ImplicitLocOpBuilder &builder) {
  return field->isMontgomery()
             ? builder
                   .create<ToMontOp>(*field,
                                     builder.create<ConstantOp>(
                                         getStandardFormType(*field), 1))
                   .getResult()
             : builder.create<ConstantOp>(*field, 1);
}

} // namespace
} // namespace field_utils

#define DEFINE_FIELD_TYPE_INTERFACE_METHODS(TYPE)                              \
  bool TYPE::isMontgomery() const { return field_utils::isMontgomery(this); }  \
  Value TYPE::createZeroConstant(ImplicitLocOpBuilder &builder) const {        \
    return field_utils::createZeroConstant(this, builder);                     \
  }                                                                            \
  Value TYPE::createOneConstant(ImplicitLocOpBuilder &builder) const {         \
    return field_utils::createOneConstant(this, builder);                      \
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

DEFINE_FIELD_TYPE_INTERFACE_METHODS(PrimeFieldType);

//===----------------------------------------------------------------------===//
// ExtensionFieldTypeInterface utilities
//===----------------------------------------------------------------------===//

namespace ext_field_utils {
namespace {

template <typename T>
Type parseExtensionFieldType(AsmParser &parser) {
  if (failed(parser.parseLess())) {
    return nullptr;
  }
  Type baseField;
  if (failed(parser.parseType(baseField))) {
    return nullptr;
  }
  if (!isa<PrimeFieldType>(baseField)) {
    parser.emitError(parser.getCurrentLocation(),
                     "base field must be a prime field");
    return nullptr;
  }
  auto pfType = cast<PrimeFieldType>(baseField);
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
  return T::get(parser.getContext(), pfType, nonResidue);
}

void printExtensionFieldType(ExtensionFieldTypeInterface extField,
                             AsmPrinter &printer) {
  auto pfType = cast<PrimeFieldType>(extField.getBaseFieldType());
  auto nonResidue = cast<IntegerAttr>(extField.getNonResidue());
  if (pfType.getIsMontgomery()) {
    nonResidue =
        mod_arith::getAttrAsStandardForm(pfType.getModulus(), nonResidue);
  }
  printer << "<" << pfType << ", " << nonResidue << ">";
}

template <typename T>
bool isMontgomery(T *extField) {
  return extField->getBaseField().getIsMontgomery();
}

template <typename T>
Type cloneWith(const T *extField, Type baseField, Attribute element) {
  return T::get(extField->getContext(), cast<PrimeFieldType>(baseField),
                cast<IntegerAttr>(element));
}

template <typename T, unsigned kDegreeOverPrime>
TypedAttr createConstantAttr(const T *extField,
                             llvm::ArrayRef<llvm::APInt> coeffs) {
  auto tensorType = RankedTensorType::get(
      {kDegreeOverPrime},
      IntegerType::get(extField->getContext(),
                       extField->getBaseField().getStorageBitWidth()));
  return DenseIntElementsAttr::get(tensorType, coeffs);
}

template <unsigned kDegreeOverPrime>
Value buildStructFromCoeffs(ImplicitLocOpBuilder &builder, Type structType,
                            llvm::ArrayRef<Value> coeffs) {
  return zkir::SimpleStructBuilder<kDegreeOverPrime>::initialized(
      builder, builder.getLoc(), structType, coeffs);
}

template <unsigned kDegreeOverBase>
llvm::SmallVector<Value> extractCoeffsFromStruct(ImplicitLocOpBuilder &builder,
                                                 Value structValue) {
  zkir::SimpleStructBuilder<kDegreeOverBase> extFieldStruct(structValue);
  return extFieldStruct.getValues(builder, builder.getLoc());
}

} // namespace
} // namespace ext_field_utils

#define DEFINE_EXTENSION_FIELD_INTERFACE_METHODS(TYPE, DEGREE_OVER_PRIME,      \
                                                 DEGREE_OVER_BASE)             \
  DEFINE_FIELD_TYPE_INTERFACE_METHODS(TYPE)                                    \
  unsigned TYPE::getDegreeOverPrime() const { return DEGREE_OVER_PRIME; }      \
  unsigned TYPE::getDegreeOverBase() const { return DEGREE_OVER_BASE; }        \
  Type TYPE::getBaseFieldType() const { return getBaseField(); }               \
  Type TYPE::cloneWith(Type baseField, Attribute element) const {              \
    return ext_field_utils::cloneWith<TYPE>(this, baseField, element);         \
  }                                                                            \
  TypedAttr TYPE::createConstantAttr(llvm::ArrayRef<llvm::APInt> coeffs)       \
      const {                                                                  \
    return ext_field_utils::createConstantAttr<TYPE, DEGREE_OVER_PRIME>(       \
        this, coeffs);                                                         \
  }                                                                            \
  Value TYPE::buildStructFromCoeffs(ImplicitLocOpBuilder &builder,             \
                                    Type structType,                           \
                                    llvm::ArrayRef<Value> coeffs) const {      \
    return ext_field_utils::buildStructFromCoeffs<DEGREE_OVER_BASE>(           \
        builder, structType, coeffs);                                          \
  }                                                                            \
  llvm::SmallVector<Value> TYPE::extractCoeffsFromStruct(                      \
      ImplicitLocOpBuilder &builder, Value structValue) const {                \
    return ext_field_utils::extractCoeffsFromStruct<DEGREE_OVER_BASE>(         \
        builder, structValue);                                                 \
  }

//===----------------------------------------------------------------------===//
// QuadraticExtFieldType
//===----------------------------------------------------------------------===//

// static
Type QuadraticExtFieldType::parse(AsmParser &parser) {
  return ext_field_utils::parseExtensionFieldType<QuadraticExtFieldType>(
      parser);
}

void QuadraticExtFieldType::print(AsmPrinter &printer) const {
  ext_field_utils::printExtensionFieldType(*this, printer);
}

llvm::TypeSize QuadraticExtFieldType::getTypeSizeInBits(
    DataLayout const &, llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return llvm::TypeSize::getFixed(getBaseField().getStorageBitWidth() * 2);
}

uint64_t QuadraticExtFieldType::getABIAlignment(
    DataLayout const &dataLayout,
    llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return dataLayout.getTypeABIAlignment(getBaseField().getStorageType());
}

DEFINE_EXTENSION_FIELD_INTERFACE_METHODS(QuadraticExtFieldType, 2, 2);

//===----------------------------------------------------------------------===//
// CubicExtFieldType
//===----------------------------------------------------------------------===//

// static
Type CubicExtFieldType::parse(AsmParser &parser) {
  return ext_field_utils::parseExtensionFieldType<CubicExtFieldType>(parser);
}

void CubicExtFieldType::print(AsmPrinter &printer) const {
  ext_field_utils::printExtensionFieldType(*this, printer);
}

llvm::TypeSize CubicExtFieldType::getTypeSizeInBits(
    DataLayout const &, llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return llvm::TypeSize::getFixed(getBaseField().getStorageBitWidth() * 3);
}

uint64_t CubicExtFieldType::getABIAlignment(
    DataLayout const &dataLayout,
    llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return dataLayout.getTypeABIAlignment(getBaseField().getStorageType());
}

DEFINE_EXTENSION_FIELD_INTERFACE_METHODS(CubicExtFieldType, 3, 3);

//===----------------------------------------------------------------------===//
// QuarticExtFieldType
//===----------------------------------------------------------------------===//

// static
Type QuarticExtFieldType::parse(AsmParser &parser) {
  return ext_field_utils::parseExtensionFieldType<QuarticExtFieldType>(parser);
}

void QuarticExtFieldType::print(AsmPrinter &printer) const {
  ext_field_utils::printExtensionFieldType(*this, printer);
}

llvm::TypeSize QuarticExtFieldType::getTypeSizeInBits(
    DataLayout const &, llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return llvm::TypeSize::getFixed(getBaseField().getStorageBitWidth() * 4);
}

uint64_t QuarticExtFieldType::getABIAlignment(
    DataLayout const &dataLayout,
    llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return dataLayout.getTypeABIAlignment(getBaseField().getStorageType());
}

DEFINE_EXTENSION_FIELD_INTERFACE_METHODS(QuarticExtFieldType, 4, 4);

} // namespace mlir::zkir::field
