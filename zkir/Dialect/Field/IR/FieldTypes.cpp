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
#include "zkir/Dialect/Field/IR/FieldOps.h"

namespace mlir::zkir::field {

#include "zkir/Dialect/Field/IR/FieldTypesInterfaces.cpp.inc"

bool isMontgomery(Type type) {
  Type element;
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    element = shapedType.getElementType();
  } else if (auto memrefType = dyn_cast<MemRefType>(type)) {
    element = memrefType.getElementType();
  } else {
    element = type;
  }
  if (auto fieldType = dyn_cast<FieldTypeInterface>(element)) {
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
bool isMontgomery(T *extField) {
  return extField->getBaseField().getIsMontgomery();
}

template <typename T>
Type cloneWith(const T *extField, Type baseField, Attribute element) {
  return T::get(extField->getContext(), cast<PrimeFieldType>(baseField),
                cast<PrimeFieldAttr>(element));
}

template <typename A, typename T, unsigned kRemainingSize, typename... Args>
TypedAttr callCreateConstantAttr(const T *extField,
                                 llvm::ArrayRef<llvm::APInt> coeffs,
                                 Args &&...currentCoeffs) {
  if constexpr (kRemainingSize == 0) {
    return A::get(extField->getContext(), *extField,
                  std::forward<Args>(currentCoeffs)...);
  } else {
    unsigned index = coeffs.size() - kRemainingSize;
    auto nextCoeff =
        PrimeFieldAttr::get(extField->getBaseField(), coeffs[index]);

    return callCreateConstantAttr<A, T, kRemainingSize - 1>(
        extField, coeffs, nextCoeff, std::forward<Args>(currentCoeffs)...);
  }
}

template <typename A, typename T, unsigned kDegreeOverPrime>
TypedAttr createConstantAttr(const T *extField,
                             llvm::ArrayRef<llvm::APInt> coeffs) {
  return callCreateConstantAttr<A, T, kDegreeOverPrime>(extField, coeffs);
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

#define DEFINE_EXTENSION_FIELD_INTERFACE_METHODS(                              \
    TYPE, ATTR, DEGREE_OVER_PRIME, DEGREE_OVER_BASE)                           \
  DEFINE_FIELD_TYPE_INTERFACE_METHODS(TYPE)                                    \
  unsigned TYPE::getDegreeOverPrime() const { return DEGREE_OVER_PRIME; }      \
  unsigned TYPE::getDegreeOverBase() const { return DEGREE_OVER_BASE; }        \
  Type TYPE::getBaseFieldType() const { return getBaseField(); }               \
  Type TYPE::cloneWith(Type baseField, Attribute element) const {              \
    return ext_field_utils::cloneWith<TYPE>(this, baseField, element);         \
  }                                                                            \
  TypedAttr TYPE::createConstantAttr(llvm::ArrayRef<llvm::APInt> coeffs)       \
      const {                                                                  \
    return ext_field_utils::createConstantAttr<ATTR, TYPE, DEGREE_OVER_PRIME>( \
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

llvm::TypeSize QuadraticExtFieldType::getTypeSizeInBits(
    DataLayout const &, llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return llvm::TypeSize::getFixed(getBaseField().getStorageBitWidth() * 2);
}

uint64_t QuadraticExtFieldType::getABIAlignment(
    DataLayout const &dataLayout,
    llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return dataLayout.getTypeABIAlignment(getBaseField().getStorageType());
}

DEFINE_EXTENSION_FIELD_INTERFACE_METHODS(QuadraticExtFieldType,
                                         QuadraticExtFieldAttr, 2, 2);

//===----------------------------------------------------------------------===//
// CubicExtFieldType
//===----------------------------------------------------------------------===//

llvm::TypeSize CubicExtFieldType::getTypeSizeInBits(
    DataLayout const &, llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return llvm::TypeSize::getFixed(getBaseField().getStorageBitWidth() * 3);
}

uint64_t CubicExtFieldType::getABIAlignment(
    DataLayout const &dataLayout,
    llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return dataLayout.getTypeABIAlignment(getBaseField().getStorageType());
}

DEFINE_EXTENSION_FIELD_INTERFACE_METHODS(CubicExtFieldType, CubicExtFieldAttr,
                                         3, 3);

//===----------------------------------------------------------------------===//
// QuarticExtFieldType
//===----------------------------------------------------------------------===//

llvm::TypeSize QuarticExtFieldType::getTypeSizeInBits(
    DataLayout const &, llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return llvm::TypeSize::getFixed(getBaseField().getStorageBitWidth() * 4);
}

uint64_t QuarticExtFieldType::getABIAlignment(
    DataLayout const &dataLayout,
    llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return dataLayout.getTypeABIAlignment(getBaseField().getStorageType());
}

DEFINE_EXTENSION_FIELD_INTERFACE_METHODS(QuarticExtFieldType,
                                         QuarticExtFieldAttr, 4, 4);

} // namespace mlir::zkir::field
