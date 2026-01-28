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

#ifndef PRIME_IR_DIALECT_FIELD_IR_FIELDTYPES_H_
#define PRIME_IR_DIALECT_FIELD_IR_FIELDTYPES_H_

#include <string_view>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"

// IWYU pragma: begin_keep
// Headers needed for FieldTypes.h.inc
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ConstantLikeInterface.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "prime_ir/Utils/SimpleStructBuilder.h"
// IWYU pragma: end_keep

namespace mlir::prime_ir::field {

// Supported extension field degrees: quadratic (2), cubic (3), quartic (4).
constexpr size_t kMinExtDegree = 2;
constexpr size_t kMaxExtDegree = 4;
constexpr size_t kNumExtDegrees = kMaxExtDegree - kMinExtDegree + 1;

class PrimeFieldType;

bool isMontgomery(Type type);
unsigned getIntOrPrimeFieldBitWidth(Type type);

mod_arith::ModArithType convertPrimeFieldType(PrimeFieldType type);

ParseResult parseColonFieldType(AsmParser &parser, Type &type);
ParseResult validateAttribute(AsmParser &parser, Type type, Attribute attr,
                              std::string_view attrName);
Attribute maybeToMontgomery(Type type, Attribute attr);
Attribute maybeToStandard(Type type, Attribute attr);

// Creates a constant value for a field type using ConstantLikeInterface.
// This is a convenience function that creates both the attribute and
// operation.
Value createFieldConstant(Type fieldType, ImplicitLocOpBuilder &builder,
                          uint64_t value);

// Creates a zero constant for a field type.
inline Value createFieldZero(Type fieldType, ImplicitLocOpBuilder &builder) {
  return createFieldConstant(fieldType, builder, 0);
}

// Creates a one constant for a field type.
inline Value createFieldOne(Type fieldType, ImplicitLocOpBuilder &builder) {
  return createFieldConstant(fieldType, builder, 1);
}

// Create a coefficient vector with first element set to value, rest zeros.
// Useful for embedding scalars into extension fields as [value, 0, 0, ...].
inline SmallVector<APInt> makeScalarCoeffs(const APInt &value,
                                           unsigned degree) {
  SmallVector<APInt> coeffs(degree, APInt::getZero(value.getBitWidth()));
  coeffs[0] = value;
  return coeffs;
}

// Create a DenseIntElementsAttr representing a scalar in an extension field.
inline DenseIntElementsAttr makeScalarExtFieldAttr(const APInt &value,
                                                   unsigned degree,
                                                   IntegerType storageType) {
  auto coeffs = makeScalarCoeffs(value, degree);
  return DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>(degree)}, storageType),
      coeffs);
}

#include "prime_ir/Dialect/Field/IR/FieldTypesInterfaces.h.inc"

} // namespace mlir::prime_ir::field

#define GET_TYPEDEF_CLASSES
#include "prime_ir/Dialect/Field/IR/FieldTypes.h.inc"

#endif // PRIME_IR_DIALECT_FIELD_IR_FIELDTYPES_H_
