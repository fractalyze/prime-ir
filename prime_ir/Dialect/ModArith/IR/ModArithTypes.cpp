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

#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"

#include "prime_ir/Dialect/ModArith/IR/ModArithOperation.h"
#include "prime_ir/IR/DenseElementBytes.h"
#include "prime_ir/Utils/AssemblyFormatUtils.h"

namespace mlir::prime_ir::mod_arith {
Type ModArithType::parse(AsmParser &parser) {
  return parseModulus<ModArithType>(parser);
}

void ModArithType::print(AsmPrinter &printer) const {
  printModulus(printer, getModulus().getValue(), getStorageType(),
               isMontgomery());
}

llvm::TypeSize ModArithType::getTypeSizeInBits(
    mlir::DataLayout const &,
    llvm::ArrayRef<mlir::DataLayoutEntryInterface>) const {
  return llvm::TypeSize::getFixed(getDenseElementBitSize());
}

uint64_t
ModArithType::getABIAlignment(DataLayout const &dataLayout,
                              llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return dataLayout.getTypeABIAlignment(getStorageType());
}

IntegerAttr getAttrAsStandardForm(IntegerAttr modulusAttr, IntegerAttr attr) {
  auto modArithType = ModArithType::get(attr.getContext(), modulusAttr, true);
  auto value = ModArithOperation::fromUnchecked(attr.getValue(), modArithType);
  return value.fromMont().getIntegerAttr();
}

IntegerAttr getAttrAsMontgomeryForm(IntegerAttr modulusAttr, IntegerAttr attr) {
  auto modArithType = ModArithType::get(attr.getContext(), modulusAttr, false);
  auto value = ModArithOperation::fromUnchecked(attr.getValue(), modArithType);
  return value.toMont().getIntegerAttr();
}

DenseElementsAttr getAttrAsStandardForm(IntegerAttr modulusAttr,
                                        DenseElementsAttr attr) {
  auto modArithType = ModArithType::get(attr.getContext(), modulusAttr, true);
  return attr.mapValues(attr.getElementType(), [&](APInt value) -> APInt {
    auto valueOp = ModArithOperation::fromUnchecked(value, modArithType);
    return valueOp.fromMont();
  });
}

DenseElementsAttr getAttrAsMontgomeryForm(IntegerAttr modulusAttr,
                                          DenseElementsAttr attr) {
  auto modArithType = ModArithType::get(attr.getContext(), modulusAttr, false);
  return attr.mapValues(attr.getElementType(), [&](APInt value) -> APInt {
    auto valueOp = ModArithOperation::fromUnchecked(value, modArithType);
    return valueOp.toMont();
  });
}

TypedAttr ModArithType::createConstantAttr(int64_t c) const {
  ModArithOperation op(c, *this);
  return op.getIntegerAttr();
}

TypedAttr
ModArithType::createConstantAttrFromValues(ArrayRef<APInt> values) const {
  assert(values.size() == 1);
  ModArithOperation op(values[0], *this);
  return op.getIntegerAttr();
}

ShapedType ModArithType::overrideShapedType(ShapedType type) const {
  return type.clone(getStorageType());
}

// DenseElementTypeInterface: a mod_arith element stores as a single
// storage int, so `DenseElementsAttr<tensor<...x!mod_arith.int>>` round-trips
// through the same scalar-int byte view the field types use.
size_t ModArithType::getDenseElementBitSize() const {
  return getStorageType().getWidth();
}

Attribute
ModArithType::convertToAttribute(::llvm::ArrayRef<char> rawData) const {
  return scalarIntConvertToAttribute(getStorageType(), rawData);
}

::llvm::LogicalResult ModArithType::convertFromAttribute(
    Attribute attr, ::llvm::SmallVectorImpl<char> &result) const {
  return scalarIntConvertFromAttribute(getStorageType().getWidth(), attr,
                                       result);
}
} // namespace mlir::prime_ir::mod_arith
