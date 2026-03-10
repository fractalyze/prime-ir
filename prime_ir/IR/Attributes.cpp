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

#include "prime_ir/IR/Attributes.h"

#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"

namespace mlir::prime_ir {

ShapedType maybeConvertPrimeIRToBuiltinType(ShapedType type) {
  auto elementType = type.getElementType();
  if (auto modArithType = dyn_cast<mod_arith::ModArithType>(elementType)) {
    return type.clone(modArithType.getStorageType());
  } else if (auto bfType = dyn_cast<field::BinaryFieldType>(elementType)) {
    return type.clone(bfType.getStorageType());
  } else if (auto fieldType =
                 dyn_cast<field::FieldTypeInterface>(elementType)) {
    // For prime fields, towerDims is empty so attrShape == type.getShape().
    // For extension fields, towerDims appends coefficient dimensions
    // (e.g., tensor<4x!EF{2}> stores as tensor<4x2xi32>).
    auto pfType = field::getBasePrimeField(elementType);
    auto towerDims = fieldType.getAttrShape();
    SmallVector<int64_t> attrShape(type.getShape());
    attrShape.append(towerDims.begin(), towerDims.end());
    return type.clone(attrShape, pfType.getStorageType());
  }
  return type;
}

} // namespace mlir::prime_ir
