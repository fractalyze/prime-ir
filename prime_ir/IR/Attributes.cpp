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

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"

namespace mlir::prime_ir {

namespace {
unsigned fieldElementBits(Type t) {
  if (auto pf = dyn_cast<field::PrimeFieldType>(t))
    return pf.getTypeSizeInBits();
  if (auto ef = dyn_cast<field::ExtensionFieldType>(t))
    return ef.getTypeSizeInBits();
  if (auto bf = dyn_cast<field::BinaryFieldType>(t))
    return bf.getTypeSizeInBits();
  llvm_unreachable("unsupported base field type");
}
} // namespace

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
  } else if (auto ptType =
                 dyn_cast<elliptic_curve::PointTypeInterface>(elementType)) {
    // A point appends a coordinate dim; each coord is one base-field-wide int
    // (e.g. tensor<4x!affine<pf>> stores as tensor<4x2xi32>).
    SmallVector<int64_t> shape(type.getShape());
    shape.push_back(ptType.getNumCoords());
    return type.clone(
        shape, IntegerType::get(type.getContext(),
                                fieldElementBits(ptType.getBaseFieldType())));
  }
  return type;
}

DenseElementsAttr maybeConvertPrimeIRToBuiltinAttr(DenseElementsAttr attr) {
  auto builtinType = maybeConvertPrimeIRToBuiltinType(attr.getType());
  if (builtinType == attr.getType())
    return attr;
  // A splat holds one logical element's bytes (EF: degree primes; EC point:
  // numCoords coords); the storage-int form has no splat encoding for a
  // multi-int row, so tile across all N elements.
  if (attr.isSplat())
    return denseIntFromRawBytes(
        builtinType,
        replicateRawBytes(attr.getRawData(), attr.getType().getNumElements()));
  return denseIntFromRawBytes(builtinType, attr.getRawData());
}

} // namespace mlir::prime_ir
