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

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "zkir/Dialect/Field/C/FieldTypes.h"

using namespace mlir;
using namespace mlir::zkir::field;

//===----------------------------------------------------------------------===//
// PrimeField types.
//===----------------------------------------------------------------------===//

MlirTypeID zkirPrimeFieldTypeGetTypeID() {
  return wrap(PrimeFieldType::getTypeID());
}

bool zkirTypeIsAPrimeField(MlirType type) {
  return llvm::isa<PrimeFieldType>(unwrap(type));
}

MlirType zkirPrimeFieldTypeGet(MlirContext ctx, MlirAttribute modulus,
                               bool isMontgomery) {
  return wrap(PrimeFieldType::get(
      unwrap(ctx), llvm::cast<IntegerAttr>(unwrap(modulus)), isMontgomery));
}

MlirAttribute zkirPrimeFieldTypeGetModulus(MlirType type) {
  return wrap(llvm::cast<PrimeFieldType>(unwrap(type)).getModulus());
}

bool zkirPrimeFieldTypeIsMontgomery(MlirType type) {
  return llvm::cast<PrimeFieldType>(unwrap(type)).isMontgomery();
}

//===----------------------------------------------------------------------===//
// QuadraticExtensionField types.
//===----------------------------------------------------------------------===//

MlirTypeID zkirQuadraticExtensionFieldTypeGetTypeID() {
  return wrap(QuadraticExtFieldType::getTypeID());
}

bool zkirTypeIsAQuadraticExtensionField(MlirType type) {
  return llvm::isa<QuadraticExtFieldType>(unwrap(type));
}

MlirType zkirQuadraticExtensionFieldTypeGet(MlirContext ctx, MlirType baseField,
                                            MlirAttribute nonResidue) {
  return wrap(QuadraticExtFieldType::get(
      unwrap(ctx), llvm::cast<PrimeFieldType>(unwrap(baseField)),
      llvm::cast<IntegerAttr>(unwrap(nonResidue))));
}

MlirType zkirQuadraticExtensionFieldTypeGetBaseField(MlirType type) {
  return wrap(llvm::cast<QuadraticExtFieldType>(unwrap(type)).getBaseField());
}

MlirAttribute zkirQuadraticExtensionFieldTypeGetNonResidue(MlirType type) {
  return wrap(llvm::cast<QuadraticExtFieldType>(unwrap(type)).getNonResidue());
}

bool zkirQuadraticExtensionFieldTypeIsMontgomery(MlirType type) {
  return llvm::cast<QuadraticExtFieldType>(unwrap(type)).isMontgomery();
}

// ===----------------------------------------------------------------------===//
// CubicExtensionField types.
// ===----------------------------------------------------------------------===//

MlirTypeID zkirCubicExtensionFieldTypeGetTypeID() {
  return wrap(CubicExtFieldType::getTypeID());
}

bool zkirTypeIsACubicExtensionField(MlirType type) {
  return llvm::isa<CubicExtFieldType>(unwrap(type));
}

MlirType zkirCubicExtensionFieldTypeGet(MlirContext ctx, MlirType baseField,
                                        MlirAttribute nonResidue) {
  return wrap(CubicExtFieldType::get(
      unwrap(ctx), llvm::cast<PrimeFieldType>(unwrap(baseField)),
      llvm::cast<IntegerAttr>(unwrap(nonResidue))));
}

MlirType zkirCubicExtensionFieldTypeGetBaseField(MlirType type) {
  return wrap(llvm::cast<CubicExtFieldType>(unwrap(type)).getBaseField());
}

MlirAttribute zkirCubicExtensionFieldTypeGetNonResidue(MlirType type) {
  return wrap(llvm::cast<CubicExtFieldType>(unwrap(type)).getNonResidue());
}

bool zkirCubicExtensionFieldTypeIsMontgomery(MlirType type) {
  return llvm::cast<CubicExtFieldType>(unwrap(type)).isMontgomery();
}
