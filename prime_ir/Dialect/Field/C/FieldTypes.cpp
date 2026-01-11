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

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "prime_ir/Dialect/Field/C/FieldTypes.h"

using namespace mlir;
using namespace mlir::prime_ir::field;

//===----------------------------------------------------------------------===//
// PrimeField types.
//===----------------------------------------------------------------------===//

MlirTypeID primeIRPrimeFieldTypeGetTypeID() {
  return wrap(PrimeFieldType::getTypeID());
}

bool primeIRTypeIsAPrimeField(MlirType type) {
  return llvm::isa<PrimeFieldType>(unwrap(type));
}

MlirType primeIRPrimeFieldTypeGet(MlirContext ctx, MlirAttribute modulus,
                                  bool isMontgomery) {
  return wrap(PrimeFieldType::get(
      unwrap(ctx), llvm::cast<IntegerAttr>(unwrap(modulus)), isMontgomery));
}

MlirAttribute primeIRPrimeFieldTypeGetModulus(MlirType type) {
  return wrap(llvm::cast<PrimeFieldType>(unwrap(type)).getModulus());
}

bool primeIRPrimeFieldTypeIsMontgomery(MlirType type) {
  return llvm::cast<PrimeFieldType>(unwrap(type)).isMontgomery();
}

//===----------------------------------------------------------------------===//
// QuadraticExtensionField types.
//===----------------------------------------------------------------------===//

MlirTypeID primeIRQuadraticExtensionFieldTypeGetTypeID() {
  return wrap(QuadraticExtFieldType::getTypeID());
}

bool primeIRTypeIsAQuadraticExtensionField(MlirType type) {
  return llvm::isa<QuadraticExtFieldType>(unwrap(type));
}

MlirType primeIRQuadraticExtensionFieldTypeGet(MlirContext ctx,
                                               MlirType baseField,
                                               MlirAttribute nonResidue) {
  return wrap(QuadraticExtFieldType::get(
      unwrap(ctx), llvm::cast<PrimeFieldType>(unwrap(baseField)),
      llvm::cast<IntegerAttr>(unwrap(nonResidue))));
}

MlirType primeIRQuadraticExtensionFieldTypeGetBaseField(MlirType type) {
  return wrap(llvm::cast<QuadraticExtFieldType>(unwrap(type)).getBaseField());
}

MlirAttribute primeIRQuadraticExtensionFieldTypeGetNonResidue(MlirType type) {
  return wrap(llvm::cast<QuadraticExtFieldType>(unwrap(type)).getNonResidue());
}

bool primeIRQuadraticExtensionFieldTypeIsMontgomery(MlirType type) {
  return llvm::cast<QuadraticExtFieldType>(unwrap(type)).isMontgomery();
}

// ===----------------------------------------------------------------------===//
// CubicExtensionField types.
// ===----------------------------------------------------------------------===//

MlirTypeID primeIRCubicExtensionFieldTypeGetTypeID() {
  return wrap(CubicExtFieldType::getTypeID());
}

bool primeIRTypeIsACubicExtensionField(MlirType type) {
  return llvm::isa<CubicExtFieldType>(unwrap(type));
}

MlirType primeIRCubicExtensionFieldTypeGet(MlirContext ctx, MlirType baseField,
                                           MlirAttribute nonResidue) {
  return wrap(CubicExtFieldType::get(
      unwrap(ctx), llvm::cast<PrimeFieldType>(unwrap(baseField)),
      llvm::cast<IntegerAttr>(unwrap(nonResidue))));
}

MlirType primeIRCubicExtensionFieldTypeGetBaseField(MlirType type) {
  return wrap(llvm::cast<CubicExtFieldType>(unwrap(type)).getBaseField());
}

MlirAttribute primeIRCubicExtensionFieldTypeGetNonResidue(MlirType type) {
  return wrap(llvm::cast<CubicExtFieldType>(unwrap(type)).getNonResidue());
}

bool primeIRCubicExtensionFieldTypeIsMontgomery(MlirType type) {
  return llvm::cast<CubicExtFieldType>(unwrap(type)).isMontgomery();
}

// ===----------------------------------------------------------------------===//
// ExtensionField types.
// ===----------------------------------------------------------------------===//

MlirTypeID primeIRExtensionFieldTypeGetTypeID() {
  return wrap(ExtensionFieldType::getTypeID());
}

bool primeIRTypeIsAnExtensionField(MlirType type) {
  return llvm::isa<ExtensionFieldType>(unwrap(type));
}

MlirType primeIRExtensionFieldTypeGet(MlirContext ctx, unsigned degree,
                                      MlirType baseField,
                                      MlirAttribute nonResidue) {
  return wrap(ExtensionFieldType::get(
      unwrap(ctx), degree, llvm::cast<PrimeFieldType>(unwrap(baseField)),
      llvm::cast<IntegerAttr>(unwrap(nonResidue))));
}

unsigned primeIRExtensionFieldTypeGetDegree(MlirType type) {
  return llvm::cast<ExtensionFieldType>(unwrap(type)).getDegree();
}

MlirType primeIRExtensionFieldTypeGetBaseField(MlirType type) {
  return wrap(llvm::cast<ExtensionFieldType>(unwrap(type)).getBaseField());
}

MlirAttribute primeIRExtensionFieldTypeGetNonResidue(MlirType type) {
  return wrap(llvm::cast<ExtensionFieldType>(unwrap(type)).getNonResidue());
}

bool primeIRExtensionFieldTypeIsMontgomery(MlirType type) {
  return llvm::cast<ExtensionFieldType>(unwrap(type)).isMontgomery();
}
