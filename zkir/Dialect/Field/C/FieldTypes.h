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

#ifndef ZKIR_DIALECT_FIELD_C_FIELDTYPES_H_
#define ZKIR_DIALECT_FIELD_C_FIELDTYPES_H_

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// PrimeField types.
//===----------------------------------------------------------------------===//

// Returns the typeID of a prime field type.
MLIR_CAPI_EXPORTED MlirTypeID zkirPrimeFieldTypeGetTypeID(void);

// Checks whether the given type is a prime field type.
MLIR_CAPI_EXPORTED bool zkirTypeIsAPrimeField(MlirType type);

// Creates a prime field type of the given modulus and montgomery form in the
// context. The type is owned by the context.
MLIR_CAPI_EXPORTED MlirType zkirPrimeFieldTypeGet(MlirContext ctx,
                                                  MlirAttribute modulus,
                                                  bool isMontgomery);

// Returns the modulus of the given prime field type.
MLIR_CAPI_EXPORTED MlirAttribute zkirPrimeFieldTypeGetModulus(MlirType type);

// Checks whether the given type is a montgomery form.
MLIR_CAPI_EXPORTED bool zkirPrimeFieldTypeIsMontgomery(MlirType type);

//===----------------------------------------------------------------------===//
// QuadraticExtensionField types.
//===----------------------------------------------------------------------===//

// Returns the typeID of a quadratic extension field type.
MLIR_CAPI_EXPORTED MlirTypeID zkirQuadraticExtensionFieldTypeGetTypeID(void);

// Checks whether the given type is a quadratic extension field type.
MLIR_CAPI_EXPORTED bool zkirTypeIsAQuadraticExtensionField(MlirType type);

// Creates a quadratic extension field type of the given base field and
// non-residue form in the context. The type is owned by the context.
MLIR_CAPI_EXPORTED MlirType zkirQuadraticExtensionFieldTypeGet(
    MlirContext ctx, MlirType baseField, MlirAttribute nonResidue);

// Returns the base field of the given quadratic extension field type.
MLIR_CAPI_EXPORTED MlirType
zkirQuadraticExtensionFieldTypeGetBaseField(MlirType type);

// Returns the non-residue of the given quadratic extension field type.
MLIR_CAPI_EXPORTED MlirAttribute
zkirQuadraticExtensionFieldTypeGetNonResidue(MlirType type);

// Checks whether the given type is a montgomery form.
MLIR_CAPI_EXPORTED bool
zkirQuadraticExtensionFieldTypeIsMontgomery(MlirType type);

// ===----------------------------------------------------------------------===//
// CubicExtensionField types.
// ===----------------------------------------------------------------------===//

// Returns the typeID of a cubic extension field type.
MLIR_CAPI_EXPORTED MlirTypeID zkirCubicExtensionFieldTypeGetTypeID(void);

// Checks whether the given type is a cubic extension field type.
MLIR_CAPI_EXPORTED bool zkirTypeIsACubicExtensionField(MlirType type);

// Creates a cubic extension field type of the given base field and
// non-residue form in the context. The type is owned by the context.
MLIR_CAPI_EXPORTED MlirType zkirCubicExtensionFieldTypeGet(
    MlirContext ctx, MlirType baseField, MlirAttribute nonResidue);

// Returns the base field of the given cubic extension field type.
MLIR_CAPI_EXPORTED MlirType
zkirCubicExtensionFieldTypeGetBaseField(MlirType type);

// Returns the non-residue of the given cubic extension field type.
MLIR_CAPI_EXPORTED MlirAttribute
zkirCubicExtensionFieldTypeGetNonResidue(MlirType type);

// Checks whether the given type is a montgomery form.
MLIR_CAPI_EXPORTED bool zkirCubicExtensionFieldTypeIsMontgomery(MlirType type);

// ===----------------------------------------------------------------------===//
// ExtensionField types.
// ===----------------------------------------------------------------------===//

// Returns the typeID of an extension field type.
MLIR_CAPI_EXPORTED MlirTypeID zkirExtensionFieldTypeGetTypeID(void);

// Checks whether the given type is an extension field type.
MLIR_CAPI_EXPORTED bool zkirTypeIsAnExtensionField(MlirType type);

// Creates an extension field type of the given degree, base field and
// non-residue in the context. The type is owned by the context.
MLIR_CAPI_EXPORTED MlirType zkirExtensionFieldTypeGet(MlirContext ctx,
                                                      unsigned degree,
                                                      MlirType baseField,
                                                      MlirAttribute nonResidue);

// Returns the degree of the given extension field type.
MLIR_CAPI_EXPORTED unsigned zkirExtensionFieldTypeGetDegree(MlirType type);

// Returns the base field of the given extension field type.
MLIR_CAPI_EXPORTED MlirType zkirExtensionFieldTypeGetBaseField(MlirType type);

// Returns the non-residue of the given extension field type.
MLIR_CAPI_EXPORTED MlirAttribute
zkirExtensionFieldTypeGetNonResidue(MlirType type);

// Checks whether the given type is a montgomery form.
MLIR_CAPI_EXPORTED bool zkirExtensionFieldTypeIsMontgomery(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // ZKIR_DIALECT_FIELD_C_FIELDTYPES_H_
