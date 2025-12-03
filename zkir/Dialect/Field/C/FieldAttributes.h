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

#ifndef ZKIR_DIALECT_FIELD_C_FIELDATTRIBUTES_H_
#define ZKIR_DIALECT_FIELD_C_FIELDATTRIBUTES_H_

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// PrimeField attributes.
//===----------------------------------------------------------------------===//

// Returns the typeID of a prime field attribute.
MLIR_CAPI_EXPORTED MlirTypeID zkirPrimeFieldAttrGetTypeID(void);

// Checks whether the given attribute is a prime field attribute.
MLIR_CAPI_EXPORTED bool zkirAttrIsAPrimeField(MlirAttribute attr);

// Creates a prime field attribute of the given type and value.
MLIR_CAPI_EXPORTED MlirAttribute zkirPrimeFieldAttrGet(MlirType type,
                                                       MlirAttribute value);

// Returns the value of the given prime field attribute.
MLIR_CAPI_EXPORTED MlirAttribute zkirPrimeFieldAttrGetValue(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // ZKIR_DIALECT_FIELD_C_FIELDATTRIBUTES_H_
