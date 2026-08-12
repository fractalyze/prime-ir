/* Copyright 2026 The PrimeIR Authors.

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

#ifndef PRIME_IR_DIALECT_RING_C_RINGTYPES_H_
#define PRIME_IR_DIALECT_RING_C_RINGTYPES_H_

#include <stdint.h>

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Rq types.
//===----------------------------------------------------------------------===//

// Which CRT basis an RNS ring element is written in. Mirrors ring::Domain.
enum PrimeIRRingDomain {
  PRIME_IR_RING_DOMAIN_COEFF = 0,
  PRIME_IR_RING_DOMAIN_EVAL = 1,
};
typedef enum PrimeIRRingDomain PrimeIRRingDomain;

// Returns the typeID of an RNS quotient-ring type.
MLIR_CAPI_EXPORTED MlirTypeID primeIRRqTypeGetTypeID(void);

// Checks whether the given type is an RNS quotient-ring type.
MLIR_CAPI_EXPORTED bool primeIRTypeIsARq(MlirType type);

// Creates the ring Z_Q[X]/(X^N+1) in RNS form, where Q is the product of
// `moduli` and N is `ringDegree`, with one residue per `storageType` word.
// The type is owned by the context.
MLIR_CAPI_EXPORTED MlirType primeIRRqTypeGet(MlirContext ctx, intptr_t nModuli,
                                             const int64_t *moduli,
                                             MlirAttribute ringDegree,
                                             MlirType storageType,
                                             PrimeIRRingDomain domain);

// Returns the number of RNS limbs.
MLIR_CAPI_EXPORTED intptr_t primeIRRqTypeGetNumModuli(MlirType type);

// Returns the modulus of limb `pos`.
MLIR_CAPI_EXPORTED int64_t primeIRRqTypeGetModulus(MlirType type, intptr_t pos);

// Returns the ring degree N.
MLIR_CAPI_EXPORTED MlirAttribute primeIRRqTypeGetRingDegree(MlirType type);

// Returns the integer word one residue occupies.
MLIR_CAPI_EXPORTED MlirType primeIRRqTypeGetStorageType(MlirType type);

// Returns the basis the element is written in.
MLIR_CAPI_EXPORTED PrimeIRRingDomain primeIRRqTypeGetDomain(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // PRIME_IR_DIALECT_RING_C_RINGTYPES_H_
