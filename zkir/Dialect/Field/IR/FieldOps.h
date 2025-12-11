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

#ifndef ZKIR_DIALECT_FIELD_IR_FIELDOPS_H_
#define ZKIR_DIALECT_FIELD_IR_FIELDOPS_H_

#include "zkir/Dialect/Field/IR/FieldAttributes.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"

// IWYU pragma: begin_keep
// Headers needed for FieldOps.h.inc
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "zkir/Dialect/Field/IR/FieldOps.h.inc"

namespace mlir::zkir::field {

template <typename OpType>
PrimeFieldType getResultPrimeFieldType(OpType op) {
  return cast<PrimeFieldType>(getElementTypeOrSelf(op.getType()));
}

Type getStandardFormType(Type type);
Type getMontgomeryFormType(Type type);

} // namespace mlir::zkir::field

#endif // ZKIR_DIALECT_FIELD_IR_FIELDOPS_H_
