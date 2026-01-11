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

#ifndef PRIME_IR_DIALECT_FIELD_IR_FIELDOPS_H_
#define PRIME_IR_DIALECT_FIELD_IR_FIELDOPS_H_

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "prime_ir/Dialect/Field/IR/FieldAttributes.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

// IWYU pragma: begin_keep
// Headers needed for FieldOps.h.inc
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "prime_ir/Dialect/Field/IR/FieldOps.h.inc"

namespace mlir::prime_ir::field {

template <typename OpType>
PrimeFieldType getResultPrimeFieldType(OpType op) {
  return cast<PrimeFieldType>(getElementTypeOrSelf(op.getType()));
}

Type getStandardFormType(Type type);
Type getMontgomeryFormType(Type type);

ParseResult parseFieldConstant(OpAsmParser &parser, OperationState &result);

} // namespace mlir::prime_ir::field

#endif // PRIME_IR_DIALECT_FIELD_IR_FIELDOPS_H_
