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

#ifndef ZKIR_DIALECT_FIELD_IR_FIELDTYPES_H_
#define ZKIR_DIALECT_FIELD_IR_FIELDTYPES_H_

#include "mlir/IR/Types.h"

// IWYU pragma: begin_keep
// Headers needed for FieldTypes.h.inc
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "zkir/Utils/SimpleStructBuilder.h"
// IWYU pragma: end_keep

namespace mlir::zkir::field {

class PrimeFieldType;

bool isMontgomery(Type type);
unsigned getIntOrPrimeFieldBitWidth(Type type);

#include "zkir/Dialect/Field/IR/FieldTypesInterfaces.h.inc"

} // namespace mlir::zkir::field

#define GET_TYPEDEF_CLASSES
#include "zkir/Dialect/Field/IR/FieldTypes.h.inc"

#endif // ZKIR_DIALECT_FIELD_IR_FIELDTYPES_H_
