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

#ifndef ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FIELDTOMODARITH_H_
#define ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FIELDTOMODARITH_H_

// IWYU pragma: begin_keep
// Headers needed for FieldToModArith.h.inc
#include "mlir/Pass/Pass.h"
#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"
// IWYU pragma: end_keep

namespace mlir::zkir::field {

#define GEN_PASS_DECL
#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h.inc"

#define GEN_PASS_REGISTRATION
#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h.inc" // NOLINT(build/include)

} // namespace mlir::zkir::field

#endif // ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FIELDTOMODARITH_H_
