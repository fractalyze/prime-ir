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

#ifndef PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDVECTORIZE_FIELDVECTORIZE_H_
#define PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDVECTORIZE_FIELDVECTORIZE_H_

// IWYU pragma: begin_keep
// Headers needed for FieldVectorize.h.inc
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
// IWYU pragma: end_keep

namespace mlir::prime_ir::field {

#define GEN_PASS_DECL
#include "prime_ir/Dialect/Field/Conversions/FieldVectorize/FieldVectorize.h.inc"

#define GEN_PASS_REGISTRATION
#include "prime_ir/Dialect/Field/Conversions/FieldVectorize/FieldVectorize.h.inc" // NOLINT(build/include)

} // namespace mlir::prime_ir::field

#endif // PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDVECTORIZE_FIELDVECTORIZE_H_
