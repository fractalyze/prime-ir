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

#include "prime_ir/Dialect/TensorExt/IR/TensorExtDialect.h"

// IWYU pragma: begin_keep
// Headers needed for TensorExtDialect.cpp.inc
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
// Headers needed for TensorExtOps.cpp.inc
#include "prime_ir/Dialect/TensorExt/IR/TensorExtOps.h"
// IWYU pragma: end_keep

// Generated definitions
#include "prime_ir/Dialect/TensorExt/IR/TensorExtDialect.cpp.inc"

#define GET_OP_CLASSES
#include "prime_ir/Dialect/TensorExt/IR/TensorExtOps.cpp.inc"

namespace mlir::prime_ir::tensor_ext {

void TensorExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "prime_ir/Dialect/TensorExt/IR/TensorExtOps.cpp.inc" // NOLINT(build/include)
      >();
}

} // namespace mlir::prime_ir::tensor_ext
