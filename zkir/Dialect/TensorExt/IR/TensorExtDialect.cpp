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

#include "zkir/Dialect/TensorExt/IR/TensorExtDialect.h"

// IWYU pragma: begin_keep
// Headers needed for TensorExtDialect.cpp.inc
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
// Headers needed for TensorExtOps.cpp.inc
#include "zkir/Dialect/TensorExt/IR/TensorExtOps.h"
// IWYU pragma: end_keep

// Generated definitions
#include "zkir/Dialect/TensorExt/IR/TensorExtDialect.cpp.inc"

#define GET_OP_CLASSES
#include "zkir/Dialect/TensorExt/IR/TensorExtOps.cpp.inc"

namespace mlir::zkir::tensor_ext {

void TensorExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "zkir/Dialect/TensorExt/IR/TensorExtOps.cpp.inc" // NOLINT(build/include)
      >();
}

} // namespace mlir::zkir::tensor_ext
