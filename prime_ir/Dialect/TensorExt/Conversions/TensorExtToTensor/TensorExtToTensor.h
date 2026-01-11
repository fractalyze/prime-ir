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

#ifndef PRIME_IR_DIALECT_TENSOREXT_CONVERSIONS_TENSOREXTTOTENSOR_TENSOREXTTOTENSOR_H_
#define PRIME_IR_DIALECT_TENSOREXT_CONVERSIONS_TENSOREXTTOTENSOR_TENSOREXTTOTENSOR_H_

// IWYU pragma: begin_keep
// Headers needed for TensorExtToTensor.h.inc
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
// IWYU pragma: end_keep

namespace mlir::prime_ir::tensor_ext {
#define GEN_PASS_DECL
#include "prime_ir/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h.inc"

#define GEN_PASS_REGISTRATION
#include "prime_ir/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h.inc" // NOLINT(build/include)
} // namespace mlir::prime_ir::tensor_ext

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_TENSOREXT_CONVERSIONS_TENSOREXTTOTENSOR_TENSOREXTTOTENSOR_H_
