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

#ifndef ZKIR_DIALECT_TENSOREXT_IR_TENSOREXTOPS_H_
#define ZKIR_DIALECT_TENSOREXT_IR_TENSOREXTOPS_H_

// IWYU pragma: begin_keep
// Headers needed for TensorExtOps.h.inc
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "zkir/Dialect/TensorExt/IR/TensorExtOps.h.inc"

#endif // ZKIR_DIALECT_TENSOREXT_IR_TENSOREXTOPS_H_
