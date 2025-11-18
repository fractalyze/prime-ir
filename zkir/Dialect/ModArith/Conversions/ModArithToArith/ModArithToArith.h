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

#ifndef ZKIR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_H_
#define ZKIR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_H_

// IWYU pragma: begin_keep
// Headers needed for ModArithToArith.h.inc
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
// IWYU pragma: end_keep

namespace mlir::zkir::mod_arith {

#define GEN_PASS_DECL
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h.inc"

#define GEN_PASS_REGISTRATION
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h.inc" // NOLINT(build/include)

} // namespace mlir::zkir::mod_arith

#endif // ZKIR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_H_
