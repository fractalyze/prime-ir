/* Copyright 2026 The PrimeIR Authors.

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

// NOLINTNEXTLINE(whitespace/line_length)
#ifndef PRIME_IR_DIALECT_MODARITH_TRANSFORMS_REDUCEOPTIMIZATION_REDUCEOPTIMIZATION_H_
// NOLINTNEXTLINE(whitespace/line_length)
#define PRIME_IR_DIALECT_MODARITH_TRANSFORMS_REDUCEOPTIMIZATION_REDUCEOPTIMIZATION_H_

#include "mlir/Pass/Pass.h"

namespace mlir::prime_ir::mod_arith {

#define GEN_PASS_DECL_REDUCEOPTIMIZATION
#define GEN_PASS_REGISTRATION
#include "prime_ir/Dialect/ModArith/Transforms/ReduceOptimization/ReduceOptimization.h.inc"

} // namespace mlir::prime_ir::mod_arith

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_MODARITH_TRANSFORMS_REDUCEOPTIMIZATION_REDUCEOPTIMIZATION_H_
