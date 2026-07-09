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

#ifndef PRIME_IR_DIALECT_RING_CONVERSIONS_RINGTOMODARITH_RINGTOMODARITH_H_
#define PRIME_IR_DIALECT_RING_CONVERSIONS_RINGTOMODARITH_RINGTOMODARITH_H_

// IWYU pragma: begin_keep
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithDialect.h"
#include "prime_ir/Dialect/Ring/IR/RingDialect.h"
// IWYU pragma: end_keep

namespace mlir::prime_ir::ring {

#define GEN_PASS_DECL
#include "prime_ir/Dialect/Ring/Conversions/RingToModArith/RingToModArith.h.inc"

#define GEN_PASS_REGISTRATION
#include "prime_ir/Dialect/Ring/Conversions/RingToModArith/RingToModArith.h.inc" // NOLINT(build/include)

} // namespace mlir::prime_ir::ring

#endif // PRIME_IR_DIALECT_RING_CONVERSIONS_RINGTOMODARITH_RINGTOMODARITH_H_
