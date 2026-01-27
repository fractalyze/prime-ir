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

#ifndef PRIME_IR_UTILS_BITSERIALALGORITHM_H_
#define PRIME_IR_UTILS_BITSERIALALGORITHM_H_

#include <functional>

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"

namespace mlir::prime_ir {

// Callback type for doubling operation.
// For field elements: square operation
// For EC points: point doubling operation
using DoubleCallback = std::function<Value(ImplicitLocOpBuilder &, Value)>;

// Callback type for accumulation operation.
// For field elements: multiplication
// For EC points: point addition
using AccumulateCallback =
    std::function<Value(ImplicitLocOpBuilder &, Value, Value)>;

// Generates a bit-serial algorithm (square-and-multiply / double-and-add)
// using scf::WhileOp.
//
// This implements an optimized LSB-first binary method by unrolling the
// first iteration. The logic is equivalent to:
//
//   result = (scalar & 1) ? base : identity
//   base_power = base
//   scalar >>= 1
//   while (scalar > 0) {
//     base_power = double(base_power)
//     if (scalar & 1) result = accumulate(result, base_power)
//     scalar >>= 1
//   }
Value generateBitSerialLoop(ImplicitLocOpBuilder &b, Value scalar, Value base,
                            Value identity, DoubleCallback doubleOp,
                            AccumulateCallback accumulateOp);

} // namespace mlir::prime_ir

#endif // PRIME_IR_UTILS_BITSERIALALGORITHM_H_
