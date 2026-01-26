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

#ifndef PRIME_IR_UTILS_BITCASTOPUTILS_H_
#define PRIME_IR_UTILS_BITCASTOPUTILS_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::prime_ir {

// Template function for canonicalizing bitcast operations.
// This function folds bitcast(bitcast(x)) -> bitcast(x) when there's a
// transitive cast and the types are compatible.
//
// Template parameters:
// - BitcastOpT: The bitcast operation type (e.g., field::BitcastOp,
//               mod_arith::BitcastOp)
//
// The BitcastOpT must have:
// - getInput() method returning the input value
// - getOutput() method returning the output value
// - static areCastCompatible(Type, Type) method for checking type
//   compatibility
template <typename BitcastOpT>
LogicalResult canonicalizeBitcast(BitcastOpT op, PatternRewriter &rewriter) {
  // Fold bitcast(bitcast(x)) -> bitcast(x) when there's a transitive cast.
  auto inputBitcast = op.getInput().template getDefiningOp<BitcastOpT>();
  if (!inputBitcast) {
    return failure();
  }

  // If the types are compatible for direct cast, replace with single bitcast.
  Type srcType = inputBitcast.getInput().getType();
  Type dstType = op.getOutput().getType();

  if (BitcastOpT::areCastCompatible(srcType, dstType)) {
    rewriter.replaceOpWithNewOp<BitcastOpT>(op, dstType,
                                            inputBitcast.getInput());
    return success();
  }

  return failure();
}

} // namespace mlir::prime_ir

#endif // PRIME_IR_UTILS_BITCASTOPUTILS_H_
