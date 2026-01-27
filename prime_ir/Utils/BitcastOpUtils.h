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

#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::prime_ir {

// Helper function to check if input and output types are the same (no-op).
inline bool isSameTypeBitcast(Type inputType, Type outputType) {
  return inputType == outputType;
}

// Helper function to check if element types are both integer types.
// This is used to reject integer-to-integer bitcasts (should use
// arith.bitcast).
inline bool areBothIntegerTypes(Type inputElementType, Type outputElementType) {
  return isa<IntegerType>(inputElementType) &&
         isa<IntegerType>(outputElementType);
}

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
// - static areCastCompatible(TypeRange, TypeRange) method for checking type
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

  if (BitcastOpT::areCastCompatible(TypeRange{srcType}, TypeRange{dstType})) {
    rewriter.replaceOpWithNewOp<BitcastOpT>(op, dstType,
                                            inputBitcast.getInput());
    return success();
  }

  return failure();
}

// Template function for folding bitcast operations.
// This function performs common bitcast folding optimizations:
// 1. Fold bitcast with same input and output types
// 2. Fold bitcast(bitcast(x)) -> x when final type matches original type
//
// Template parameters:
// - BitcastOpT: The bitcast operation type (e.g., field::BitcastOp,
//               mod_arith::BitcastOp)
//
// The BitcastOpT must have:
// - getInput() method returning the input value
// - getOutput() method returning the output value
// - FoldAdaptor type for accessing constant attributes
template <typename BitcastOpT>
OpFoldResult foldBitcast(BitcastOpT op,
                         typename BitcastOpT::FoldAdaptor adaptor) {
  // Fold bitcast with same input and output types.
  if (op.getInput().getType() == op.getOutput().getType()) {
    return op.getInput();
  }

  // Fold bitcast(bitcast(x)) -> x when final type matches original type.
  if (auto inputBitcast = op.getInput().template getDefiningOp<BitcastOpT>()) {
    if (inputBitcast.getInput().getType() == op.getOutput().getType()) {
      return inputBitcast.getInput();
    }
  }

  return {};
}

} // namespace mlir::prime_ir

#endif // PRIME_IR_UTILS_BITCASTOPUTILS_H_
