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

#ifndef PRIME_IR_UTILS_SHAPEDFIELDMUL_H_
#define PRIME_IR_UTILS_SHAPEDFIELDMUL_H_

#include <cstdint>

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir::prime_ir::field {
namespace shaped_mul_detail {

// Extract the scalar field element at row-major linear index `linear` from a
// static-shaped tensor/vector value. Delinearizes `linear` against the shape so
// any rank works (tensor.extract wants per-dim `index` operands, while
// vector.extract wants a static position array).
inline Value extractElement(ImplicitLocOpBuilder &b, Value shapedVal,
                            int64_t linear, ShapedType shapedType) {
  ArrayRef<int64_t> shape = shapedType.getShape();
  SmallVector<int64_t> multiIndex(shape.size());
  int64_t rem = linear;
  for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
    multiIndex[d] = rem % shape[d];
    rem /= shape[d];
  }

  if (isa<VectorType>(shapedType))
    return vector::ExtractOp::create(b, shapedVal, multiIndex);

  SmallVector<Value> indices;
  indices.reserve(multiIndex.size());
  for (int64_t idx : multiIndex)
    indices.push_back(arith::ConstantIndexOp::create(b, idx));
  return tensor::ExtractOp::create(b, shapedVal, indices);
}

// Reassemble `elements` (row-major) into a value of the shaped `resultType`.
inline Value buildFromElements(ImplicitLocOpBuilder &b, Type resultType,
                               ValueRange elements) {
  if (isa<VectorType>(resultType))
    return vector::FromElementsOp::create(b, resultType, elements);
  return tensor::FromElementsOp::create(b, resultType, elements);
}

} // namespace shaped_mul_detail

// Replace a flat-basis binary-field multiply (`op`, whose single result is
// `lhs * rhs`; for a square pass `lhs == rhs == input`) with `mulScalar`
// applied per element. `mulScalar` lowers one scalar field multiply (the
// hardware carryless-multiply ladder). The carryless-multiply intrinsics are
// scalar inline asm, so a shaped mul is unrolled lane by lane
// (extract -> scalar ladder -> from_elements); the field-typed tensor/vector
// scaffolding is then converted to its integer storage form by
// binary-field-to-arith downstream (its ConvertAny handles the extract/
// from_elements, and the unrealized casts collapse to no-ops). Only static
// shapes can be unrolled; a dynamic shape fails to match and falls through to
// the portable elementwise software lowering -- the same correct-but-slow path
// as before. A static zero-element shape unrolls to an empty `from_elements`
// (valid for `tensor<0x...>`); rejecting it would instead leave an
// un-lowerable `field.mul` on the empty tensor for binary-field-to-arith.
//
// Op-agnostic (takes `Operation *` + explicit operands) so the ghash/aes
// patterns of the NVPTX, x86, and ARM specializers share one definition, for
// both `field.mul` and `field.square`.
inline LogicalResult replaceFlatFieldMul(
    PatternRewriter &rewriter, Operation *op, Value lhs, Value rhs,
    llvm::function_ref<Value(ImplicitLocOpBuilder &, Value, Value)> mulScalar) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  Type resultType = op->getResult(0).getType();

  auto shapedType = dyn_cast<ShapedType>(resultType);
  if (!shapedType) {
    rewriter.replaceOp(op, mulScalar(b, lhs, rhs));
    return success();
  }

  if (!shapedType.hasStaticShape())
    return failure();

  int64_t numElements = shapedType.getNumElements();
  SmallVector<Value> results;
  results.reserve(numElements);
  for (int64_t i = 0; i < numElements; ++i) {
    Value lhsElem = shaped_mul_detail::extractElement(b, lhs, i, shapedType);
    // A square (or `mul %x, %x`) passes the same operand for both sides; reuse
    // the single extract rather than emit an identical second one per lane.
    Value rhsElem =
        rhs == lhs ? lhsElem
                   : shaped_mul_detail::extractElement(b, rhs, i, shapedType);
    results.push_back(mulScalar(b, lhsElem, rhsElem));
  }
  rewriter.replaceOp(
      op, shaped_mul_detail::buildFromElements(b, resultType, results));
  return success();
}

} // namespace mlir::prime_ir::field

#endif // PRIME_IR_UTILS_SHAPEDFIELDMUL_H_
