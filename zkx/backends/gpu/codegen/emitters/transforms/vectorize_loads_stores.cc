/* Copyright 2024 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "zkx/codegen/emitters/ir/zkx_ops.h"

namespace zkx::gpu {
namespace {

#define GEN_PASS_DECL_VECTORIZELOADSANDSTORESPASS
#define GEN_PASS_DEF_VECTORIZELOADSANDSTORESPASS
#include "zkx/backends/gpu/codegen/emitters/transforms/passes.h.inc"

using mlir::Value;

namespace arith = ::mlir::arith;
namespace scf = mlir::scf;

// Tries to find the stride of a symbol or dimension in an affine expression.
// Returns std::nullopt if the stride could not be determined.
//
// Note: this function only attempts to handle the cases where the stride is
// known to be 0 or 1.
//
// Examples:
//   - GetStride(d0 + d1, d0) = 1
//   - GetStride(d0 * 2, d0) = unknown (nullopt)
std::optional<int> GetStride(mlir::AffineExpr expr,
                             mlir::AffineExpr dim_or_sym) {
  if (auto binop = mlir::dyn_cast_or_null<mlir::AffineBinaryOpExpr>(expr)) {
    std::optional<int> lhs_stride = GetStride(binop.getLHS(), dim_or_sym);
    std::optional<int> rhs_stride = GetStride(binop.getRHS(), dim_or_sym);

    if (binop.getKind() == mlir::AffineExprKind::Add) {
      if (lhs_stride && rhs_stride) {
        return *lhs_stride + *rhs_stride;
      }
      return std::nullopt;
    }
    // Just return 0 if the expression doesn't occur on either side.
    if (lhs_stride == 0 && rhs_stride == 0) {
      return 0;
    }
    // Otherwise, we don't know the stride.
    return std::nullopt;
  }
  return expr == dim_or_sym ? 1 : 0;
}

// Tries to compute the alignment (periodicity) of the remainder when reducing
// an affine expression modulo a symbol or dimension.
//
// The "alignment of remainder" here means the granularity with which the value
// of the expression can change when the specified dimension/symbol varies.
// In other words, it gives the greatest common divisor (gcd)-like factor that
// constrains the remainder pattern.
//
// This utility is conservative: it only handles basic affine binary
// expressions, and returns simple gcd/constant factors for known cases.
//
// clang-format off
// Examples:
//   - GetAlignmentOfRemainder(d0 + d1, d0) = align(d1) = 1
//   - GetAlignmentOfRemainder(d0 + 5, d0) = align(5) = 5
//   - GetAlignmentOfRemainder(d0 * 2, d0) = align(d0) * align(2) = 2
//   - GetAlignmentOfRemainder((d0 * 4) % 8, d0) = gcd(align(d0 * 4), align(8)) = 4
//   - Division cases (floor/ceil div) always return 1 (most conservative).
// clang-format on
//
// Note: constant expressions return their literal value as alignment,
//       and non-constant, non-binary expressions default to 1.
int64_t GetAlignmentOfRemainder(mlir::AffineExpr expr,
                                mlir::AffineExpr dim_or_sym) {
  if (auto binop = mlir::dyn_cast_or_null<mlir::AffineBinaryOpExpr>(expr)) {
    int64_t lhs_align = GetAlignmentOfRemainder(binop.getLHS(), dim_or_sym);
    int64_t rhs_align = GetAlignmentOfRemainder(binop.getRHS(), dim_or_sym);

    switch (binop.getKind()) {
      case mlir::AffineExprKind::Add:
        if (binop.getLHS() == dim_or_sym) return rhs_align;
        if (binop.getRHS() == dim_or_sym) return lhs_align;
        return std::gcd(lhs_align, rhs_align);
      case mlir::AffineExprKind::Mul:
        return lhs_align * rhs_align;
      case mlir::AffineExprKind::FloorDiv:
      case mlir::AffineExprKind::CeilDiv:
        return 1;
      case mlir::AffineExprKind::Mod:
        // (a * c) % (b * c) = (a % b) * c.
        return std::gcd(lhs_align, rhs_align);
      default:
        llvm_unreachable("expr is none of the binary expressions");
    }
  }
  if (auto cst = mlir::dyn_cast<mlir::AffineConstantExpr>(expr)) {
    return cst.getValue();
  }
  return 1;
}

// Attempts to extract the vector type for the given loop. This means:
// - checks that the lower bound is 0
// - checks that the step is 1
// - checks that the upper bound is 2, 4, or 8.
// Returns a vector type with the given upper bound and the tensor's element
// type.
// All tensors are 1D after flatten-tensors pass.
mlir::VectorType GetVectorType(mlir::RankedTensorType tensor_type,
                               scf::ForOp loop) {
  if (tensor_type.getEncoding()) {
    return nullptr;
  }
  if (tensor_type.getRank() != 1) {
    return nullptr;
  }
  if (!mlir::VectorType::isValidElementType(tensor_type.getElementType())) {
    return nullptr;
  }
  if (mlir::getConstantIntValue(loop.getStep()) != 1 ||
      mlir::getConstantIntValue(loop.getLowerBound()) != 0) {
    return nullptr;
  }
  const std::optional<int64_t> vector_size_opt =
      mlir::getConstantIntValue(loop.getUpperBound());
  if (!vector_size_opt) {
    return nullptr;
  }
  const int64_t vector_size = *vector_size_opt;
  if (vector_size != 2 && vector_size != 4 && vector_size != 8) {
    return nullptr;  // Unsupported vector size.
  }
  // TODO(chokobole): Support misaligned start indices.
  // See https://github.com/fractalyze/zkx/pull/81#discussion_r2347747425.
  if (tensor_type.getShape().back() % vector_size) {
    return nullptr;  // Misaligned start indices.
  }
  return mlir::VectorType::get({vector_size}, tensor_type.getElementType());
}

std::optional<Value> GetVectorBaseIndices(Value index, scf::ForOp loop,
                                          mlir::VectorType vector_type,
                                          mlir::ImplicitLocOpBuilder& b) {
  Value induction_var = loop.getInductionVar();
  if (index == induction_var) {
    return b.create<arith::ConstantIndexOp>(0);
  }

  auto apply_indexing =
      mlir::dyn_cast_or_null<ApplyIndexingOp>(index.getDefiningOp());
  if (!apply_indexing) {
    return std::nullopt;
  }

  // We don't generate these, but they are allowed in theory.
  if (apply_indexing->getNumResults() != 1) {
    return std::nullopt;
  }
  mlir::AffineMap map = apply_indexing.getAffineMap();

  int induction_var_operand_index;
  mlir::AffineExpr induction_var_expr = nullptr;
  for (auto [index, operand] : llvm::enumerate(apply_indexing.getOperands())) {
    if (operand == induction_var) {
      if (induction_var_expr) {
        // The induction variable should be used only once.
        return std::nullopt;
      }
      induction_var_operand_index = index;
      induction_var_expr = index < map.getNumDims()
                               ? mlir::getAffineDimExpr(index, b.getContext())
                               : mlir::getAffineSymbolExpr(
                                     index - map.getNumDims(), b.getContext());
    } else if (!operand.getParentRegion()->isProperAncestor(
                   &loop.getBodyRegion())) {
      // If the operand is defined inside the loop, we can't hoist the
      // apply_indexing outside the loop.
      return std::nullopt;
    }
  }
  if (!induction_var_expr) {
    return std::nullopt;
  }

  if (GetStride(map.getResult(0), induction_var_expr) != 1) {
    // The indexing map is not contiguous in the vectorized dimension.
    return std::nullopt;
  }

  if (GetAlignmentOfRemainder(map.getResult(0), induction_var_expr) %
      vector_type.getNumElements()) {
    return std::nullopt;
  }

  auto operands = llvm::to_vector(apply_indexing.getOperands());
  operands[induction_var_operand_index] = b.create<arith::ConstantIndexOp>(0);

  return b.create<ApplyIndexingOp>(operands, apply_indexing.getIndexingMap())
      ->getResult(0);
}

bool IsConflictFree(mlir::tensor::ExtractOp op) {
  return op.getTensor().getParentRegion()->isProperAncestor(
      op->getParentRegion());
}

struct VectorizeLoad : mlir::OpRewritePattern<mlir::tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::tensor::ExtractOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto loop = mlir::dyn_cast_or_null<scf::ForOp>(op->getParentOp());
    if (!loop) {
      return rewriter.notifyMatchFailure(op, "no loop found");
    }
    if (!IsConflictFree(op)) {
      return rewriter.notifyMatchFailure(op,
                                         "source may be written in the loop");
    }

    mlir::VectorType vector_type =
        GetVectorType(op.getTensor().getType(), loop);
    if (!vector_type) {
      return rewriter.notifyMatchFailure(op, "not a vectorizable loop");
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    b.setInsertionPoint(loop);
    std::optional<Value> vector_index =
        GetVectorBaseIndices(op.getIndices().front(), loop, vector_type, b);
    if (!vector_index) {
      return rewriter.notifyMatchFailure(
          op, "the instruction does not access contiguous elements");
    }
    auto loaded_vector = b.create<mlir::vector::TransferReadOp>(
        vector_type, op.getTensor(), *vector_index, /*padding=*/std::nullopt,
        llvm::ArrayRef<bool>{true});
    rewriter.replaceOpWithNewOp<mlir::vector::ExtractOp>(
        op, loaded_vector, loop.getInductionVar());
    return mlir::success();
  }
};

// Verifies that the insertions happening in the loop can all safely be batched
// in the end.
bool IsConflictFree(mlir::tensor::InsertOp op) {
  // The insertion's only use must be the yield.
  if (!op->hasOneUse() || !mlir::isa<scf::YieldOp>(*op->user_begin())) {
    return false;
  }
  // The destination must be one of the loop's block arguments, and the
  // destination must be the argument's only use.
  auto bbarg = mlir::dyn_cast<mlir::BlockArgument>(op.getDest());
  return bbarg && bbarg.hasOneUse() &&
         bbarg.getOwner()->getParentOp() == op->getParentOp();
}

struct VectorizeStore : mlir::OpRewritePattern<mlir::tensor::InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::tensor::InsertOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto loop = mlir::dyn_cast_or_null<scf::ForOp>(op->getParentOp());
    if (!loop) {
      return rewriter.notifyMatchFailure(op, "no loop found");
    }
    if (!IsConflictFree(op)) {
      return rewriter.notifyMatchFailure(op, "write may be read back by loop");
    }
    mlir::VectorType vector_type = GetVectorType(op.getDest().getType(), loop);
    if (!vector_type) {
      return rewriter.notifyMatchFailure(op, "loop is not vectorizable");
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    b.setInsertionPoint(loop);
    std::optional<Value> vector_index =
        GetVectorBaseIndices(op.getIndices().front(), loop, vector_type, b);
    if (!vector_index) {
      return rewriter.notifyMatchFailure(
          op, "the instruction does not access contiguous elements");
    }

    auto zero_attr = b.getZeroAttr(vector_type);
    auto& dialect = vector_type.getElementType().getDialect();
    auto* materialize_op =
        dialect.materializeConstant(b, zero_attr, vector_type, b.getLoc());
    auto init = materialize_op
                    ? materialize_op->getResult(0)
                    : b.create<arith::ConstantOp>(zero_attr).getResult();

    auto yield_fn = [&](mlir::OpBuilder& yield_b, mlir::Location yield_loc,
                        llvm::ArrayRef<mlir::BlockArgument> bbarg) {
      auto induction_var =
          mlir::cast<scf::ForOp>(bbarg.front().getOwner()->getParentOp())
              .getInductionVar();
      auto insert_op = yield_b.create<mlir::vector::InsertOp>(
          yield_loc, op.getScalar(), bbarg.front(), induction_var);
      return llvm::SmallVector<Value>{insert_op.getResult()};
    };
    int result_index = op->use_begin()->getOperandNumber();
    mlir::LoopLikeOpInterface new_for = *loop.replaceWithAdditionalYields(
        rewriter, init,
        /*replaceInitOperandUsesInLoop=*/false, yield_fn);

    b.setInsertionPointAfter(new_for);
    rewriter.replaceOp(op, op.getDest());

    mlir::OpResult filled_vector = new_for->getResults().back();
    auto written = b.create<mlir::vector::TransferWriteOp>(
        filled_vector, new_for.getInits()[result_index], *vector_index,
        llvm::ArrayRef<bool>{true});
    new_for->getResult(result_index).replaceAllUsesWith(written.getResult());

    return mlir::success();
  }
};

class VectorizeLoadsAndStoresPass
    : public impl::VectorizeLoadsAndStoresPassBase<
          VectorizeLoadsAndStoresPass> {
 public:
  void runOnOperation() override {
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);
    patterns.add<VectorizeLoad, VectorizeStore>(mlir_context);
    // TODO(chokobole): Implement this for integer types.
    // Since XLA only supports 2xf32 and 4xf32 floating point vector additions,
    // we don't support this pattern.
    // patterns.add<VectorizeAtomicRMW>(mlir_context, &device_description_);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateVectorizeLoadsAndStoresPass() {
  return std::make_unique<VectorizeLoadsAndStoresPass>();
}

}  // namespace zkx::gpu
