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

#include "prime_ir/Dialect/ModArith/Transforms/ReduceOptimization/ReduceOptimization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithDialect.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithOps.h"

namespace mlir::prime_ir::mod_arith {

#define GEN_PASS_DEF_REDUCEOPTIMIZATION
#include "prime_ir/Dialect/ModArith/Transforms/ReduceOptimization/ReduceOptimization.h.inc"

namespace {

// Helper to check if a range is fully contained in [0, modulus)
bool isCanonicalRange(LLVM::ConstantRangeAttr rangeAttr, const APInt &modulus) {
  APInt lower = rangeAttr.getLower();
  APInt upper = rangeAttr.getUpper();

  // Check if range is [0, modulus) or a subset
  // Note: ConstantRange uses half-open intervals [lower, upper)
  return !lower.isNegative() && upper.ule(modulus);
}

// Pattern: Eliminate reduce when input is already canonical
// If input_range is within [0, modulus), the reduce is a no-op
struct EliminateCanonicalReduce : public OpRewritePattern<IntrReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IntrReduceOp op,
                                PatternRewriter &rewriter) const override {
    APInt modulus = op.getModulusAttr().getValue();
    LLVM::ConstantRangeAttr rangeAttr = op.getInputRangeAttr();

    // Check if input range is already in [0, modulus)
    if (isCanonicalRange(rangeAttr, modulus)) {
      rewriter.replaceOp(op, op.getInput());
      return success();
    }

    return failure();
  }
};

// Pattern: Merge consecutive reduces
// reduce(reduce(x)) -> reduce(x)
// The inner reduce already produces [0, modulus), so the outer is a no-op.
struct MergeConsecutiveReduces : public OpRewritePattern<IntrReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IntrReduceOp op,
                                PatternRewriter &rewriter) const override {
    // Check if input is from another reduce
    auto inputReduce = op.getInput().getDefiningOp<IntrReduceOp>();
    if (!inputReduce)
      return failure();

    // Verify same modulus
    if (op.getModulusAttr() != inputReduce.getModulusAttr())
      return failure();

    // Inner reduce produces [0, modulus), so outer is a no-op
    rewriter.replaceOp(op, inputReduce.getResult());
    return success();
  }
};

// Pattern: Reduce after MontReduce can be skipped if we chain them
// mont_reduce -> reduce -> use  ==>  mont_reduce -> canonicalize later
// This is an optimization placeholder - currently IntrMontReduceOp always
// outputs [0, 2*modulus) and IntrReduceOp always outputs [0, modulus).
// Future optimization: track ranges through operations to eliminate redundant
// canonicalization.
struct SkipReduceAfterMontReduce : public OpRewritePattern<IntrReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IntrReduceOp op,
                                PatternRewriter &rewriter) const override {
    // Check if input is from IntrMontReduceOp
    auto montReduce = op.getInput().getDefiningOp<IntrMontReduceOp>();
    if (!montReduce)
      return failure();

    // Verify same modulus
    if (op.getModulusAttr() != montReduce.getModulusAttr())
      return failure();

    // The mont_reduce output is [0, 2*modulus).
    // The reduce's input_range should be [0, 2*modulus).
    // This is just a validation - we keep the reduce for now since it's needed
    // for canonicalization. Future work: propagate [0, 2*modulus) and delay
    // canonicalization.
    return failure();
  }
};

// Pattern: Eliminate reduce when input comes from mont_reduce followed
// by arithmetic that keeps the result in [0, 2*modulus).
// This is a placeholder for future optimization work.
struct SkipIntermediateReduceInAddChain
    : public OpRewritePattern<IntrReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IntrReduceOp op,
                                PatternRewriter &rewriter) const override {
    // Check if input is an add
    auto addOp = op.getInput().getDefiningOp<arith::AddIOp>();
    if (!addOp)
      return failure();

    // Check if one operand of add comes from a reduce (which outputs [0, P))
    // or from mont_reduce (which outputs [0, 2P))
    auto lhsReduce = addOp.getLhs().getDefiningOp<IntrReduceOp>();
    auto rhsReduce = addOp.getRhs().getDefiningOp<IntrReduceOp>();
    auto lhsMontReduce = addOp.getLhs().getDefiningOp<IntrMontReduceOp>();
    auto rhsMontReduce = addOp.getRhs().getDefiningOp<IntrMontReduceOp>();

    // Currently this is a placeholder for future optimization
    // The idea is: if we're adding [0, P) + [0, 2P), result is [0, 3P)
    // We could potentially handle this with extended reduction logic
    (void)lhsReduce;
    (void)rhsReduce;
    (void)lhsMontReduce;
    (void)rhsMontReduce;

    return failure();
  }
};

struct ReduceOptimization : impl::ReduceOptimizationBase<ReduceOptimization> {
  using ReduceOptimizationBase::ReduceOptimizationBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<EliminateCanonicalReduce, MergeConsecutiveReduces,
                 SkipReduceAfterMontReduce, SkipIntermediateReduceInAddChain>(
        context);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::prime_ir::mod_arith
