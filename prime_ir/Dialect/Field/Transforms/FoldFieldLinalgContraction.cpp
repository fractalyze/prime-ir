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

#include "prime_ir/Dialect/Field/Transforms/FoldFieldLinalgContraction.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"

namespace mlir::prime_ir::field {

#define GEN_PASS_DEF_FOLDFIELDLINALGCONTRACTION
#include "prime_ir/Dialect/Field/Transforms/FoldFieldLinalgContraction.h.inc"

namespace {

// Pattern to fold linalg.matvec when the matrix operand is a field.constant.
// Transforms: out[i] = Σⱼ matrix[i,j] * vec[j]
// Into: out[i] = c₀ᵢ * x[0] + c₁ᵢ * x[1] + ...
struct FoldMatvecWithConstantMatrix : OpRewritePattern<linalg::MatvecOp> {
  FoldMatvecWithConstantMatrix(MLIRContext *context, unsigned maxUnrollSize)
      : OpRewritePattern<linalg::MatvecOp>(context),
        maxUnrollSize(maxUnrollSize) {}

  LogicalResult matchAndRewrite(linalg::MatvecOp op,
                                PatternRewriter &rewriter) const override {
    // Get operands: matrix (MxN), vec (N), output (M)
    Value matrix = op.getInputs()[0];
    Value vec = op.getInputs()[1];
    Value output = op.getOutputs()[0];

    // Check if matrix is a field.constant
    auto matrixConst = matrix.getDefiningOp<ConstantOp>();
    if (!matrixConst)
      return failure();

    // Get the dense attribute from the constant
    auto denseAttr = dyn_cast<DenseIntElementsAttr>(matrixConst.getValue());
    if (!denseAttr)
      return failure();

    // Get matrix shape
    auto matrixType = cast<RankedTensorType>(matrix.getType());
    if (matrixType.getRank() != 2)
      return failure();

    int64_t numRows = matrixType.getShape()[0];
    int64_t numCols = matrixType.getShape()[1];

    // Check size limit
    if (static_cast<unsigned>(numCols) > maxUnrollSize)
      return failure();

    Location loc = op.getLoc();
    Type elementType = matrixType.getElementType();

    // Extract vector elements
    SmallVector<Value> vecElements;
    for (int64_t j = 0; j < numCols; ++j) {
      Value idx = rewriter.create<arith::ConstantIndexOp>(loc, j);
      Value elem = rewriter.create<tensor::ExtractOp>(loc, vec, idx);
      vecElements.push_back(elem);
    }

    // Get the constant coefficients
    auto coeffValues = denseAttr.getValues<APInt>();

    // Compute each output element
    SmallVector<Value> resultElements;
    for (int64_t i = 0; i < numRows; ++i) {
      Value sum;
      bool hasTerms = false;

      for (int64_t j = 0; j < numCols; ++j) {
        APInt coeff = coeffValues[{static_cast<uint64_t>(i),
                                   static_cast<uint64_t>(j)}];

        // Skip zero coefficients
        if (coeff.isZero())
          continue;

        Value term;
        if (coeff.isOne()) {
          // Optimization: 1 * x = x
          term = vecElements[j];
        } else {
          // Create constant for coefficient
          Value coeffVal = rewriter.create<ConstantOp>(
              loc, elementType,
              IntegerAttr::get(
                  cast<PrimeFieldType>(elementType).getStorageType(), coeff));
          term = rewriter.create<MulOp>(loc, coeffVal, vecElements[j]);
        }

        if (!hasTerms) {
          sum = term;
          hasTerms = true;
        } else {
          sum = rewriter.create<AddOp>(loc, sum, term);
        }
      }

      // Handle the case where all coefficients are zero
      if (!hasTerms) {
        sum = rewriter.create<ConstantOp>(
            loc, elementType,
            IntegerAttr::get(
                cast<PrimeFieldType>(elementType).getStorageType(), 0));
      }

      resultElements.push_back(sum);
    }

    // Build result tensor using tensor.from_elements
    auto resultType = cast<RankedTensorType>(output.getType());
    Value result =
        rewriter.create<tensor::FromElementsOp>(loc, resultType, resultElements);

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  unsigned maxUnrollSize;
};

// Pattern to fold linalg.dot when one operand is a field.constant vector.
// Transforms: result = Σᵢ vec1[i] * vec2[i]
// Into: result = c₀ * x[0] + c₁ * x[1] + ...
struct FoldDotWithConstantVector : OpRewritePattern<linalg::DotOp> {
  FoldDotWithConstantVector(MLIRContext *context, unsigned maxUnrollSize)
      : OpRewritePattern<linalg::DotOp>(context),
        maxUnrollSize(maxUnrollSize) {}

  LogicalResult matchAndRewrite(linalg::DotOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getInputs()[0];
    Value rhs = op.getInputs()[1];

    // Try to find a constant operand (either lhs or rhs)
    auto lhsConst = lhs.getDefiningOp<ConstantOp>();
    auto rhsConst = rhs.getDefiningOp<ConstantOp>();

    ConstantOp constOp;
    Value varVec;
    if (lhsConst) {
      constOp = lhsConst;
      varVec = rhs;
    } else if (rhsConst) {
      constOp = rhsConst;
      varVec = lhs;
    } else {
      return failure();
    }

    // Get the dense attribute from the constant
    auto denseAttr = dyn_cast<DenseIntElementsAttr>(constOp.getValue());
    if (!denseAttr)
      return failure();

    // Get vector shape
    auto vecType = cast<RankedTensorType>(constOp.getType());
    if (vecType.getRank() != 1)
      return failure();

    int64_t vecSize = vecType.getShape()[0];

    // Check size limit
    if (static_cast<unsigned>(vecSize) > maxUnrollSize)
      return failure();

    Location loc = op.getLoc();
    Type elementType = vecType.getElementType();

    // Get the constant coefficients
    auto coeffValues = denseAttr.getValues<APInt>();

    // Compute the dot product
    Value sum;
    bool hasTerms = false;

    for (int64_t i = 0; i < vecSize; ++i) {
      APInt coeff = coeffValues[static_cast<uint64_t>(i)];

      // Skip zero coefficients
      if (coeff.isZero())
        continue;

      // Extract variable vector element
      Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value varElem = rewriter.create<tensor::ExtractOp>(loc, varVec, idx);

      Value term;
      if (coeff.isOne()) {
        // Optimization: 1 * x = x
        term = varElem;
      } else {
        // Create constant for coefficient
        Value coeffVal = rewriter.create<ConstantOp>(
            loc, elementType,
            IntegerAttr::get(
                cast<PrimeFieldType>(elementType).getStorageType(), coeff));
        term = rewriter.create<MulOp>(loc, coeffVal, varElem);
      }

      if (!hasTerms) {
        sum = term;
        hasTerms = true;
      } else {
        sum = rewriter.create<AddOp>(loc, sum, term);
      }
    }

    // Handle the case where all coefficients are zero
    if (!hasTerms) {
      sum = rewriter.create<ConstantOp>(
          loc, elementType,
          IntegerAttr::get(
              cast<PrimeFieldType>(elementType).getStorageType(), 0));
    }

    // Wrap the scalar result in a 0-D tensor
    auto resultType = cast<RankedTensorType>(op.getOutputs()[0].getType());
    Value result =
        rewriter.create<tensor::FromElementsOp>(loc, resultType, sum);

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  unsigned maxUnrollSize;
};

} // namespace

struct FoldFieldLinalgContraction
    : impl::FoldFieldLinalgContractionBase<FoldFieldLinalgContraction> {
  using FoldFieldLinalgContractionBase::FoldFieldLinalgContractionBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<FoldMatvecWithConstantMatrix>(context, maxUnrollSize);
    patterns.add<FoldDotWithConstantVector>(context, maxUnrollSize);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::prime_ir::field
