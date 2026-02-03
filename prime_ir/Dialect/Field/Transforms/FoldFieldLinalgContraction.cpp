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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "prime_ir/Dialect/Field/IR/FieldOperation.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::prime_ir::field {

#define GEN_PASS_DEF_FOLDFIELDLINALGCONTRACTION
#include "prime_ir/Dialect/Field/Transforms/FoldFieldLinalgContraction.h.inc"

namespace {

// Get the degree of a field type (1 for PrimeField, n for ExtensionField).
unsigned getFieldDegree(Type elementType) {
  if (isa<PrimeFieldType>(elementType))
    return 1;
  return cast<ExtensionFieldType>(elementType).getDegreeOverPrime();
}

// Extract a scalar field element's attribute from a dense tensor attribute.
// For tensor<MxN x PrimeField>: denseAttr has M*N elements
// For tensor<MxN x ExtensionField<degree>>: denseAttr has M*N*degree elements
TypedAttr extractFieldElementAttr(DenseIntElementsAttr denseAttr,
                                  Type elementType, int64_t linearIndex) {
  unsigned degree = getFieldDegree(elementType);
  auto values = denseAttr.getValues<APInt>();

  if (auto pfType = dyn_cast<PrimeFieldType>(elementType)) {
    return IntegerAttr::get(pfType.getStorageType(), values[linearIndex]);
  }

  auto efType = cast<ExtensionFieldType>(elementType);
  SmallVector<APInt> coeffs;
  for (unsigned d = 0; d < degree; ++d) {
    coeffs.push_back(values[linearIndex * degree + d]);
  }
  auto tensorType =
      RankedTensorType::get({static_cast<int64_t>(degree)},
                            efType.getBasePrimeField().getStorageType());
  return DenseIntElementsAttr::get(tensorType, coeffs);
}

// Create a zero constant using
// ConstantLikeInterface::createConstantAttrFromValues.
Value createFieldZero(PatternRewriter &rewriter, Location loc,
                      Type elementType) {
  auto constantLike = cast<ConstantLikeInterface>(elementType);
  return rewriter.create<ConstantOp>(loc, elementType,
                                     constantLike.createConstantAttr(0));
}

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

    // Compute each output element
    SmallVector<Value> resultElements;
    for (int64_t i = 0; i < numRows; ++i) {
      Value sum;
      bool hasTerms = false;

      for (int64_t j = 0; j < numCols; ++j) {
        int64_t linearIndex = i * numCols + j;
        TypedAttr coeffAttr =
            extractFieldElementAttr(denseAttr, elementType, linearIndex);
        FieldOperation coeffOp =
            FieldOperation::fromUnchecked(coeffAttr, elementType);

        // Skip zero coefficients
        if (coeffOp.isZero())
          continue;

        Value term;
        if (coeffOp.isOne()) {
          // Optimization: 1 * x = x
          term = vecElements[j];
        } else {
          // Create constant for coefficient
          Value coeffVal =
              rewriter.create<ConstantOp>(loc, elementType, coeffAttr);
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
        sum = createFieldZero(rewriter, loc, elementType);
      }

      resultElements.push_back(sum);
    }

    // Build result tensor using tensor.from_elements
    auto resultType = cast<RankedTensorType>(output.getType());
    Value result = rewriter.create<tensor::FromElementsOp>(loc, resultType,
                                                           resultElements);

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
      : OpRewritePattern<linalg::DotOp>(context), maxUnrollSize(maxUnrollSize) {
  }

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

    // Compute the dot product
    Value sum;
    bool hasTerms = false;

    for (int64_t i = 0; i < vecSize; ++i) {
      TypedAttr coeffAttr = extractFieldElementAttr(denseAttr, elementType, i);
      FieldOperation coeffOp =
          FieldOperation::fromUnchecked(coeffAttr, elementType);

      // Skip zero coefficients
      if (coeffOp.isZero())
        continue;

      // Extract variable vector element
      Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value varElem = rewriter.create<tensor::ExtractOp>(loc, varVec, idx);

      Value term;
      if (coeffOp.isOne()) {
        // Optimization: 1 * x = x
        term = varElem;
      } else {
        // Create constant for coefficient
        Value coeffVal =
            rewriter.create<ConstantOp>(loc, elementType, coeffAttr);
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
      sum = createFieldZero(rewriter, loc, elementType);
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
