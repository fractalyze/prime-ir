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

#include "prime_ir/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h"

#include <utility>

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "prime_ir/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "prime_ir/Dialect/TensorExt/IR/TensorExtOps.h"

namespace mlir::prime_ir::tensor_ext {

#define GEN_PASS_DEF_TENSOREXTTOTENSOR
#include "prime_ir/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h.inc"

struct ConvertBitReverse : public OpConversionPattern<BitReverseOp> {
  explicit ConvertBitReverse(MLIRContext *context)
      : OpConversionPattern<BitReverseOp>(context) {}
  LogicalResult
  matchAndRewrite(BitReverseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto tensorType = cast<RankedTensorType>(adaptor.getSource().getType());
    MemRefType memrefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    int64_t rank = tensorType.getRank();
    int64_t dimension = op.getDimension();

    // Validate dimension
    assert(dimension >= 0 && dimension < rank && "dimension out of range");

    unsigned numElements = tensorType.getShape()[dimension];
    assert(llvm::has_single_bit(numElements) &&
           "expected the size of the dimension to be a power of 2");
    unsigned indexBitWidth = llvm::countr_zero(numElements);

    auto c0 = arith::ConstantIndexOp::create(b, 0);
    auto c1 = arith::ConstantIndexOp::create(b, 1);

    // Create upper bounds for all dimensions
    SmallVector<Value> lowerBounds(rank, c0);
    SmallVector<Value> upperBounds;
    SmallVector<Value> steps(rank, c1);
    for (int64_t i = 0; i < rank; ++i) {
      upperBounds.push_back(
          arith::ConstantIndexOp::create(b, tensorType.getShape()[i]));
    }

    auto sourceMemref =
        bufferization::ToBufferOp::create(b, memrefType, adaptor.getSource(),
                                          /*read_only=*/true);
    auto destMemref =
        bufferization::ToBufferOp::create(b, memrefType, adaptor.getDest());

    scf::ParallelOp::create(
        b, /*lowerBound=*/lowerBounds,
        /*upperBound=*/upperBounds,
        /*steps=*/steps,
        /*bodyBuilder=*/
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          ImplicitLocOpBuilder nb(nestedLoc, nestedBuilder);

          // Compute bit-reversed index for the target dimension
          auto index = arith::IndexCastUIOp::create(
              nb, nb.getIntegerType(indexBitWidth), args[dimension]);
          auto bitReversed = LLVM::BitReverseOp::create(nb, index);
          auto isGE = arith::CmpIOp::create(nb, arith::CmpIPredicate::uge,
                                            bitReversed, index);
          scf::IfOp::create(
              nb, isGE, /*thenBuilder=*/
              [&](OpBuilder &thenB, Location thenLoc) {
                ImplicitLocOpBuilder thenBuilder(thenLoc, thenB);
                auto bitReversedIndex = arith::IndexCastUIOp::create(
                    thenBuilder, nb.getIndexType(), bitReversed);

                // Build indices for source position and bit-reversed position
                SmallVector<Value> revIndices(args.begin(), args.end());
                revIndices[dimension] = bitReversedIndex;

                auto e1 =
                    memref::LoadOp::create(thenBuilder, sourceMemref, args);
                auto e2 = memref::LoadOp::create(thenBuilder, sourceMemref,
                                                 revIndices);
                memref::StoreOp::create(thenBuilder, e1, destMemref,
                                        revIndices);
                memref::StoreOp::create(thenBuilder, e2, destMemref, args);
                scf::YieldOp::create(thenBuilder);
              });
        });

    auto result = bufferization::ToTensorOp::create(
        b, tensorType, destMemref, /*restrict=*/true, /*writable=*/true);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct TensorExtToTensor : impl::TensorExtToTensorBase<TensorExtToTensor> {
  using TensorExtToTensorBase::TensorExtToTensorBase;

  void runOnOperation() override;
};

void TensorExtToTensor::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  ConversionTarget target(*context);

  target.addIllegalDialect<TensorExtDialect>();
  target.addLegalDialect<
      // clang-format off
      arith::ArithDialect,
      bufferization::BufferizationDialect,
      LLVM::LLVMDialect,
      memref::MemRefDialect,
      scf::SCFDialect
      // clang-format on
      >();
  RewritePatternSet patterns(context);

  patterns.add<ConvertBitReverse>(context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace mlir::prime_ir::tensor_ext
