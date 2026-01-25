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

#include "mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h"
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

    auto c0 = b.create<arith::ConstantIndexOp>(0);
    auto c1 = b.create<arith::ConstantIndexOp>(1);

    // Create upper bounds for all dimensions
    SmallVector<Value> lowerBounds(rank, c0);
    SmallVector<Value> upperBounds;
    SmallVector<Value> steps(rank, c1);
    for (int64_t i = 0; i < rank; ++i) {
      upperBounds.push_back(
          b.create<arith::ConstantIndexOp>(tensorType.getShape()[i]));
    }

    auto sourceMemref =
        b.create<bufferization::ToBufferOp>(memrefType, adaptor.getSource(),
                                            /*read_only=*/true);
    auto destMemref =
        b.create<bufferization::ToBufferOp>(memrefType, adaptor.getDest());

    auto parallelOp = b.create<scf::ParallelOp>(
        /*lowerBound=*/lowerBounds,
        /*upperBound=*/upperBounds,
        /*steps=*/steps,
        /*bodyBuilder=*/
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          ImplicitLocOpBuilder nb(nestedLoc, nestedBuilder);

          // Compute bit-reversed index for the target dimension
          auto index = nb.create<arith::IndexCastUIOp>(
              nb.getIntegerType(indexBitWidth), args[dimension]);
          auto bitReversed = nb.create<LLVM::BitReverseOp>(index);
          auto isGE = nb.create<arith::CmpIOp>(arith::CmpIPredicate::sge,
                                               bitReversed, index);
          nb.create<scf::IfOp>(
              isGE, /*thenBuilder=*/
              [&](OpBuilder &thenB, Location thenLoc) {
                ImplicitLocOpBuilder thenBuilder(thenLoc, thenB);
                auto bitReversedIndex =
                    thenBuilder.create<arith::IndexCastUIOp>(nb.getIndexType(),
                                                             bitReversed);

                // Build indices for source position and bit-reversed position
                SmallVector<Value> revIndices(args.begin(), args.end());
                revIndices[dimension] = bitReversedIndex;

                auto e1 =
                    thenBuilder.create<memref::LoadOp>(sourceMemref, args);
                auto e2 = thenBuilder.create<memref::LoadOp>(sourceMemref,
                                                             revIndices);
                thenBuilder.create<memref::StoreOp>(e1, destMemref, revIndices);
                thenBuilder.create<memref::StoreOp>(e2, destMemref, args);
                thenBuilder.create<scf::YieldOp>();
              });
        });

    // Forward GPU mapping attribute if present.
    StringRef gpuMappingAttrName = gpu::getMappingAttrName();
    if (auto gpuMappingAttr = op->getAttr(gpuMappingAttrName)) {
      parallelOp->setAttr(gpuMappingAttrName, gpuMappingAttr);
    }

    auto result = b.create<bufferization::ToTensorOp>(
        tensorType, destMemref, /*restrict=*/true, /*writable=*/true);
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
