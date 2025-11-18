/* Copyright 2025 The ZKIR Authors.

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

#include "zkir/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h"

#include <utility>

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "zkir/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "zkir/Dialect/TensorExt/IR/TensorExtOps.h"

namespace mlir::zkir::tensor_ext {

#define GEN_PASS_DEF_TENSOREXTTOTENSOR
#include "zkir/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h.inc"

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
    unsigned numCoeffs = tensorType.getShape()[0];
    assert(llvm::has_single_bit(numCoeffs) &&
           "expected the number of coefficients to be a power of 2");
    unsigned indexBitWidth = llvm::countr_zero(numCoeffs);

    auto c0 = b.create<arith::ConstantIndexOp>(0);
    auto c1 = b.create<arith::ConstantIndexOp>(1);
    auto cN = b.create<arith::ConstantIndexOp>(numCoeffs);
    auto sourceMemref =
        b.create<bufferization::ToBufferOp>(memrefType, adaptor.getSource(),
                                            /*read_only=*/true);
    auto destMemref =
        b.create<bufferization::ToBufferOp>(memrefType, adaptor.getDest());
    auto parallelOp = b.create<scf::ParallelOp>(
        /*lowerBound=*/ValueRange{c0},
        /*upperBound=*/ValueRange{cN},
        /*steps=*/ValueRange{c1},
        /*bodyBuilder=*/
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          ImplicitLocOpBuilder nb(nestedLoc, nestedBuilder);

          auto index = nb.create<arith::IndexCastUIOp>(
              nb.getIntegerType(indexBitWidth), args[0]);
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
                auto e1 = thenBuilder.create<memref::LoadOp>(
                    sourceMemref, ValueRange{args[0]});
                auto e2 = thenBuilder.create<memref::LoadOp>(
                    sourceMemref, ValueRange{bitReversedIndex});
                thenBuilder.create<memref::StoreOp>(
                    e1, destMemref, ValueRange{bitReversedIndex});
                thenBuilder.create<memref::StoreOp>(e2, destMemref,
                                                    ValueRange{args[0]});
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
  target.addLegalDialect<tensor::TensorDialect>();
  target.addLegalDialect<memref::MemRefDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<bufferization::BufferizationDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalDialect<scf::SCFDialect>();
  RewritePatternSet patterns(context);

  patterns.add<ConvertBitReverse>(context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace mlir::zkir::tensor_ext
