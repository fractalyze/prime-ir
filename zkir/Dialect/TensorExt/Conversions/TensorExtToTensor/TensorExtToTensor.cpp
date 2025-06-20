#include "zkir/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h"

#include <tuple>
#include <utility>

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "zkir/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "zkir/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "zkir/Dialect/TensorExt/IR/TensorExtOps.h"

namespace mlir::zkir::tensor_ext {

#define GEN_PASS_DEF_TENSOREXTTOTENSOR
#include "zkir/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h.inc"

struct ConvertBitReverse : public OpConversionPattern<BitReverseOp> {
  explicit ConvertBitReverse(MLIRContext *context)
      : OpConversionPattern<BitReverseOp>(context) {}
  LogicalResult matchAndRewrite(
      BitReverseOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto tensorType = cast<RankedTensorType>(adaptor.getInput().getType());
    MemRefType memrefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    unsigned numCoeffs = tensorType.getShape()[0];
    assert(llvm::has_single_bit(numCoeffs) &&
           "expected the number of coefficients to be a power of 2");
    unsigned indexBitWidth = llvm::countr_zero(numCoeffs);

    auto indicesAttr = BitReverseIndicesAttr::get(
        IntegerAttr::get(b.getIndexType(), indexBitWidth));
    auto indices = b.create<arith::ConstantOp>(indicesAttr.getIndicesType(),
                                               indicesAttr.getIndices());
    auto numSwaps =
        b.create<arith::ConstantIndexOp>(indicesAttr.getIndices().size());
    auto c0 = b.create<arith::ConstantIndexOp>(0);
    auto c1 = b.create<arith::ConstantIndexOp>(1);
    auto c2 = b.create<arith::ConstantIndexOp>(2);
    auto inputMemref =
        b.create<bufferization::ToMemrefOp>(memrefType, op.getInput());
    b.create<scf::ParallelOp>(
        /*lowerBound=*/ValueRange{c0},
        /*lowerBound=*/ValueRange{numSwaps},
        /*steps=*/ValueRange{c2},
        /*bodyBuilder=*/
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          ImplicitLocOpBuilder nb(nestedLoc, nestedBuilder);
          auto fromIndex = args[0];
          auto toIndex = nb.create<arith::AddIOp>(fromIndex, c1);
          auto i1 =
              nb.create<tensor::ExtractOp>(indices, ValueRange{fromIndex});
          auto i2 = nb.create<tensor::ExtractOp>(indices, ValueRange{toIndex});
          auto e1 = nb.create<memref::LoadOp>(inputMemref, ValueRange{i1});
          auto e2 = nb.create<memref::LoadOp>(inputMemref, ValueRange{i2});
          nb.create<memref::StoreOp>(e1, inputMemref, ValueRange{i2});
          nb.create<memref::StoreOp>(e2, inputMemref, ValueRange{i1});
        });
    auto result = b.create<bufferization::ToTensorOp>(
        tensorType, inputMemref, /*restrict=*/true, /*writable=*/true);
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
  target.addLegalDialect<scf::SCFDialect>();
  RewritePatternSet patterns(context);

  patterns.add<ConvertBitReverse>(context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace mlir::zkir::tensor_ext
