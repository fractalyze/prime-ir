#include "zkir/Dialect/TensorExt/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Operation.h"
#include "zkir/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "zkir/Dialect/TensorExt/IR/TensorExtOps.h"

namespace mlir::zkir::tensor_ext {
namespace {

using namespace bufferization;

/// Bufferization of tensor_ext.bit_reverse.
struct BitReverseOpInterface
    : public DstBufferizableOpInterfaceExternalModel<BitReverseOpInterface,
                                                     tensor_ext::BitReverseOp> {
  bool bufferizesToElementwiseAccess(Operation *op, const AnalysisState &state,
                                     ArrayRef<OpOperand *> opOperands) const {
    // No read happens after writing.
    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    auto bitReverseOp = cast<BitReverseOp>(op);

    // Get source buffer.
    FailureOr<Value> srcMemref =
        getBuffer(rewriter, bitReverseOp.getSource(), options);
    if (failed(srcMemref)) return failure();

    // Get destination buffer.
    FailureOr<Value> dstMemref =
        getBuffer(rewriter, bitReverseOp.getDest(), options);
    if (failed(dstMemref)) return failure();

    MemRefType srcType = cast<MemRefType>(srcMemref->getType());
    unsigned numCoeffs = srcType.getShape()[0];
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
          auto e1 = nb.create<tensor::ExtractOp>(bitReverseOp.getSource(),
                                                 ValueRange{i1});
          auto e2 = nb.create<tensor::ExtractOp>(bitReverseOp.getSource(),
                                                 ValueRange{i2});
          nb.create<memref::StoreOp>(e1, *dstMemref, ValueRange{i2});
          nb.create<memref::StoreOp>(e2, *dstMemref, ValueRange{i1});
        });

    replaceOpWithBufferizedValues(rewriter, op, *dstMemref);
    return success();
  }
};

}  // namespace

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, TensorExtDialect *dialect) {
    BitReverseOp::attachInterface<BitReverseOpInterface>(*ctx);

    // Load additional dialects of which ops may get created.
    ctx->loadDialect<arith::ArithDialect, scf::SCFDialect>();
  });
}
}  // namespace mlir::zkir::tensor_ext
