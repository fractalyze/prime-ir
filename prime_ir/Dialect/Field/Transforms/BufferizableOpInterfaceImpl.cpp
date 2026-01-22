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

#include "prime_ir/Dialect/Field/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::prime_ir::field;

namespace mlir::prime_ir::field {
namespace {

// Bufferization of field.bitcast for tensor reinterpret bitcasts.
// This converts tensor<NxEF> <-> tensor<MxPF> bitcasts to their memref
// equivalents, preserving zero-copy semantics.
struct BitcastOpInterface
    : public BufferizableOpInterface::ExternalModel<BitcastOpInterface,
                                                    BitcastOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // Bitcast doesn't read the tensor contents, just reinterprets the memory.
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    // The result aliases the input - they share the same underlying memory.
    return {{op->getResult(0), BufferRelation::Equivalent}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto bitcastOp = cast<BitcastOp>(op);

    // Get the buffer for the input tensor.
    FailureOr<Value> inputBuffer =
        getBuffer(rewriter, bitcastOp.getInput(), options, state);
    if (failed(inputBuffer))
      return failure();

    // Determine the output memref type.
    auto outputTensorType = dyn_cast<RankedTensorType>(bitcastOp.getType());
    if (!outputTensorType) {
      return op->emitOpError("expected ranked tensor output type");
    }

    // Get the memory space from the input buffer.
    auto inputMemRefType = dyn_cast<MemRefType>(inputBuffer->getType());
    if (!inputMemRefType) {
      return op->emitOpError("expected memref input buffer type");
    }

    // Create the output memref type with the same element type as the tensor
    // but with the input buffer's memory space.
    auto outputMemRefType = MemRefType::get(
        outputTensorType.getShape(), outputTensorType.getElementType(),
        /*layout=*/MemRefLayoutAttrInterface(),
        inputMemRefType.getMemorySpace());

    // Create a new field.bitcast with memref types.
    // This will be lowered to LLVM pointer casts in ExtFieldToLLVM.
    auto memrefBitcast = rewriter.create<BitcastOp>(
        op->getLoc(), outputMemRefType, *inputBuffer);

    replaceOpWithBufferizedValues(rewriter, op, memrefBitcast.getResult());
    return success();
  }
};

} // namespace

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, FieldDialect *dialect) {
    BitcastOp::attachInterface<BitcastOpInterface>(*ctx);
  });
}

} // namespace mlir::prime_ir::field
