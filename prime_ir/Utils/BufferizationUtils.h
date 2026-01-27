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

#ifndef PRIME_IR_UTILS_BUFFERIZATIONUTILS_H_
#define PRIME_IR_UTILS_BUFFERIZATIONUTILS_H_

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::prime_ir {

//===----------------------------------------------------------------------===//
// BitcastOpInterface - Shared implementation for bitcast bufferization
//===----------------------------------------------------------------------===//

// Template for bufferizable bitcast operations.
// Works with any dialect's BitcastOp that has getInput() and getType()
// methods.
template <typename BitcastOpT>
struct BitcastOpBufferizableInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          BitcastOpBufferizableInterface<BitcastOpT>, BitcastOpT> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const bufferization::AnalysisState &state) const {
    // Bitcast doesn't read the tensor contents, just reinterprets the memory.
    return false;
  }

  bool
  bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                          const bufferization::AnalysisState &state) const {
    return false;
  }

  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opOperand,
                    const bufferization::AnalysisState &state) const {
    // The result aliases the input - they share the same underlying memory.
    return {{op->getResult(0), bufferization::BufferRelation::Equivalent}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto bitcastOp = cast<BitcastOpT>(op);

    // Get the buffer for the input tensor.
    FailureOr<Value> inputBuffer = bufferization::getBuffer(
        rewriter, bitcastOp.getInput(), options, state);
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

    // Create a new bitcast with memref types.
    auto memrefBitcast = rewriter.create<BitcastOpT>(
        op->getLoc(), outputMemRefType, *inputBuffer);

    bufferization::replaceOpWithBufferizedValues(rewriter, op,
                                                 memrefBitcast.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConstantOpInterface - Shared implementation for constant bufferization
//===----------------------------------------------------------------------===//

// Template for bufferizable constant operations with tensor types.
// Works with dialects that have ConstantOp and BitcastOp where the element
// type has a getStorageType() method.
//
// Template parameters:
// - ConstantOpT: The constant op type (e.g., field::ConstantOp)
// - BitcastOpT: The bitcast op type (e.g., field::BitcastOp)
// - ElementTypeT: The element type (e.g., PrimeFieldType, ModArithType)
// - globalPrefix: Prefix for generated global names
template <typename ConstantOpT, typename BitcastOpT, typename ElementTypeT,
          const char *globalPrefix>
struct ConstantOpBufferizableInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          ConstantOpBufferizableInterface<ConstantOpT, BitcastOpT, ElementTypeT,
                                          globalPrefix>,
          ConstantOpT> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const bufferization::AnalysisState &state) const {
    return false;
  }

  bool
  bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                          const bufferization::AnalysisState &state) const {
    return false;
  }

  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opOperand,
                    const bufferization::AnalysisState &state) const {
    return {};
  }

  bool isWritable(Operation *op, Value value,
                  const bufferization::AnalysisState &state) const {
    // Global memory is read-only.
    return false;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto constantOp = cast<ConstantOpT>(op);

    auto tensorType = dyn_cast<RankedTensorType>(constantOp.getType());
    if (!tensorType) {
      // Scalar constant, nothing to bufferize.
      return success();
    }

    // Get the dense elements attribute.
    auto denseAttr = dyn_cast<DenseElementsAttr>(constantOp.getValue());
    if (!denseAttr) {
      return op->emitOpError("expected dense elements attribute for tensor");
    }

    // Get the element type and its underlying storage type.
    auto elementType = dyn_cast<ElementTypeT>(tensorType.getElementType());
    if (!elementType) {
      return op->emitOpError("expected supported element type");
    }
    auto storageType = elementType.getStorageType();

    // Create storage tensor type for the global (memref.global doesn't support
    // custom types).
    auto storageTensorType =
        RankedTensorType::get(tensorType.getShape(), storageType);
    auto storageMemrefType =
        MemRefType::get(tensorType.getShape(), storageType);

    // Create the target memref type for the result.
    auto targetMemrefType = MemRefType::get(tensorType.getShape(), elementType);

    // Find or create a module-level global for this constant.
    auto moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      return op->emitOpError("expected to be within a module");
    }

    // Generate a unique name for the global.
    SymbolTable symbolTable(moduleOp);
    unsigned counter = 0;
    auto uniqueName = SymbolTable::generateSymbolName<64>(
        globalPrefix,
        [&](StringRef candidate) { return symbolTable.lookup(candidate); },
        counter);
    std::string globalName(uniqueName.data(), uniqueName.size());

    // Create a new dense attribute with storage type from raw buffer.
    auto storageDenseAttr = DenseElementsAttr::getFromRawBuffer(
        storageTensorType, denseAttr.getRawData());

    // Insert the global at the beginning of the module.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    auto globalOp = rewriter.create<memref::GlobalOp>(
        op->getLoc(), globalName,
        /*sym_visibility=*/rewriter.getStringAttr("private"),
        /*type=*/storageMemrefType,
        /*initial_value=*/storageDenseAttr,
        /*constant=*/true,
        /*alignment=*/nullptr);

    // Replace the constant with a get_global + bitcast operation.
    rewriter.setInsertionPoint(op);
    auto getGlobalOp = rewriter.create<memref::GetGlobalOp>(
        op->getLoc(), storageMemrefType, globalOp.getSymName());

    // Bitcast from storage memref to target memref.
    auto bitcastOp = rewriter.create<BitcastOpT>(op->getLoc(), targetMemrefType,
                                                 getGlobalOp.getResult());

    bufferization::replaceOpWithBufferizedValues(rewriter, op,
                                                 bitcastOp.getResult());
    return success();
  }
};

// Global name prefixes for constants.
inline constexpr char kFieldConstantPrefix[] = "__field_constant_";
inline constexpr char kModArithConstantPrefix[] = "__mod_arith_constant_";

} // namespace mlir::prime_ir

#endif // PRIME_IR_UTILS_BUFFERIZATIONUTILS_H_
