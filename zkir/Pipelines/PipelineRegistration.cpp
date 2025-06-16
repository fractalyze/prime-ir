#include "zkir/Pipelines/PipelineRegistration.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h"
#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h"
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"
#include "zkir/Dialect/Poly/Conversions/PolyToField/PolyToField.h"

using mlir::func::FuncOp;

namespace mlir::zkir::pipelines {

void oneShotBufferize(OpPassManager &manager) {
  // NOTE: One-shot bufferize does not deallocate buffers. This is done by the
  // ownership-based buffer deallocation pass.
  // https://mlir.llvm.org/docs/OwnershipBasedBufferDeallocation/
  bufferization::OneShotBufferizePassOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  manager.addPass(
      bufferization::createOneShotBufferizePass(bufferizationOptions));
  manager.addPass(memref::createExpandReallocPass());
  manager.addPass(bufferization::createOwnershipBasedBufferDeallocationPass());
  manager.addPass(createCanonicalizerPass());
  manager.addPass(bufferization::createBufferDeallocationSimplificationPass());
  manager.addPass(bufferization::createLowerDeallocationsPass());
  manager.addPass(createCSEPass());
  manager.addPass(createConvertBufferizationToMemRefPass());
  manager.addPass(createCanonicalizerPass());
}

template <bool allowOpenMP>
void ellipticCurveToLLVMPipelineBuilder(OpPassManager &manager) {
  manager.addNestedPass<FuncOp>(createConvertLinalgToParallelLoopsPass());
  manager.addPass(elliptic_curve::createEllipticCurveToField());
  fieldToLLVMPipelineBuilder<allowOpenMP>(manager);
}

template <bool allowOpenMP>
void polyToLLVMPipelineBuilder(OpPassManager &manager) {
  manager.addPass(poly::createPolyToField());
  fieldToLLVMPipelineBuilder<allowOpenMP>(manager);
}

template <bool allowOpenMP>
void fieldToLLVMPipelineBuilder(OpPassManager &manager) {
  manager.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
  manager.addPass(field::createFieldToModArith());
  manager.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
  manager.addPass(mod_arith::createModArithToArith());
  // FIXME(batzor): With this, some memref loads are canonicalized even though
  // it was modified in the middle, causing `poly_ntt_runner` test to fail.
  // manager.addPass(createCanonicalizerPass());

  // Linalg
  manager.addNestedPass<FuncOp>(createConvertElementwiseToLinalgPass());
  manager.addNestedPass<FuncOp>(createLinalgElementwiseOpFusionPass());
  // Needed to lower affine.map and affine.apply
  manager.addNestedPass<FuncOp>(affine::createAffineExpandIndexOpsPass());
  manager.addNestedPass<FuncOp>(affine::createSimplifyAffineStructuresPass());
  manager.addPass(affine::createAffineParallelize());
  manager.addPass(createLowerAffinePass());
  manager.addNestedPass<FuncOp>(memref::createExpandOpsPass());
  manager.addNestedPass<FuncOp>(memref::createExpandStridedMetadataPass());

  // Bufferize
  oneShotBufferize(manager);

  // Linalg must be bufferized before it can be lowered
  // But lowering to loops also re-introduces affine.apply, so re-lower that
  manager.addNestedPass<FuncOp>(createConvertLinalgToParallelLoopsPass());
  manager.addPass(createLowerAffinePass());
  manager.addPass(createConvertBufferizationToMemRefPass());

  // Cleanup
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createSCCPPass());
  manager.addPass(createCSEPass());
  manager.addPass(createSymbolDCEPass());

  // ToLLVM
  manager.addPass(arith::createArithExpandOpsPass());
  if constexpr (allowOpenMP) {
    manager.addPass(createConvertSCFToOpenMPPass());
  }
  manager.addNestedPass<FuncOp>(memref::createExpandStridedMetadataPass());
  manager.addPass(createFinalizeMemRefToLLVMConversionPass());
  manager.addPass(createSCFToControlFlowPass());

  // expand strided metadata will create affine map. Needed to lower affine.map
  // and affine.apply
  manager.addNestedPass<FuncOp>(affine::createAffineExpandIndexOpsPass());
  manager.addNestedPass<FuncOp>(affine::createSimplifyAffineStructuresPass());
  manager.addPass(affine::createAffineParallelize());
  manager.addPass(createLowerAffinePass());
  manager.addPass(createConvertToLLVMPass());

  // Cleanup
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createSCCPPass());
  manager.addPass(createCSEPass());
  manager.addPass(createSymbolDCEPass());
}

template void fieldToLLVMPipelineBuilder<false>(OpPassManager &manager);
template void fieldToLLVMPipelineBuilder<true>(OpPassManager &manager);
template void polyToLLVMPipelineBuilder<false>(OpPassManager &manager);
template void polyToLLVMPipelineBuilder<true>(OpPassManager &manager);
template void ellipticCurveToLLVMPipelineBuilder<false>(OpPassManager &manager);
template void ellipticCurveToLLVMPipelineBuilder<true>(OpPassManager &manager);

}  // namespace mlir::zkir::pipelines
