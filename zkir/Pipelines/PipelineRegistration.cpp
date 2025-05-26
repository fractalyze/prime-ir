#include "zkir/Pipelines/PipelineRegistration.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
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
  bufferization::OneShotBufferizationOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  manager.addPass(
      bufferization::createOneShotBufferizePass(bufferizationOptions));
  manager.addPass(memref::createExpandReallocPass());
  manager.addPass(bufferization::createOwnershipBasedBufferDeallocationPass());
  manager.addPass(createCanonicalizerPass());
  manager.addPass(bufferization::createBufferDeallocationSimplificationPass());
  manager.addPass(bufferization::createLowerDeallocationsPass());
  manager.addPass(createCSEPass());
  manager.addPass(createBufferizationToMemRefPass());
  manager.addPass(createCanonicalizerPass());
}

void ellipticCurveToLLVMPipelineBuilder(OpPassManager &manager) {
  manager.addPass(elliptic_curve::createEllipticCurveToField());
  manager.addPass(field::createFieldToModArith());
  manager.addPass(mod_arith::createModArithToArith());
  manager.addPass(createCanonicalizerPass());

  // Linalg
  manager.addNestedPass<FuncOp>(createConvertElementwiseToLinalgPass());
  // Needed to lower affine.map and affine.apply
  manager.addNestedPass<FuncOp>(affine::createAffineExpandIndexOpsPass());
  manager.addNestedPass<FuncOp>(affine::createSimplifyAffineStructuresPass());
  manager.addPass(createLowerAffinePass());
  manager.addNestedPass<FuncOp>(memref::createExpandOpsPass());
  manager.addNestedPass<FuncOp>(memref::createExpandStridedMetadataPass());

  // Bufferize
  oneShotBufferize(manager);

  // Linalg must be bufferized before it can be lowered
  // But lowering to loops also re-introduces affine.apply, so re-lower that
  manager.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  manager.addPass(createLowerAffinePass());
  manager.addPass(createBufferizationToMemRefPass());

  // Cleanup
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createSCCPPass());
  manager.addPass(createCSEPass());
  manager.addPass(createSymbolDCEPass());

  // ToLLVM
  manager.addPass(arith::createArithExpandOpsPass());
  manager.addPass(createConvertSCFToCFPass());
  manager.addNestedPass<FuncOp>(memref::createExpandStridedMetadataPass());

  // expand strided metadata will create affine map. Needed to lower affine.map
  // and affine.apply
  manager.addNestedPass<FuncOp>(affine::createAffineExpandIndexOpsPass());
  manager.addNestedPass<FuncOp>(affine::createSimplifyAffineStructuresPass());
  manager.addPass(createLowerAffinePass());
  manager.addPass(createConvertToLLVMPass());

  // Cleanup
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createSCCPPass());
  manager.addPass(createCSEPass());
  manager.addPass(createSymbolDCEPass());
}

template <bool allowOpenMP>
void polyToLLVMPipelineBuilder(OpPassManager &manager) {
  manager.addPass(poly::createPolyToField());
  manager.addPass(field::createFieldToModArith());
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
  manager.addPass(createBufferizationToMemRefPass());

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
  manager.addPass(createConvertSCFToCFPass());
  manager.addNestedPass<FuncOp>(memref::createExpandStridedMetadataPass());

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

template void polyToLLVMPipelineBuilder<false>(OpPassManager &manager);
template void polyToLLVMPipelineBuilder<true>(OpPassManager &manager);

}  // namespace mlir::zkir::pipelines
