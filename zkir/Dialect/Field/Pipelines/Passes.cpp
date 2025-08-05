#include "zkir/Dialect/Field/Pipelines/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h"
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"
#include "zkir/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h"

//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

namespace mlir::zkir::field {

void buildFieldToLLVM(OpPassManager &pm, const FieldToLLVMOptions &options) {
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::createLinalgGeneralizeNamedOpsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::createConvertElementwiseToLinalgPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::createLinalgElementwiseOpFusionPass());
  pm.addPass(createFieldToModArith());
  pm.addPass(createCanonicalizerPass());

  pm.addPass(mod_arith::createModArithToArith());
  pm.addPass(createCanonicalizerPass());

  pm.addPass(tensor_ext::createTensorExtToTensor());

  pm.addPass(bufferization::createOneShotBufferizePass(
      options.bufferizationOptions()));
  pm.addPass(createCanonicalizerPass());

  if (options.bufferResultsToOutParams) {
    pm.addPass(bufferization::createBufferResultsToOutParamsPass(
        options.bufferResultsToOutParamsOptions()));
  }

  pm.addNestedPass<func::FuncOp>(createConvertLinalgToParallelLoopsPass());
  pm.addPass(createLowerAffinePass());

  if (options.enableOpenMP) {
    pm.addPass(createConvertSCFToOpenMPPass());
  }

  pm.addNestedPass<func::FuncOp>(memref::createExpandStridedMetadataPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createConvertToLLVMPass());
  pm.addPass(createCanonicalizerPass());
}

void buildFieldToGPU(OpPassManager &pm, const FieldToGPUOptions &options) {
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::createLinalgGeneralizeNamedOpsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::createConvertElementwiseToLinalgPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::createLinalgElementwiseOpFusionPass());
  pm.addPass(createFieldToModArith());
  pm.addPass(createCanonicalizerPass());

  pm.addPass(mod_arith::createModArithToArith());
  pm.addPass(createCanonicalizerPass());

  pm.addPass(tensor_ext::createTensorExtToTensor());

  pm.addPass(bufferization::createOneShotBufferizePass(
      options.bufferizationOptions()));
  pm.addPass(createCanonicalizerPass());

  if (options.bufferResultsToOutParams) {
    pm.addPass(bufferization::createBufferResultsToOutParamsPass(
        options.bufferResultsToOutParamsOptions()));
  }

  pm.addPass(createConvertLinalgToAffineLoopsPass());
  // FIXME(batzor): 1-D `affine::ForOp` is making the GPU conversion pass to
  // fail so I added this pass as a temporary workaround. Due to this, some
  // VecOps will not be lowered to GPU dialect.
  if (options.parallelizeAffine) {
    pm.addPass(affine::createAffineParallelize());
  }
  pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
  pm.addNestedPass<func::FuncOp>(createConvertAffineForToGPUPass());
  // -gpu-map-parallel-loops greedily maps loops to GPU hardware dimensions if
  // it's not already mapped.
  pm.addPass(createGpuMapParallelLoopsPass());
  pm.addNestedPass<func::FuncOp>(createConvertParallelLoopToGpuPass());
  pm.addPass(createGpuKernelOutliningPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createGpuDecomposeMemrefsPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(memref::createNormalizeMemRefsPass());

  pm.addNestedPass<gpu::GPUModuleOp>(
      createConvertGpuOpsToNVVMOps(options.convertGpuOpsToNVVMOpsOptions()));

  pm.addPass(createGpuNVVMAttachTarget(options.targetOptions()));

  pm.addPass(createConvertNVVMToLLVMPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createReconcileUnrealizedCastsPass());

  GpuToLLVMConversionPassOptions opt;
  opt.hostBarePtrCallConv = options.nvvmUseBarePtrCallConv;
  opt.kernelBarePtrCallConv = options.nvvmUseBarePtrCallConv;
  pm.addPass(createGpuToLLVMConversionPass(opt));
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(
      createGpuModuleToBinaryPass(options.gpuModuleToBinaryPassOptions()));
  pm.addPass(createConvertToLLVMPass());
  pm.addPass(createCanonicalizerPass());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerFieldPipelines() {
  PassPipelineRegistration<FieldToLLVMOptions>(
      "field-to-llvm",
      "The standard pipeline for taking field-agnostic IR using the"
      " field type, and lowering it to LLVM IR with concrete"
      " representations and algorithms for fields.",
      buildFieldToLLVM);

  PassPipelineRegistration<FieldToGPUOptions>(
      "field-to-gpu",
      "The standard pipeline for taking field-agnostic IR using the"
      " field type, and lowering it to GPU+LLVM IR with concrete"
      " representations and algorithms for fields.",
      buildFieldToGPU);
}

}  // namespace mlir::zkir::field
