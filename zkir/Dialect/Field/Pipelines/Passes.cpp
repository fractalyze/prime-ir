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

#include "zkir/Dialect/Field/Pipelines/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
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
#include "zkir/Dialect/ArithExt/Conversions/SpecializeArithToAVX/SpecializeArithToAVX.h"
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToLLVM/EllipticCurveToLLVM.h"
#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h"
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"
#include "zkir/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h"

//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

namespace mlir::zkir::field {

void buildFieldToLLVM(OpPassManager &pm, const FieldToLLVMOptions &options) {
  pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());

  // If we convert elementwise to linalg, tensor folding in ModArithDialect will
  // not work.
  pm.addPass(createFieldToModArith());
  pm.addPass(createCanonicalizerPass());

  pm.addNestedPass<func::FuncOp>(createConvertElementwiseToLinalgPass());
  pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());

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

  pm.addPass(affine::createLoopFusionPass());
  pm.addPass(affine::createRaiseMemrefToAffine());
  pm.addNestedPass<func::FuncOp>(affine::createLoopUnrollPass());
  pm.addPass(createInlinerPass());
  pm.addPass(affine::createAffineScalarReplacementPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createLowerAffinePass());

  pm.addNestedPass<func::FuncOp>(createConvertLinalgToParallelLoopsPass());
  pm.addPass(createLowerAffinePass());

  if (options.enableOpenMP) {
    pm.addPass(createConvertSCFToOpenMPPass());
  }

  pm.addNestedPass<func::FuncOp>(memref::createExpandStridedMetadataPass());
  // Expand strided metadata can introduce affine ops so we need to lower them
  // again.
  pm.addPass(createLowerAffinePass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createSCFToControlFlowPass());
  if (options.specializeAVX) {
    pm.addPass(arith_ext::createSpecializeArithToAVX());
  }
  pm.addPass(createConvertToLLVMPass());
  pm.addPass(createCanonicalizerPass());
}

void buildFieldToGPU(OpPassManager &pm, const FieldToGPUOptions &options) {
  pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
  pm.addNestedPass<func::FuncOp>(createConvertElementwiseToLinalgPass());
  pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
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
    pm.addPass(createLowerAffinePass());
  }
  pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
  pm.addNestedPass<func::FuncOp>(createConvertAffineForToGPUPass());
  // -gpu-map-parallel-loops greedily maps loops to GPU hardware dimensions if
  // it's not already mapped.
  pm.addPass(createGpuMapParallelLoopsPass());
  pm.addNestedPass<func::FuncOp>(createConvertParallelLoopToGpuPass());
  pm.addPass(elliptic_curve::createEllipticCurveToLLVM());
  pm.addPass(createGpuKernelOutliningPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createGpuDecomposeMemrefsPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  // Expand strided metadata can introduce affine ops so we need to lower them
  // again.
  pm.addPass(createLowerAffinePass());
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
  pm.addPass(createConvertToLLVMPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(
      createGpuModuleToBinaryPass(options.gpuModuleToBinaryPassOptions()));
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

} // namespace mlir::zkir::field
