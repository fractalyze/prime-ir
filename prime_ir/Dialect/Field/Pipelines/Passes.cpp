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

#include "prime_ir/Dialect/Field/Pipelines/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "prime_ir/Dialect/ArithExt/Conversions/SpecializeArithToAVX/SpecializeArithToAVX.h"
#include "prime_ir/Dialect/Field/Conversions/BinaryFieldToArith/BinaryFieldToArith.h"
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h"
#include "prime_ir/Dialect/Field/Conversions/SpecializeBinaryFieldToARM/SpecializeBinaryFieldToARM.h"
#include "prime_ir/Dialect/Field/Conversions/SpecializeBinaryFieldToX86/SpecializeBinaryFieldToX86.h"
#include "prime_ir/Dialect/Field/Transforms/FoldFieldLinalgContraction.h"
#include "prime_ir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"
#include "prime_ir/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h"

//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

namespace mlir::prime_ir::field {

void buildFieldToLLVM(OpPassManager &pm, const FieldToLLVMOptions &options) {
  pm.addNestedPass<func::FuncOp>(createFoldFieldLinalgContraction());
  pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());

  // If we convert elementwise to linalg, tensor folding in ModArithDialect will
  // not work.
  pm.addPass(createFieldToModArith());
  // Specialize binary field operations to GFNI/PCLMULQDQ if enabled (x86)
  if (options.specializeGFNI || options.specializePCLMULQDQ) {
    SpecializeBinaryFieldToX86Options gfniOpts;
    gfniOpts.useGFNI = options.specializeGFNI;
    gfniOpts.usePCLMULQDQ = options.specializePCLMULQDQ;
    pm.addPass(createSpecializeBinaryFieldToX86(gfniOpts));
  }
  // Specialize binary field operations to PMULL if enabled (ARM)
  if (options.specializePMULL) {
    SpecializeBinaryFieldToARMOptions armOpts;
    armOpts.usePMULL = options.specializePMULL;
    pm.addPass(createSpecializeBinaryFieldToARM(armOpts));
  }
  // Binary fields lower directly to arith (not through mod_arith)
  pm.addPass(createBinaryFieldToArith());
  // Reconcile unrealized casts from binary field specialization and conversion
  // (e.g., i64 -> bf<6> -> i64 chains from PCLMULQDQ + BinaryFieldToArith)
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(createCanonicalizerPass());

  // LinalgGeneralizeNamedOpsPass uses greedy pattern rewriting with folding.
  // Must run after BinaryFieldToArith to avoid tensor.from_elements folding
  // with binary field types (MLIR's folder doesn't understand custom types).
  pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
  pm.addNestedPass<func::FuncOp>(createConvertElementwiseToLinalgPass());
  pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());

  pm.addPass(mod_arith::createModArithToArith(
      mod_arith::ModArithToArithOptions{options.lazyReduction}));
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

  // Apply affine super-vectorization if enabled
  if (options.vectorize) {
    affine::AffineVectorizeOptions vectorizeOpts;
    vectorizeOpts.vectorSizes = {static_cast<int64_t>(options.vectorSize)};
    pm.addPass(affine::createAffineVectorize(vectorizeOpts));
  }

  pm.addNestedPass<func::FuncOp>(affine::createLoopUnrollPass());
  // NOTE: The MLIR inliner is intentionally disabled. It inlines ALL callable
  // functions regardless of visibility, which defeats PairingOutliner's
  // strategy of outlining CyclotomicSquare/MulBy034/MulBy014 as shared
  // func.func helpers. In the default "inline" lowering mode for
  // FieldToModArith, there are no IntrinsicFunctionGenerator functions to
  // inline either.
  // pm.addPass(createInlinerPass());
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
  // Convert vector ops to LLVM (needed when vectorization is enabled)
  if (options.vectorize) {
    pm.addPass(createConvertVectorToLLVMPass());
  }
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
}

} // namespace mlir::prime_ir::field
