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
#include "mlir/Dialect/Affine/Transforms/Passes.h"
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

  // Must precede ConvertElementwiseToLinalg below: once elementwise field ops
  // are wrapped in linalg.generic, ModArithDialect's tensor folding no longer
  // sees them. That ordering constraint is between these two passes only --
  // it says nothing about where the binary-field lowering goes.
  //
  // Note that field.inverse is deliberately NOT ElementwiseMappable, so
  // ConvertElementwiseToLinalg leaves a shaped inverse alone regardless of
  // order and it reaches this pass whole, where it becomes Montgomery's batch
  // inversion (one inversion plus ~3(N-1) multiplies for N elements). Making
  // it elementwise would trade that for N independent scalar inverses --
  // still correct, and silently orders of magnitude slower.
  // batch_inverse_not_scalarized.mlir pins this.
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
  // Scalarize elementwise field ops BEFORE lowering binary fields.
  // BinaryFieldToArith's emitters are scalar-only -- they truncate to a
  // scalar half-width type -- so a shaped operand cannot be lowered. Running
  // this after the lowering (as it used to) handed the pass shaped tower ops
  // it could not legalize.
  //
  // The rule, exactly: every ElementwiseMappable field op arrives here as the
  // scalar body of a linalg.generic. field.inverse is the deliberate
  // exception -- it is not ElementwiseMappable (see the note above), so a
  // shaped inverse passes through untouched. For prime fields that is the
  // point: FieldToModArith has already turned it into a batch inversion. For
  // BINARY fields there is no such lowering yet, so a shaped binary-field
  // inverse still reaches BinaryFieldToArith whole and fails to legalize.
  // That gap is fractalyze/prime-ir#453 (batch inversion for binary fields),
  // and batch_inverse_not_scalarized.mlir pins today's behaviour for it.
  pm.addNestedPass<func::FuncOp>(createConvertElementwiseToLinalgPass());

  // Binary fields lower directly to arith (not through mod_arith)
  BinaryFieldToArithOptions bfOpts;
  bfOpts.outlineTowerOps = options.outlineTowerOps;
  bfOpts.outlineMinTowerLevel = options.outlineMinTowerLevel;
  pm.addPass(createBinaryFieldToArith(bfOpts));
  // Reconcile unrealized casts from binary field specialization and conversion
  // (e.g., i64 -> bf<6> -> i64 chains from PCLMULQDQ + BinaryFieldToArith)
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(createCanonicalizerPass());

  // LinalgGeneralizeNamedOpsPass uses greedy pattern rewriting with folding.
  // Must run after BinaryFieldToArith to avoid tensor.from_elements folding
  // with binary field types (MLIR's folder doesn't understand custom types).
  pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
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
  // FieldToModArith, there are no outlined functions to
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
