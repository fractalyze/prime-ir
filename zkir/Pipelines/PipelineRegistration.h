#ifndef ZKIR_PIPELINES_PIPELINEREGISTRATION_H_
#define ZKIR_PIPELINES_PIPELINEREGISTRATION_H_

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::zkir::pipelines {

void oneShotBufferize(OpPassManager &manager);

template <bool allowOpenMP>
void fieldToLLVMPipelineBuilder(OpPassManager &manager);

template <bool allowOpenMP>
void polyToLLVMPipelineBuilder(OpPassManager &manager);

template <bool allowOpenMP>
void ellipticCurveToLLVMPipelineBuilder(OpPassManager &manager);

extern template void fieldToLLVMPipelineBuilder<false>(OpPassManager &manager);
extern template void fieldToLLVMPipelineBuilder<true>(OpPassManager &manager);
extern template void polyToLLVMPipelineBuilder<false>(OpPassManager &manager);
extern template void polyToLLVMPipelineBuilder<true>(OpPassManager &manager);
extern template void ellipticCurveToLLVMPipelineBuilder<false>(
    OpPassManager &manager);
extern template void ellipticCurveToLLVMPipelineBuilder<true>(
    OpPassManager &manager);

}  // namespace mlir::zkir::pipelines

#endif  // ZKIR_PIPELINES_PIPELINEREGISTRATION_H_
