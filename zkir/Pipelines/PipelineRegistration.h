#ifndef ZKIR_PIPELINES_PIPELINEREGISTRATION_H_
#define ZKIR_PIPELINES_PIPELINEREGISTRATION_H_

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::zkir::pipelines {

void oneShotBufferize(OpPassManager &manager);

template <bool allowOpenMP>
void polyToLLVMPipelineBuilder(OpPassManager &manager);
void ellipticCurveToLLVMPipelineBuilder(OpPassManager &manager);

extern template void polyToLLVMPipelineBuilder<false>(OpPassManager &manager);
extern template void polyToLLVMPipelineBuilder<true>(OpPassManager &manager);

}  // namespace mlir::zkir::pipelines

#endif  // ZKIR_PIPELINES_PIPELINEREGISTRATION_H_
