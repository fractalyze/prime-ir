#ifndef ZKIR_PIPELINES_PIPELINEREGISTRATION_H_
#define ZKIR_PIPELINES_PIPELINEREGISTRATION_H_

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::zkir::pipelines {

void oneShotBufferize(OpPassManager &manager);

void polyToLLVMPipelineBuilder(OpPassManager &manager);

}  // namespace mlir::zkir::pipelines

#endif  // ZKIR_PIPELINES_PIPELINEREGISTRATION_H_
