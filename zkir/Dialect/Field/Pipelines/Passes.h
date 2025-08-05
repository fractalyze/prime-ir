#ifndef ZKIR_DIALECT_FIELD_PIPELINES_PASSES_H_
#define ZKIR_DIALECT_FIELD_PIPELINES_PASSES_H_

#include <string>

#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir::zkir::field {

struct FieldToLLVMOptions : public PassPipelineOptions<FieldToLLVMOptions> {
  PassOptions::Option<bool> enableOpenMP{
      *this, "enable-openmp",
      llvm::cl::desc("Lowers parallel loops to OpenMP dialect"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> bufferizeFunctionBoundaries{
      *this, "bufferize-function-boundaries",
      llvm::cl::desc("Bufferize function boundaries"), llvm::cl::init(false)};

  PassOptions::Option<bool> bufferResultsToOutParams{
      *this, "buffer-results-to-out-params",
      llvm::cl::desc("Buffer results to out params"), llvm::cl::init(true)};

  PassOptions::Option<bool> hoistStaticAllocs{
      *this, "hoist-static-allocs", llvm::cl::desc("Hoist static allocs"),
      llvm::cl::init(true)};

  // Projects out the options for `OneShotBufferizePass`.
  bufferization::OneShotBufferizePassOptions bufferizationOptions() const {
    bufferization::OneShotBufferizePassOptions opts{};
    opts.bufferizeFunctionBoundaries = bufferizeFunctionBoundaries;
    opts.functionBoundaryTypeConversion =
        bufferization::LayoutMapOption::IdentityLayoutMap;
    return opts;
  }

  // Projects out the options for `BufferResultsToOutParamsPass`.
  bufferization::BufferResultsToOutParamsPassOptions
  bufferResultsToOutParamsOptions() const {
    bufferization::BufferResultsToOutParamsPassOptions opts{};
    opts.hoistStaticAllocs = hoistStaticAllocs;
    return opts;
  }
};

// Adds the "field-to-llvm" pipeline to the `OpPassManager`.  This
// is the standard pipeline for taking field-based IR and lowering it
// to LLVM IR.
void buildFieldToLLVM(OpPassManager &pm, const FieldToLLVMOptions &options);

struct FieldToGPUOptions : public PassPipelineOptions<FieldToGPUOptions> {
  PassOptions::Option<bool> parallelizeAffine{
      *this, "parallelize-affine", llvm::cl::desc("Parallelize affine loops"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> bufferizeFunctionBoundaries{
      *this, "bufferize-function-boundaries",
      llvm::cl::desc("Bufferize function boundaries"), llvm::cl::init(false)};

  PassOptions::Option<bool> bufferResultsToOutParams{
      *this, "buffer-results-to-out-params",
      llvm::cl::desc("Buffer results to out params"), llvm::cl::init(false)};

  PassOptions::Option<bool> hoistStaticAllocs{
      *this, "hoist-static-allocs", llvm::cl::desc("Hoist static allocs"),
      llvm::cl::init(true)};

  PassOptions::Option<std::string> targetFormat{*this, "target-format",
                                                llvm::cl::desc("Target format"),
                                                llvm::cl::init("fatbin")};
  PassOptions::Option<std::string> targetChip{*this, "target-chip",
                                              llvm::cl::desc("Target chip"),
                                              llvm::cl::init("sm_80")};
  PassOptions::Option<std::string> targetFeatures{
      *this, "target-features", llvm::cl::desc("Target features"),
      llvm::cl::init("+ptx80")};
  PassOptions::Option<unsigned> targetOptLevel{
      *this, "target-opt-level", llvm::cl::desc("Optimization level"),
      llvm::cl::init(3)};

  PassOptions::Option<unsigned> nvvmIndexBitwidth{
      *this, "nvvm-index-bitwidth", llvm::cl::desc("NVVM index bitwidth"),
      llvm::cl::init(64)};
  PassOptions::Option<bool> nvvmUseBarePtrCallConv{
      *this, "nvvm-use-bare-ptr-call-conv",
      llvm::cl::desc("NVVM use bare ptr call conv"), llvm::cl::init(false)};

  // Projects out the options for `OneShotBufferizePass`.
  bufferization::OneShotBufferizePassOptions bufferizationOptions() const {
    bufferization::OneShotBufferizePassOptions opts{};
    opts.bufferizeFunctionBoundaries = bufferizeFunctionBoundaries;
    opts.functionBoundaryTypeConversion =
        bufferization::LayoutMapOption::IdentityLayoutMap;
    return opts;
  }

  // Projects out the options for `BufferResultsToOutParamsPass`.
  bufferization::BufferResultsToOutParamsPassOptions
  bufferResultsToOutParamsOptions() const {
    bufferization::BufferResultsToOutParamsPassOptions opts{};
    opts.hoistStaticAllocs = hoistStaticAllocs;
    return opts;
  }

  ConvertGpuOpsToNVVMOpsOptions convertGpuOpsToNVVMOpsOptions() const {
    ConvertGpuOpsToNVVMOpsOptions opts{};
    opts.indexBitwidth = nvvmIndexBitwidth;
    opts.useBarePtrCallConv = nvvmUseBarePtrCallConv;
    return opts;
  }

  GpuNVVMAttachTargetOptions targetOptions() const {
    GpuNVVMAttachTargetOptions opts{};
    opts.chip = targetChip;
    opts.features = targetFeatures;
    opts.optLevel = targetOptLevel;
    return opts;
  }

  GpuModuleToBinaryPassOptions gpuModuleToBinaryPassOptions() const {
    GpuModuleToBinaryPassOptions opts{};
    opts.compilationTarget = targetFormat;
    return opts;
  }
};

// Adds the "field-to-gpu" pipeline to the `OpPassManager`.  This
// is the standard pipeline for taking field-based IR and lowering it
// to GPU+LLVM IR.
void buildFieldToGPU(OpPassManager &pm, const FieldToGPUOptions &options);

void registerFieldPipelines();

}  // namespace mlir::zkir::field

#endif  // ZKIR_DIALECT_FIELD_PIPELINES_PASSES_H_
