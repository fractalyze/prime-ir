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

#ifndef PRIME_IR_DIALECT_FIELD_PIPELINES_PASSES_H_
#define PRIME_IR_DIALECT_FIELD_PIPELINES_PASSES_H_

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir::prime_ir::field {

struct FieldToLLVMOptions : public PassPipelineOptions<FieldToLLVMOptions> {
  PassOptions::Option<bool> enableOpenMP{
      *this, "enable-openmp",
      llvm::cl::desc("Lowers parallel loops to OpenMP dialect"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> specializeAVX{
      *this, "specialize-avx",
      llvm::cl::desc("Specialize arithmetic operations to AVX instructions"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> specializeGFNI{
      *this, "specialize-gfni",
      llvm::cl::desc("Specialize binary field operations to GFNI instructions"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> specializePCLMULQDQ{
      *this, "specialize-pclmulqdq",
      llvm::cl::desc(
          "Specialize binary field operations to PCLMULQDQ instructions"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> specializePMULL{
      *this, "specialize-pmull",
      llvm::cl::desc(
          "Specialize binary field operations to ARM PMULL instructions"),
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

  PassOptions::Option<bool> vectorize{
      *this, "vectorize", llvm::cl::desc("Enable affine super-vectorization"),
      llvm::cl::init(false)};

  PassOptions::Option<unsigned> vectorSize{
      *this, "vector-size",
      llvm::cl::desc("Vector size for super-vectorization"),
      llvm::cl::init(16)};

  PassOptions::Option<bool> lazyReduction{
      *this, "lazy-reduction",
      llvm::cl::desc("Enable lazy reduction optimization via integer range "
                     "analysis in mod_arith to arith lowering"),
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

void registerFieldPipelines();

} // namespace mlir::prime_ir::field

#endif // PRIME_IR_DIALECT_FIELD_PIPELINES_PASSES_H_
