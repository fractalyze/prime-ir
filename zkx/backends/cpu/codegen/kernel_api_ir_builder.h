/* Copyright 2024 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

#ifndef ZKX_BACKENDS_CPU_CODEGEN_KERNEL_API_IR_BUILDER_H_
#define ZKX_BACKENDS_CPU_CODEGEN_KERNEL_API_IR_BUILDER_H_

#include <stdint.h>

#include <memory>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"

#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/runtime/buffer_use.h"
#include "zkx/service/buffer_assignment.h"
#include "zkx/service/hlo_module_config.h"
#include "zkx/shape.h"

namespace zkx::cpu {

class KernelApiIrBuilder {
 public:
  struct Options {
    bool enable_invariant_load_metadata;
    int32_t prefer_vector_width;

    static Options FromHloModuleConfig(const HloModuleConfig& config);
  };

  // Thread dimensions of the kernel invocation.
  struct ThreadDims {
    llvm::Value* x;
    llvm::Value* y;
    llvm::Value* z;
  };

  // Thread coordinates of the kernel invocation.
  struct ThreadId {
    llvm::Value* x;
    llvm::Value* y;
    llvm::Value* z;
  };

  // Kernel parameter (argument or result buffer) passed to a kernel function.
  // We rely on buffer allocation slice information to infer buffer aliasing
  // scopes for LLVM codegen.
  struct KernelParameter {
    Shape shape;
    BufferAllocation::Slice slice;
  };

  // A kernel function prototype with all the LLVM values that might be needed
  // to emit the actual kernel body.
  struct KernelPrototype {
    llvm::Function* function;
    llvm::BasicBlock* return_block;

    // LLVM values identifying kernel invocation thread coordinates.
    ThreadDims thread_dims;
    ThreadId thread_id;

    // LLVM values corresponding to the kernel arguments and results arrays. All
    // tuples are flattened as we do not have any tuples at run time and only
    // read and write data from/to leaf arrays.
    std::vector<llvm::Value*> arguments;
    std::vector<llvm::Value*> results;

    // Set containing all invariant (read-only) buffers indices. A buffer is
    // read-only if it is not aliased with any result.
    absl::flat_hash_set<int64_t> invariant_arguments;

    // the set of buffer uses for this kernel, can be empty if buffer
    // was not provided.
    absl::InlinedVector<BufferUse, 8> buffer_uses;
  };

  KernelApiIrBuilder(llvm::LLVMContext& context, Options options);

  // Emits a kernel prototype for the given HLO instruction.
  // buffer_assignment may be null, in which case we will not compute alias
  // metadata.
  absl::StatusOr<KernelPrototype> EmitKernelPrototype(
      llvm::Module& module, const HloInstruction* instr,
      const BufferAssignment* buffer_assignment, std::string_view suffix = "");

  absl::StatusOr<KernelPrototype> EmitKernelPrototype(
      llvm::Module& module, std::string_view name,
      absl::Span<const KernelParameter> arguments,
      absl::Span<const KernelParameter> results);

  // Create a module with the given name, the name is given a prefix that is
  // specific to ZKX and relied on further down the pipeline.
  static std::unique_ptr<llvm::Module> CreateModule(std::string_view name,
                                                    llvm::LLVMContext& context);

 private:
  ThreadDims EmitKernelThreadDims(llvm::IRBuilderBase& builder,
                                  llvm::Value* call_frame);
  ThreadId EmitKernelThread(llvm::IRBuilderBase& builder,
                            llvm::Value* call_frame);
  llvm::Value* EmitKernelArgument(llvm::IRBuilderBase& builder,
                                  llvm::Value* call_frame, int64_t index,
                                  const Shape& shape);
  llvm::Function* EmitKernelFunction(llvm::Module& module,
                                     std::string_view name);

 private:
  llvm::LLVMContext& context_;

  Options options_;

  llvm::StructType* thread_dim_ty_;  // not owned
  llvm::StructType* thread_ty_;      // not owned
  llvm::StructType* arg_ty_;         // not owned
  llvm::StructType* call_frame_ty_;  // not owned

  absl::flat_hash_map<int64_t, llvm::StructType*> memref_tys_;
  llvm::FunctionType* kernel_function_ty_;  // not owned
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_CODEGEN_KERNEL_API_IR_BUILDER_H_
