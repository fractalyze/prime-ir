/* Copyright 2026 The ZKX Authors.

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
#ifndef ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_LARGE_INT_RETURN_TO_OUT_PARAM_H_
#define ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_LARGE_INT_RETURN_TO_OUT_PARAM_H_

#include "llvm/IR/PassManager.h"

namespace zkx::gpu {

// Rewrites internal functions returning large integer types (> 128 bits) to use
// an output pointer parameter instead. NVPTX PTX does not support .b256 in
// function ABIs, so non-inlined helpers returning e.g. i256 cause ptxas errors.
//
// Transforms:
//   define internal i256 @helper(ptr %a, i32 %idx) {
//     ...
//     ret i256 %result
//   }
//   %val = call i256 @helper(ptr %a, i32 42)
//
// Into:
//   define internal void @helper.out(ptr %a, i32 %idx, ptr %out) {
//     ...
//     store i256 %result, ptr %out
//     ret void
//   }
//   %tmp = alloca i256
//   call void @helper.out(ptr %a, i32 42, ptr %tmp)
//   %val = load i256, ptr %tmp
class LargeIntReturnToOutParamPass
    : public llvm::PassInfoMixin<LargeIntReturnToOutParamPass> {
 public:
  llvm::PreservedAnalyses run(llvm::Module& module,
                              llvm::ModuleAnalysisManager& AM);
};

}  // namespace zkx::gpu

#endif  // ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_LARGE_INT_RETURN_TO_OUT_PARAM_H_
