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
#include "zkx/backends/gpu/codegen/emitters/large_int_return_to_out_param.h"

#include "absl/log/check.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"

namespace zkx::gpu {

llvm::PreservedAnalyses LargeIntReturnToOutParamPass::run(
    llvm::Module& module, llvm::ModuleAnalysisManager& AM) {
  llvm::SmallVector<llvm::Function*> to_rewrite;
  for (auto& fn : module) {
    if (fn.isDeclaration() || !fn.hasInternalLinkage()) continue;
    auto* ret_ty = fn.getReturnType();
    if (!ret_ty->isIntegerTy() || ret_ty->getIntegerBitWidth() <= 128) continue;
    to_rewrite.push_back(&fn);
  }

  if (to_rewrite.empty()) {
    return llvm::PreservedAnalyses::all();
  }

  for (auto* old_fn : to_rewrite) {
    auto* ret_ty = old_fn->getReturnType();
    auto* void_ty = llvm::Type::getVoidTy(module.getContext());
    auto* ptr_ty = llvm::PointerType::get(module.getContext(), 0);

    // Build new function type: same params + ptr out param, returns void.
    llvm::SmallVector<llvm::Type*> new_param_types;
    for (auto& arg : old_fn->args()) {
      new_param_types.push_back(arg.getType());
    }
    new_param_types.push_back(ptr_ty);
    auto* new_fn_ty = llvm::FunctionType::get(void_ty, new_param_types, false);

    // Create new function.
    auto* new_fn = llvm::Function::Create(new_fn_ty, old_fn->getLinkage(),
                                          old_fn->getName() + ".out", &module);
    new_fn->setCallingConv(old_fn->getCallingConv());

    // Map old args to new args.
    llvm::ValueToValueMapTy vmap;
    auto new_arg_it = new_fn->arg_begin();
    for (auto& old_arg : old_fn->args()) {
      new_arg_it->setName(old_arg.getName());
      vmap[&old_arg] = &*new_arg_it;
      ++new_arg_it;
    }
    // The last new arg is the output pointer.
    auto* out_ptr_arg = &*new_arg_it;
    out_ptr_arg->setName("out");

    // Clone function body.
    llvm::SmallVector<llvm::ReturnInst*, 4> returns;
    llvm::CloneFunctionInto(new_fn, old_fn, vmap,
                            llvm::CloneFunctionChangeType::LocalChangesOnly,
                            returns);

    // Replace ret i256 %val → store + ret void.
    for (auto* ret : returns) {
      llvm::IRBuilder<> builder(ret);
      builder.CreateStore(ret->getReturnValue(), out_ptr_arg);
      builder.CreateRetVoid();
      ret->eraseFromParent();
    }

    // Update all call sites.
    llvm::SmallVector<llvm::CallInst*> calls;
    for (auto* user : old_fn->users()) {
      auto* call = llvm::dyn_cast<llvm::CallInst>(user);
      CHECK(call) << "Unsupported user of function with large integer return";
      calls.push_back(call);
    }
    for (auto* call : calls) {
      llvm::IRBuilder<> builder(call);
      auto* alloca = builder.CreateAlloca(ret_ty);
      llvm::SmallVector<llvm::Value*> args(call->arg_begin(), call->arg_end());
      args.push_back(alloca);
      auto* new_call = builder.CreateCall(new_fn, args);
      new_call->setCallingConv(new_fn->getCallingConv());
      auto* load = builder.CreateLoad(ret_ty, alloca);
      call->replaceAllUsesWith(load);
      call->eraseFromParent();
    }

    old_fn->eraseFromParent();
  }

  return llvm::PreservedAnalyses::none();
}

}  // namespace zkx::gpu
