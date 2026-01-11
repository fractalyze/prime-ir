/* Copyright 2026 The PrimeIR Authors.

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

#ifndef PRIME_IR_UTILS_BUILDERCONTEXT_H_
#define PRIME_IR_UTILS_BUILDERCONTEXT_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Support/LLVM.h"

namespace mlir::prime_ir {

class BuilderContext {
public:
  static BuilderContext &GetInstance() {
    thread_local static BuilderContext &instance = *(new BuilderContext());
    return instance;
  }

  void Push(ImplicitLocOpBuilder *b) { builders.push_back(b); }
  void Pop() { builders.pop_back(); }
  ImplicitLocOpBuilder *Top() { return builders.back(); }

private:
  BuilderContext() = default;

  SmallVector<ImplicitLocOpBuilder *> builders;
};

class ScopedBuilderContext {
public:
  explicit ScopedBuilderContext(ImplicitLocOpBuilder *b) {
    BuilderContext::GetInstance().Push(b);
  }
  ~ScopedBuilderContext() { BuilderContext::GetInstance().Pop(); }
};

} // namespace mlir::prime_ir

#endif // PRIME_IR_UTILS_BUILDERCONTEXT_H_
