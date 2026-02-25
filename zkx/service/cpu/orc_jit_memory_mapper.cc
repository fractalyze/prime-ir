/* Copyright 2017 The OpenXLA Authors.
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

#include "zkx/service/cpu/orc_jit_memory_mapper.h"

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"

namespace zkx::cpu::orc_jit_memory_mapper {

namespace {

ABSL_CONST_INIT absl::Mutex g_mapper_instance_mutex(absl::kConstInit);
llvm::SectionMemoryManager::MemoryMapper* g_mapper_instance
    ABSL_GUARDED_BY(g_mapper_instance_mutex) = nullptr;

}  // namespace

llvm::SectionMemoryManager::MemoryMapper* GetInstance() {
  absl::MutexLock lock(&g_mapper_instance_mutex);
  return g_mapper_instance;
}

Registrar::Registrar(
    std::unique_ptr<llvm::SectionMemoryManager::MemoryMapper> mapper) {
  absl::MutexLock lock(&g_mapper_instance_mutex);
  g_mapper_instance = mapper.release();
}

}  // namespace zkx::cpu::orc_jit_memory_mapper
