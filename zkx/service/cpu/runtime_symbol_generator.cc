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

#include "zkx/service/cpu/runtime_symbol_generator.h"

#include <dlfcn.h>  // for dlopen, dlsym on Linux/macOS
#include <stdio.h>
#include <string.h>

#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include "zkx/service/cpu/cpu_runtime.h"
#include "zkx/service/cpu/runtime_custom_call_status.h"
#include "zkx/service/cpu/runtime_fork_join.h"
#include "zkx/service/custom_call_target_registry.h"

namespace zkx::cpu {

[[maybe_unused]] static void* GetSymbolFromCurrentProcess(
    const char* symbol_name) {
  void* symbol_ptr = dlsym(RTLD_DEFAULT, symbol_name);
  CHECK(symbol_ptr) << symbol_name << " not found in process symbols.";
  return symbol_ptr;
}

llvm::Error RuntimeSymbolGenerator::tryToGenerate(
    llvm::orc::LookupState&, llvm::orc::LookupKind kind,
    llvm::orc::JITDylib& jit_dylib, llvm::orc::JITDylibLookupFlags,
    const llvm::orc::SymbolLookupSet& names) {
  llvm::orc::SymbolMap new_defs;

  for (const auto& kv : names) {
    const auto& name = kv.first;
    if (auto symbol = ResolveRuntimeSymbol(*name)) {
      new_defs[name] = *symbol;
    }
  }

  cantFail(jit_dylib.define(llvm::orc::absoluteSymbols(std::move(new_defs))));
  return llvm::Error::success();
}

std::optional<llvm::orc::ExecutorSymbolDef>
RuntimeSymbolGenerator::ResolveRuntimeSymbol(llvm::StringRef name) {
  void* fn_addr = nullptr;
  if (name.size() > 1 && name.front() == data_layout_.getGlobalPrefix()) {
    // On Mac OS X, 'name' may have a leading underscore prefix, even though the
    // registered name may not.
    std::string stripped_name(name.begin() + 1, name.end());
    fn_addr = CustomCallTargetRegistry::Global()->Lookup(stripped_name, "Host");
    // Fallback: resolve from the current process (compiler-rt builtins like
    // __umodti3, __udivti3, etc.).
    if (!fn_addr) {
      fn_addr = dlsym(RTLD_DEFAULT, stripped_name.c_str());
    }
  } else {
    fn_addr = CustomCallTargetRegistry::Global()->Lookup(name.str(), "Host");
    if (!fn_addr) {
      fn_addr = dlsym(RTLD_DEFAULT, name.str().c_str());
    }
  }

  if (!fn_addr) return std::nullopt;

  return llvm::orc::ExecutorSymbolDef{
      llvm::orc::ExecutorAddr(reinterpret_cast<uint64_t>(fn_addr)),
      llvm::JITSymbolFlags::None};
}

//===----------------------------------------------------------------------===//
// Register ZKX:CPU runtime symbols with the CustomCallTargetRegistry.
//===----------------------------------------------------------------------===//

#if defined(PLATFORM_WINDOWS)
// This function is used by compiler-generated code on windows, but it's not
// declared anywhere. The signature does not matter, we just need the address.
extern "C" void __chkstk(size_t);
#endif

#define REGISTER_CPU_RUNTIME_SYMBOL(base_name)                              \
  do {                                                                      \
    auto* function_address =                                                \
        reinterpret_cast<void*>(__zkx_cpu_runtime_##base_name);             \
    registry->Register(zkx::cpu::runtime::k##base_name##SymbolName,         \
                       function_address, "Host");                           \
    CHECK_EQ(std::string_view(zkx::cpu::runtime::k##base_name##SymbolName), \
             "__zkx_cpu_runtime_" #base_name);                              \
  } while (false)

namespace {

bool RegisterKnownJITSymbols() {
  zkx::CustomCallTargetRegistry* registry =
      zkx::CustomCallTargetRegistry::Global();
  registry->Register("printf", reinterpret_cast<void*>(&printf), "Host");
  registry->Register("puts", reinterpret_cast<void*>(&puts), "Host");

  REGISTER_CPU_RUNTIME_SYMBOL(AcquireInfeedBufferForDequeue);
  REGISTER_CPU_RUNTIME_SYMBOL(AcquireOutfeedBufferForPopulation);
  REGISTER_CPU_RUNTIME_SYMBOL(AllReduce);
  REGISTER_CPU_RUNTIME_SYMBOL(CollectivePermute);
  REGISTER_CPU_RUNTIME_SYMBOL(AllToAll);
  REGISTER_CPU_RUNTIME_SYMBOL(AllGather);
  REGISTER_CPU_RUNTIME_SYMBOL(ReduceScatter);
  REGISTER_CPU_RUNTIME_SYMBOL(PartitionId);
  REGISTER_CPU_RUNTIME_SYMBOL(ReplicaId);
  REGISTER_CPU_RUNTIME_SYMBOL(ParallelForkJoin);
  REGISTER_CPU_RUNTIME_SYMBOL(PrintfToStderr);
  REGISTER_CPU_RUNTIME_SYMBOL(ReleaseInfeedBufferAfterDequeue);
  REGISTER_CPU_RUNTIME_SYMBOL(ReleaseOutfeedBufferAfterPopulation);
  REGISTER_CPU_RUNTIME_SYMBOL(StatusIsSuccess);
  REGISTER_CPU_RUNTIME_SYMBOL(TracingStart);
  REGISTER_CPU_RUNTIME_SYMBOL(TracingEnd);

  registry->Register("memcpy", reinterpret_cast<void*>(memcpy), "Host");
  registry->Register("memmove", reinterpret_cast<void*>(memmove), "Host");
  registry->Register("memset", reinterpret_cast<void*>(memset), "Host");

  // Used by MLIR lowering.
  registry->Register("abort", reinterpret_cast<void*>(abort), "Host");
  registry->Register("malloc", reinterpret_cast<void*>(malloc), "Host");
  registry->Register("calloc", reinterpret_cast<void*>(calloc), "Host");
  registry->Register("free", reinterpret_cast<void*>(free), "Host");
#ifndef _WIN32
  // TODO(b/246980307): fails to link on windows because it's marked dllimport.
  registry->Register("memrefCopy", reinterpret_cast<void*>(memrefCopy), "Host");
#endif

#ifdef __APPLE__
  registry->Register("__bzero", reinterpret_cast<void*>(bzero), "Host");
  registry->Register("bzero", reinterpret_cast<void*>(bzero), "Host");
  registry->Register("memset_pattern16",
                     reinterpret_cast<void*>(memset_pattern16), "Host");
#endif

#ifdef MEMORY_SANITIZER
  registry->Register("__msan_unpoison",
                     reinterpret_cast<void*>(__msan_unpoison), "Host");
#endif

#if defined(PLATFORM_WINDOWS)
  registry->Register("__chkstk", reinterpret_cast<void*>(__chkstk), "Host");
#endif

#ifdef ZKX_HAS_OPENMP
  const char* openmp_symbols[] = {
      "__kmpc_barrier",
      "__kmpc_for_static_fini",
      "__kmpc_for_static_init_8u",
      "__kmpc_fork_call",
      "__kmpc_global_thread_num",
  };
  for (const char* symbol : openmp_symbols) {
    registry->Register(
        symbol, reinterpret_cast<void*>(GetSymbolFromCurrentProcess(symbol)),
        "Host");
  }
#endif

  return true;
}

}  // namespace

#undef REGISTER_CPU_RUNTIME_SYMBOL

namespace {

bool unused = RegisterKnownJITSymbols();

}  // namespace

}  // namespace zkx::cpu
