/* Copyright 2023 The OpenXLA Authors.
Copyright 2026 The ZKX Authors.

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

#ifndef ZKX_FFI_FFI_API_H_
#define ZKX_FFI_FFI_API_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "xla/tsl/concurrency/chain.h"
#include "zkx/executable_run_options.h"
#include "zkx/ffi/api/api.h"
#include "zkx/ffi/api/c_api.h"
#include "zkx/ffi/api/c_api_internal.h"  // IWYU pragma: keep
#include "zkx/ffi/call_frame.h"
#include "zkx/ffi/execution_context.h"
#include "zkx/ffi/execution_state.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/stream_executor/device_memory_allocator.h"
#include "zkx/stream_executor/stream.h"

namespace zkx::ffi {

// This is an implementation of ZKX FFI API defined in `api/c_api.h` header. It
// should be linked statically into the "main" ZKX binary, and third party FFI
// handlers can be linked and registered dynamically.
//
// FFI handlers registered statically (and built from the same ZKX commit with
// the same toolchain) can also use `api/c_api_internal.h` to get access to
// various internal data structures.

//===----------------------------------------------------------------------===//
// Calling ZKX FFI handlers
//===----------------------------------------------------------------------===//

// Options for calling ZKX FFI handlers. Backend specific options must be
// constructed from `zkx::ExecuteRunOptions`, to give FFI handlers access to
// ZKX runtime internals.
struct CallOptions {
  struct CpuOptions {
    const Eigen::ThreadPoolDevice* intra_op_thread_pool = nullptr;
  };

  struct GpuOptions {
    se::Stream* stream = nullptr;
    se::DeviceMemoryAllocator* allocator = nullptr;
  };

  using BackendOptions = std::variant<std::monostate, CpuOptions, GpuOptions>;

  zkx::RunId run_id = zkx::RunId{-1};
  int32_t device_ordinal = -1;
  BackendOptions backend_options;

  const HloComputation* called_computation = nullptr;
  const ExecutionContext* execution_context = nullptr;
  ExecutionState* execution_state = nullptr;
};

// Takes ownership of the ZKX FFI error and returns underlying status. Frees
// `error` if it's not nullptr. If `error` is nullptr, returns OkStatus.
absl::Status TakeStatus(ZKX_FFI_Error* error);

// Takes ownership of the ZKX FFI future and returns underlying AsyncValue.
// Frees `future` if it's not nullptr. If `future` is nullptr, returns available
// async value.
tsl::AsyncValueRef<tsl::Chain> TakeFuture(ZKX_FFI_Future* future);

// Calls an ZKX FFI handler with the given call frame and options. This is a
// synchronous call and it might block the caller thread if the handler is
// asynchronous. It is unsafe to call if from a thread pool that runs tasks
// scheduled by the handler itself.
absl::Status Call(Ffi& handler, CallFrame& call_frame,
                  const CallOptions& options = {},
                  ExecutionStage stage = ExecutionStage::kExecute);

absl::Status Call(
    ZKX_FFI_Handler* handler, CallFrame& call_frame,
    const CallOptions& options = {},
    ZKX_FFI_ExecutionStage stage = ZKX_FFI_ExecutionStage_EXECUTE);

// Calls an ZKX FFI handler with the given call frame and options. This is an
// asynchronous call and it will not block the caller thread. Returned async
// value will become available when the handler completes execution.
tsl::AsyncValueRef<tsl::Chain> CallAsync(
    Ffi& handler, CallFrame& call_frame, const CallOptions& options = {},
    ExecutionStage stage = ExecutionStage::kExecute);

tsl::AsyncValueRef<tsl::Chain> CallAsync(
    ZKX_FFI_Handler* handler, CallFrame& call_frame,
    const CallOptions& options = {},
    ZKX_FFI_ExecutionStage stage = ZKX_FFI_ExecutionStage_EXECUTE);

// Gets metadata from the handler by calling it with a special call frame.
absl::StatusOr<ZKX_FFI_Metadata> GetMetadata(Ffi& handler);
absl::StatusOr<ZKX_FFI_Metadata> GetMetadata(ZKX_FFI_Handler* handler);

namespace internal {
// This is an internal workaround to override FFI execution context for FFI
// calls executed in the current thread with `context` in tests that use legacy
// zkx::Client, zkx::Service and zkx::Backend APIs because it's not worth it to
// add proper execution context support throughout all abstraction layers
// (legacy client APIs should be eventually deleted instead). This workaround
// should not be used outside of tests.
class ScopedExecutionContext {
 public:
  explicit ScopedExecutionContext(const ExecutionContext* context);
  ~ScopedExecutionContext();

  ScopedExecutionContext(ScopedExecutionContext&&) = delete;
  ScopedExecutionContext& operator=(ScopedExecutionContext&&) = delete;

  // Returns an execution context that should be used for FFI calls based on the
  // call options and the current thread's execution context.
  static const ExecutionContext* GetCallExecutionContext(
      const CallOptions& options);

 private:
  const ExecutionContext* recover_;
};
}  // namespace internal

//===----------------------------------------------------------------------===//
// ZKX FFI registry
//===----------------------------------------------------------------------===//

struct HandlerRegistration {
  ZKX_FFI_Handler_Bundle bundle = {};
  ZKX_FFI_Handler_Traits traits = {};
};

bool IsCommandBufferCompatible(ZKX_FFI_Handler_Traits traits);

// Returns registered FFI handler for a given name and platform, or an error if
// it's not found in the static registry.
absl::StatusOr<HandlerRegistration> FindHandler(std::string_view name,
                                                std::string_view platform);

// Returns all registered calls in the static registry for a given platform.
absl::StatusOr<absl::flat_hash_map<std::string, HandlerRegistration>>
StaticRegisteredHandlers(std::string_view platform);

//===----------------------------------------------------------------------===//
// ZKX FFI Api Implementation
//===----------------------------------------------------------------------===//

const ZKX_FFI_Api* GetZkxFfiApi();

}  // namespace zkx::ffi

#endif  // ZKX_FFI_FFI_API_H_
