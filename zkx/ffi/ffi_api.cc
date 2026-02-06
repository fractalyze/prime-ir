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

#include "zkx/ffi/ffi_api.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"

#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/ffi/type_id_registry.h"
#include "zkx/service/platform_util.h"
#include "zkx/stream_executor/device_memory.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

//===----------------------------------------------------------------------===//
// ZKX FFI C structs definition
//===----------------------------------------------------------------------===//

struct ZKX_FFI_Error {
  absl::Status status;
};

struct ZKX_FFI_Future {
  tsl::AsyncValueRef<tsl::Chain> async_value;
};

struct ZKX_FFI_ExecutionContext {
  struct CpuContext {
    const Eigen::ThreadPoolDevice* intra_op_thread_pool = nullptr;
  };

  struct GpuContext {
    stream_executor::Stream* stream = nullptr;
    stream_executor::DeviceMemoryAllocator* allocator = nullptr;
  };

  using BackendContext = std::variant<std::monostate, CpuContext, GpuContext>;

  zkx::RunId run_id = {};
  int32_t device_ordinal = -1;
  BackendContext backend_context = {};

  const zkx::HloComputation* called_computation = nullptr;
  const zkx::ffi::ExecutionContext* execution_context = nullptr;
  zkx::ffi::ExecutionState* execution_state = nullptr;
};

//===----------------------------------------------------------------------===//

namespace zkx::ffi {

bool IsCommandBufferCompatible(ZKX_FFI_Handler_Traits traits) {
  return traits & ZKX_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE;
}

namespace {

ZKX_FFI_ExecutionContext CreateExecutionContext(const CallOptions& options) {
  using BackendContext = ZKX_FFI_ExecutionContext::BackendContext;

  // Converts CallOptions to corresponding backend context.
  struct BackendVisitor {
    BackendContext operator()(const std::monostate&) const {
      return std::monostate{};
    }

    BackendContext operator()(const CallOptions::CpuOptions& options) const {
      return ZKX_FFI_ExecutionContext::CpuContext{options.intra_op_thread_pool};
    }

    BackendContext operator()(const CallOptions::GpuOptions& options) const {
      return ZKX_FFI_ExecutionContext::GpuContext{options.stream,
                                                  options.allocator};
    }
  };

  return ZKX_FFI_ExecutionContext{
      options.run_id,
      options.device_ordinal,
      std::visit(BackendVisitor{}, options.backend_options),
      options.called_computation,
      internal::ScopedExecutionContext::GetCallExecutionContext(options),
      options.execution_state,
  };
}

}  // namespace

//===----------------------------------------------------------------------===//
// Calling ZKX FFI handlers
//===----------------------------------------------------------------------===//

absl::Status TakeStatus(ZKX_FFI_Error* error) {
  if (ABSL_PREDICT_TRUE(error == nullptr)) return absl::OkStatus();
  absl::Status status = std::move(error->status);
  delete error;
  return status;
}

tsl::AsyncValueRef<tsl::Chain> TakeFuture(ZKX_FFI_Future* future) {
  // Non-reference-counted async value ref for synchronous FFI handlers.
  static tsl::AsyncValueOwningRef<tsl::Chain>* chain = [] {
    auto* storage = new tsl::internal::AsyncValueStorage<tsl::Chain>();
    return new tsl::AsyncValueOwningRef<tsl::Chain>(
        tsl::MakeAvailableAsyncValueRef<tsl::Chain>(*storage));
  }();

  if (ABSL_PREDICT_TRUE(future == nullptr)) return chain->AsRef();

  // If the future is already completed, immediately return the underlying async
  // value and delete the ZKX_FFI_Future.
  if (ABSL_PREDICT_TRUE(future->async_value.IsAvailable())) {
    tsl::AsyncValueRef<tsl::Chain> async_value = std::move(future->async_value);
    delete future;
    return async_value;
  }

  // If the future is not completed, return a copy of the underlying async value
  // and keep ZKX_FFI_Future alive until it is completed.
  tsl::AsyncValueRef<tsl::Chain> async_value = future->async_value;
  async_value.AndThen([future] { delete future; });
  return async_value;
}

namespace {

template <typename Handler>
absl::StatusOr<ZKX_FFI_Future*> Call(Handler& handler, CallFrame& call_frame,
                                     const CallOptions& options,
                                     ExecutionStage stage) {
  ZKX_FFI_ExecutionContext ctx = CreateExecutionContext(options);
  ZKX_FFI_CallFrame ffi_call_frame = call_frame.Build(
      GetZkxFfiApi(), &ctx, static_cast<ZKX_FFI_ExecutionStage>(stage));

  ZKX_FFI_Error* error = nullptr;

  // FFI handlers might be defined in external libraries and use exceptions, so
  // take extra care to catch them and convert to a status.
  try {
    if constexpr (std::is_same_v<Handler, Ffi>) {
      error = handler.Call(&ffi_call_frame);
    } else if constexpr (std::is_same_v<Handler, ZKX_FFI_Handler*>) {
      error = (*handler)(&ffi_call_frame);
    } else {
      static_assert(sizeof(Handler) == 0, "Unsupported handler type");
    }
  } catch (std::exception& e) {
    return absl::UnknownError(absl::StrCat("ZKX FFI call failed: ", e.what()));
  }

  // If FFI handler returned synchronous error, it must not launch any
  // asynchronous work that can also return an error.
  if (error != nullptr) {
    DCHECK_EQ(ffi_call_frame.future, nullptr)
        << "Error must not be used together with a future";
    return TakeStatus(error);
  }

  return ffi_call_frame.future;
}

absl::Status BlockUntilReady(ZKX_FFI_Future* future) {
  if (ABSL_PREDICT_TRUE(future == nullptr)) return absl::OkStatus();

  tsl::AsyncValueRef<tsl::Chain> av = TakeFuture(future);
  tsl::BlockUntilReady(av);
  return ABSL_PREDICT_FALSE(av.IsError()) ? av.GetError() : absl::OkStatus();
}

}  // namespace

absl::Status Call(Ffi& handler, CallFrame& call_frame,
                  const CallOptions& options, ExecutionStage stage) {
  TF_ASSIGN_OR_RETURN(ZKX_FFI_Future * future,
                      Call<Ffi>(handler, call_frame, options, stage));
  return BlockUntilReady(future);
}

absl::Status Call(ZKX_FFI_Handler* handler, CallFrame& call_frame,
                  const CallOptions& options, ZKX_FFI_ExecutionStage stage) {
  TF_ASSIGN_OR_RETURN(
      ZKX_FFI_Future * future,
      Call<ZKX_FFI_Handler*>(handler, call_frame, options,
                             static_cast<ExecutionStage>(stage)));
  return BlockUntilReady(future);
}

tsl::AsyncValueRef<tsl::Chain> CallAsync(Ffi& handler, CallFrame& call_frame,
                                         const CallOptions& options,
                                         ExecutionStage stage) {
  TF_ASSIGN_OR_RETURN(ZKX_FFI_Future * future,
                      Call<Ffi>(handler, call_frame, options, stage));
  return TakeFuture(future);
}

tsl::AsyncValueRef<tsl::Chain> CallAsync(ZKX_FFI_Handler* handler,
                                         CallFrame& call_frame,
                                         const CallOptions& options,
                                         ZKX_FFI_ExecutionStage stage) {
  TF_ASSIGN_OR_RETURN(
      ZKX_FFI_Future * future,
      Call<ZKX_FFI_Handler*>(handler, call_frame, options,
                             static_cast<ExecutionStage>(stage)));
  return TakeFuture(future);
}

namespace {

ZKX_FFI_Metadata BuildMetadata() {
  return ZKX_FFI_Metadata{ZKX_FFI_Metadata_STRUCT_SIZE,
                          ZKX_FFI_Api_Version{ZKX_FFI_Api_Version_STRUCT_SIZE}};
}

ZKX_FFI_Metadata_Extension BuildMetadataExtension(ZKX_FFI_Metadata* metadata) {
  return ZKX_FFI_Metadata_Extension{
      ZKX_FFI_Extension_Base{ZKX_FFI_Metadata_Extension_STRUCT_SIZE,
                             ZKX_FFI_Extension_Metadata},
      metadata};
}

ZKX_FFI_CallFrame BuildMetadataCallFrame(
    ZKX_FFI_Metadata_Extension* extension) {
  return ZKX_FFI_CallFrame{
      ZKX_FFI_CallFrame_STRUCT_SIZE,
      &extension->extension_base,
      /*api=*/nullptr,
      /*context=*/nullptr,
      /*stage=*/ZKX_FFI_ExecutionStage_EXECUTE,
      /*args=*/ZKX_FFI_Args{ZKX_FFI_Args_STRUCT_SIZE},
      /*rets=*/ZKX_FFI_Rets{ZKX_FFI_Rets_STRUCT_SIZE},
      /*attrs=*/ZKX_FFI_Attrs{ZKX_FFI_Attrs_STRUCT_SIZE},
  };
}

}  // namespace

absl::StatusOr<ZKX_FFI_Metadata> GetMetadata(Ffi& handler) {
  ZKX_FFI_Metadata metadata = BuildMetadata();
  ZKX_FFI_Metadata_Extension extension = BuildMetadataExtension(&metadata);
  ZKX_FFI_CallFrame call_frame = BuildMetadataCallFrame(&extension);
  ZKX_FFI_Error* error = nullptr;
  try {
    error = handler.Call(&call_frame);
  } catch (std::exception& e) {
    return absl::UnknownError(
        absl::StrCat("Fetching ZKX FFI metadata failed: ", e.what()));
  }
  if (error != nullptr) {
    return TakeStatus(error);
  }
  return metadata;
}

absl::StatusOr<ZKX_FFI_Metadata> GetMetadata(ZKX_FFI_Handler* handler) {
  ZKX_FFI_Metadata metadata = BuildMetadata();
  ZKX_FFI_Metadata_Extension extension = BuildMetadataExtension(&metadata);
  ZKX_FFI_CallFrame call_frame = BuildMetadataCallFrame(&extension);
  ZKX_FFI_Error* error = nullptr;
  try {
    error = (*handler)(&call_frame);
  } catch (std::exception& e) {
    return absl::UnknownError(
        absl::StrCat("Fetching ZKX FFI metadata failed: ", e.what()));
  }
  if (error != nullptr) {
    return TakeStatus(error);
  }
  return metadata;
}

namespace internal {
static thread_local const ExecutionContext* scoped_execution_context = nullptr;

ScopedExecutionContext::ScopedExecutionContext(const ExecutionContext* context)
    : recover_(scoped_execution_context) {
  scoped_execution_context = context;
}

ScopedExecutionContext::~ScopedExecutionContext() {
  scoped_execution_context = recover_;
}

const ExecutionContext* ScopedExecutionContext::GetCallExecutionContext(
    const CallOptions& options) {
  if (scoped_execution_context != nullptr) {
    return scoped_execution_context;
  }
  return options.execution_context;
}
}  // namespace internal

//===----------------------------------------------------------------------===//
// ZKX FFI registry
//===----------------------------------------------------------------------===//

using HandlerKey = std::pair<std::string, std::string>;
using HandlerRegistry = absl::flat_hash_map<HandlerKey, HandlerRegistration>;

namespace {

HandlerKey MakeHandlerKey(std::string_view name, std::string_view platform) {
  return std::make_pair(std::string(name), absl::AsciiStrToLower(platform));
}

HandlerRegistry& GetHandlerRegistry() {
  static auto* registry = new HandlerRegistry();
  return *registry;
}

std::vector<std::string> GetHandlerStages(
    const ZKX_FFI_Handler_Bundle& bundle) {
  std::vector<std::string> stages;
  if (bundle.instantiate != nullptr) stages.push_back("instantiate");
  if (bundle.prepare != nullptr) stages.push_back("prepare");
  if (bundle.initialize != nullptr) stages.push_back("initialize");
  if (bundle.execute != nullptr) stages.push_back("execute");
  return stages;
}

absl::Status RegisterHandler(std::string_view name, std::string_view platform,
                             ZKX_FFI_Handler_Bundle bundle,
                             ZKX_FFI_Handler_Traits traits) {
  TF_ASSIGN_OR_RETURN(std::string canonical_platform,
                      PlatformUtil::CanonicalPlatformName(platform));

  if (bundle.execute == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("FFI handler for ", name, " on a platform ", platform,
                     " must provide an execute implementation"));
  }

  // Check the API versions.
  TF_ASSIGN_OR_RETURN(auto metadata, GetMetadata(bundle.execute));
  const ZKX_FFI_Api_Version& api_version = metadata.api_version;
  if (api_version.major_version != ZKX_FFI_API_MAJOR ||
      api_version.minor_version != ZKX_FFI_API_MINOR) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "FFI handler registration for %s on platform %s (canonical %s) failed "
        "because the handler's API version (%d.%d) is incompatible with the "
        "framework's API version (%d.%d)",
        name, platform, canonical_platform, api_version.major_version,
        api_version.minor_version, ZKX_FFI_API_MAJOR, ZKX_FFI_API_MINOR));
  }

  // Incorporate handler traits.
  traits |= metadata.traits;

  VLOG(2) << absl::StreamFormat(
      "Register ZKX FFI handler for '%s'; platform=%s (canonical=%s), "
      "stages=[%s], command_buffer_compatible=%v",
      name, platform, canonical_platform,
      absl::StrJoin(GetHandlerStages(bundle), ", "),
      IsCommandBufferCompatible(traits));

  auto emplaced =
      GetHandlerRegistry().try_emplace(MakeHandlerKey(name, canonical_platform),
                                       HandlerRegistration{bundle, traits});
  if (!emplaced.second) {
    auto existing = emplaced.first->second;
    if (existing.traits != traits) {
      return absl::InvalidArgumentError(
          absl::StrCat("Duplicate FFI handler registration for ", name,
                       " on platform ", platform, " (canonical ",
                       canonical_platform, ") with different traits"));
    }
    if (existing.bundle.prepare != bundle.prepare ||
        existing.bundle.initialize != bundle.initialize ||
        existing.bundle.execute != bundle.execute) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Duplicate FFI handler registration for ", name, " on platform ",
          platform, " (canonical ", canonical_platform,
          ") with different bundle addresses"));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<HandlerRegistration> FindHandler(std::string_view name,
                                                std::string_view platform) {
  TF_ASSIGN_OR_RETURN(std::string canonical_platform,
                      PlatformUtil::CanonicalPlatformName(platform));

  auto it = GetHandlerRegistry().find(MakeHandlerKey(name, canonical_platform));
  if (it == GetHandlerRegistry().end()) {
    return absl::NotFoundError(
        absl::StrCat("No FFI handler registered for ", name, " on a platform ",
                     platform, " (canonical ", canonical_platform, ")"));
  }
  return it->second;
}

absl::StatusOr<absl::flat_hash_map<std::string, HandlerRegistration>>
StaticRegisteredHandlers(std::string_view platform) {
  TF_ASSIGN_OR_RETURN(std::string canonical_platform,
                      PlatformUtil::CanonicalPlatformName(platform));

  absl::flat_hash_map<std::string, HandlerRegistration> calls;
  for (const auto& [metadata, handler] : GetHandlerRegistry()) {
    if (canonical_platform == metadata.second) {
      calls[metadata.first] = handler;
    }
  }

  return calls;
}

//===----------------------------------------------------------------------===//
// ZKX FFI Api Implementation
//===----------------------------------------------------------------------===//

namespace {

std::string StructSizeErrorMsg(std::string_view struct_name, size_t expected,
                               size_t actual) {
  return absl::StrCat("Unexpected ", struct_name, " size: expected ", expected,
                      ", got ", actual, ". Check installed software versions. ",
                      "The framework ZKX FFI API version is ",
                      ZKX_FFI_API_MAJOR, ".", ZKX_FFI_API_MINOR, ".");
}

absl::Status ActualStructSizeIsGreaterOrEqual(std::string_view struct_name,
                                              size_t expected, size_t actual) {
  if (actual < expected) {
    return absl::InvalidArgumentError(
        StructSizeErrorMsg(struct_name, expected, actual));
  }
  if (actual > expected) {
    VLOG(2) << StructSizeErrorMsg(struct_name, expected, actual);
  }
  return absl::OkStatus();
}

absl::StatusCode ToStatusCode(ZKX_FFI_Error_Code errc) {
  switch (errc) {
    case ZKX_FFI_Error_Code_OK:
      return absl::StatusCode::kOk;
    case ZKX_FFI_Error_Code_CANCELLED:
      return absl::StatusCode::kCancelled;
    case ZKX_FFI_Error_Code_UNKNOWN:
      return absl::StatusCode::kUnknown;
    case ZKX_FFI_Error_Code_INVALID_ARGUMENT:
      return absl::StatusCode::kInvalidArgument;
    case ZKX_FFI_Error_Code_DEADLINE_EXCEEDED:
      return absl::StatusCode::kDeadlineExceeded;
    case ZKX_FFI_Error_Code_NOT_FOUND:
      return absl::StatusCode::kNotFound;
    case ZKX_FFI_Error_Code_ALREADY_EXISTS:
      return absl::StatusCode::kAlreadyExists;
    case ZKX_FFI_Error_Code_PERMISSION_DENIED:
      return absl::StatusCode::kPermissionDenied;
    case ZKX_FFI_Error_Code_RESOURCE_EXHAUSTED:
      return absl::StatusCode::kResourceExhausted;
    case ZKX_FFI_Error_Code_FAILED_PRECONDITION:
      return absl::StatusCode::kFailedPrecondition;
    case ZKX_FFI_Error_Code_ABORTED:
      return absl::StatusCode::kAborted;
    case ZKX_FFI_Error_Code_OUT_OF_RANGE:
      return absl::StatusCode::kOutOfRange;
    case ZKX_FFI_Error_Code_UNIMPLEMENTED:
      return absl::StatusCode::kUnimplemented;
    case ZKX_FFI_Error_Code_INTERNAL:
      return absl::StatusCode::kInternal;
    case ZKX_FFI_Error_Code_UNAVAILABLE:
      return absl::StatusCode::kUnavailable;
    case ZKX_FFI_Error_Code_DATA_LOSS:
      return absl::StatusCode::kDataLoss;
    case ZKX_FFI_Error_Code_UNAUTHENTICATED:
      return absl::StatusCode::kUnauthenticated;
  }
}

#define ZKX_FFI_RETURN_IF_ERROR(expr)                                   \
  do {                                                                  \
    absl::Status _status = (expr);                                      \
    if (!_status.ok()) {                                                \
      ZKX_FFI_Error* _c_status = new ZKX_FFI_Error{std::move(_status)}; \
      return _c_status;                                                 \
    }                                                                   \
  } while (false)

ZKX_FFI_Error* ZKX_FFI_Error_Create(ZKX_FFI_Error_Create_Args* args) {
  ZKX_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "ZKX_FFI_Error_Create", ZKX_FFI_Error_Create_Args_STRUCT_SIZE,
      args->struct_size));

  return new ZKX_FFI_Error{
      absl::Status(ToStatusCode(args->errc), args->message)};
}

void ZKX_FFI_Error_GetMessage(ZKX_FFI_Error_GetMessage_Args* args) {
  absl::Status struct_size_check = ActualStructSizeIsGreaterOrEqual(
      "ZKX_FFI_Error_GetMessage", ZKX_FFI_Error_GetMessage_Args_STRUCT_SIZE,
      args->struct_size);
  if (!struct_size_check.ok()) {
    LOG(ERROR) << struct_size_check.message();
  }
  // absl::Status owns error message in a std::string which guarantees that
  // we'll get a null terminated string.
  args->message = args->error->status.message().data();
}

void ZKX_FFI_Error_Destroy(ZKX_FFI_Error_Destroy_Args* args) {
  absl::Status struct_size_check = ActualStructSizeIsGreaterOrEqual(
      "ZKX_FFI_Error_Destroy", ZKX_FFI_Error_Destroy_Args_STRUCT_SIZE,
      args->struct_size);
  if (!struct_size_check.ok()) {
    LOG(ERROR) << struct_size_check.message();
  }
  delete args->error;
}

ZKX_FFI_Error* ZKX_FFI_Future_Create(ZKX_FFI_Future_Create_Args* args) {
  ZKX_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "ZKX_FFI_Future_Create", ZKX_FFI_Future_Create_Args_STRUCT_SIZE,
      args->struct_size));
  args->future =
      new ZKX_FFI_Future{tsl::MakeConstructedAsyncValueRef<tsl::Chain>()};
  return nullptr;
}

ZKX_FFI_Error* ZKX_FFI_Future_SetAvailable(
    ZKX_FFI_Future_SetAvailable_Args* args) {
  ZKX_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "ZKX_FFI_Future_SetAvailable",
      ZKX_FFI_Future_SetAvailable_Args_STRUCT_SIZE, args->struct_size));
  args->future->async_value.SetStateConcrete();
  return nullptr;
}

ZKX_FFI_Error* ZKX_FFI_Future_SetError(ZKX_FFI_Future_SetError_Args* args) {
  ZKX_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "ZKX_FFI_Future_SetError", ZKX_FFI_Future_SetError_Args_STRUCT_SIZE,
      args->struct_size));

  if (args->error == nullptr || args->error->status.ok()) {
    return new ZKX_FFI_Error{
        absl::InvalidArgumentError("Error must not be null or OK")};
  }

  absl::Status error = TakeStatus(args->error);
  args->future->async_value.SetError(std::move(error));

  return nullptr;
}

ZKX_FFI_Error* ZKX_FFI_Handler_Register(ZKX_FFI_Handler_Register_Args* args) {
  ZKX_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "ZKX_FFI_Handler_Register", ZKX_FFI_Handler_Register_Args_STRUCT_SIZE,
      args->struct_size));

  if (auto status = RegisterHandler(
          std::string_view(args->name.ptr, args->name.len),
          std::string_view(args->platform.ptr, args->platform.len),
          args->bundle, args->traits);
      !status.ok()) {
    return new ZKX_FFI_Error{std::move(status)};
  }
  return nullptr;
}

ZKX_FFI_Error* ZKX_FFI_Stream_Get(ZKX_FFI_Stream_Get_Args* args) {
  ZKX_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "ZKX_FFI_Stream_Get", ZKX_FFI_Stream_Get_Args_STRUCT_SIZE,
      args->struct_size));

  auto* gpu = std::get_if<ZKX_FFI_ExecutionContext::GpuContext>(
      &args->ctx->backend_context);

  if (ABSL_PREDICT_FALSE(gpu == nullptr)) {
    return new ZKX_FFI_Error{
        absl::UnimplementedError("ZKX FFI GPU context is not available")};
  }

  if (ABSL_PREDICT_FALSE(gpu->stream == nullptr)) {
    return new ZKX_FFI_Error{
        absl::UnimplementedError("ZKX FFI GPU stream is not available")};
  }

  auto handle = gpu->stream->platform_specific_handle();
  args->stream = handle.stream;

  return nullptr;
}

ZKX_FFI_Error* ZKX_FFI_RunId_Get(ZKX_FFI_RunId_Get_Args* args) {
  ZKX_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "ZKX_FFI_RunId_Get", ZKX_FFI_RunId_Get_Args_STRUCT_SIZE,
      args->struct_size));

  args->run_id = args->ctx->run_id.ToInt();

  return nullptr;
}

ZKX_FFI_Error* ZKX_FFI_TypeId_Register(ZKX_FFI_TypeId_Register_Args* args) {
  ZKX_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "ZKX_FFI_ExecutionContext_Get_Args",
      ZKX_FFI_ExecutionContext_Get_Args_STRUCT_SIZE, args->struct_size));

  auto type_id = TypeIdRegistry::RegisterExternalTypeId(
      std::string_view(args->name.ptr, args->name.len));
  if (!type_id.ok()) {
    return new ZKX_FFI_Error{std::move(type_id).status()};
  }

  args->type_id->type_id = type_id->value();
  return nullptr;
}

ZKX_FFI_Error* ZKX_FFI_ExecutionContext_Get(
    ZKX_FFI_ExecutionContext_Get_Args* args) {
  ZKX_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "ZKX_FFI_ExecutionContext_Get_Args",
      ZKX_FFI_ExecutionContext_Get_Args_STRUCT_SIZE, args->struct_size));

  DCHECK(args->ctx->execution_context) << "ExecutionContext must be set";
  auto user_data = args->ctx->execution_context->Lookup(
      TypeIdRegistry::TypeId(args->type_id->type_id));
  if (!user_data.ok()) {
    return new ZKX_FFI_Error{std::move(user_data).status()};
  }

  args->data = *user_data;
  return nullptr;
}

ZKX_FFI_Error* ZKX_FFI_State_Set(ZKX_FFI_State_Set_Args* args) {
  ZKX_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "ZKX_FFI_State_Set_Args", ZKX_FFI_State_Set_Args_STRUCT_SIZE,
      args->struct_size));

  DCHECK(args->ctx->execution_state) << "ExecutionState must be set";
  absl::Status status = args->ctx->execution_state->Set(
      TypeIdRegistry::TypeId(args->type_id->type_id), args->state,
      [deleter = args->deleter](void* state) { deleter(state); });

  if (!status.ok()) {
    return new ZKX_FFI_Error{std::move(status)};
  }

  return nullptr;
}

ZKX_FFI_Error* ZKX_FFI_State_Get(ZKX_FFI_State_Get_Args* args) {
  ZKX_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "ZKX_FFI_State_Get_Args", ZKX_FFI_State_Get_Args_STRUCT_SIZE,
      args->struct_size));

  DCHECK(args->ctx->execution_state) << "ExecutionState must be set";
  absl::StatusOr<void*> state = args->ctx->execution_state->Get(
      TypeIdRegistry::TypeId(args->type_id->type_id));
  if (!state.ok()) {
    return new ZKX_FFI_Error{std::move(state).status()};
  }

  args->state = *state;
  return nullptr;
}

ZKX_FFI_Error* ZKX_FFI_DeviceMemory_Allocate(
    ZKX_FFI_DeviceMemory_Allocate_Args* args) {
  ZKX_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "ZKX_FFI_DeviceMemory_Allocate_Args",
      ZKX_FFI_DeviceMemory_Allocate_Args_STRUCT_SIZE, args->struct_size));

  auto* gpu = std::get_if<ZKX_FFI_ExecutionContext::GpuContext>(
      &args->ctx->backend_context);

  // TODO(ezhulenev): Device memory allocation should be supported for all
  // backends, not just GPU, although for CPU it doesn't make much sense, as
  // plain `new` is sufficient.
  if (ABSL_PREDICT_FALSE(gpu == nullptr)) {
    return new ZKX_FFI_Error{
        absl::InvalidArgumentError("ZKX FFI GPU context is not available")};
  }

  if (ABSL_PREDICT_FALSE(gpu->allocator == nullptr)) {
    return new ZKX_FFI_Error{absl::UnimplementedError(
        "No device memory allocator available on this platform")};
  }

  // TODO(ezhulenev): We happen to have the same alignment requirement for
  // device memory on CPU and GPU backends, but instead of hardcoding it here
  // we should query it for the platform ZKX FFI handler is registered with.
  static constexpr int64_t kMaxAlignment = 16;

  if (!absl::has_single_bit(args->alignment) ||
      args->alignment > kMaxAlignment) {
    return new ZKX_FFI_Error{absl::InvalidArgumentError(
        absl::StrCat("Unsupported alignment: ", args->alignment))};
  }

  absl::StatusOr<stream_executor::OwningDeviceMemory> memory =
      gpu->allocator->Allocate(args->ctx->device_ordinal, args->size);
  if (!memory.ok()) {
    return new ZKX_FFI_Error{std::move(memory).status()};
  }

  args->data = memory->Release().opaque();
  return nullptr;
}

ZKX_FFI_Error* ZKX_FFI_DeviceMemory_Free(ZKX_FFI_DeviceMemory_Free_Args* args) {
  ZKX_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "ZKX_FFI_DeviceMemory_Free_Args",
      ZKX_FFI_DeviceMemory_Free_Args_STRUCT_SIZE, args->struct_size));

  auto* gpu = std::get_if<ZKX_FFI_ExecutionContext::GpuContext>(
      &args->ctx->backend_context);

  // TODO(ezhulenev): Device memory allocation should be supported for all
  // backends, not just GPU, although for CPU it doesn't make much sense, as
  // plain `new` is sufficient.
  if (ABSL_PREDICT_FALSE(gpu == nullptr)) {
    return new ZKX_FFI_Error{
        absl::UnimplementedError("ZKX FFI GPU context is not available")};
  }

  if (ABSL_PREDICT_FALSE(gpu->allocator == nullptr)) {
    return new ZKX_FFI_Error{absl::UnimplementedError(
        "No device memory allocator available on this platform")};
  }

  absl::Status status = gpu->allocator->Deallocate(
      args->ctx->device_ordinal,
      stream_executor::DeviceMemoryBase(args->data, args->size));
  if (!status.ok()) {
    return new ZKX_FFI_Error{std::move(status)};
  }

  return nullptr;
}

absl::StatusOr<const Eigen::ThreadPoolDevice*> GetIntraOpThreadPool(
    const ZKX_FFI_ExecutionContext* ctx) {
  auto* cpu =
      std::get_if<ZKX_FFI_ExecutionContext::CpuContext>(&ctx->backend_context);

  if (ABSL_PREDICT_FALSE(cpu == nullptr)) {
    return absl::UnimplementedError("ZKX FFI CPU context is not available");
  }

  if (ABSL_PREDICT_FALSE(cpu->intra_op_thread_pool == nullptr)) {
    return absl::UnimplementedError(
        "No intra-op thread pool available on this platform");
  }

  return cpu->intra_op_thread_pool;
}

ZKX_FFI_Error* ZKX_FFI_ThreadPool_Schedule(
    ZKX_FFI_ThreadPool_Schedule_Args* args) {
  ZKX_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "ZKX_FFI_ThreadPool_Schedule_Args",
      ZKX_FFI_ThreadPool_Schedule_Args_STRUCT_SIZE, args->struct_size));

  auto intra_op_thread_pool = GetIntraOpThreadPool(args->ctx);
  if (!intra_op_thread_pool.ok()) {
    return new ZKX_FFI_Error{std::move(intra_op_thread_pool).status()};
  }

  (*intra_op_thread_pool)
      ->enqueueNoNotification(
          [task = args->task, data = args->data] { (*task)(data); });

  return nullptr;
}

ZKX_FFI_Error* ZKX_FFI_ThreadPool_NumThreads(
    ZKX_FFI_ThreadPool_NumThreads_Args* args) {
  ZKX_FFI_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "ZKX_FFI_ThreadPool_NumThreads_Args",
      ZKX_FFI_ThreadPool_NumThreads_Args_STRUCT_SIZE, args->struct_size));

  auto intra_op_thread_pool = GetIntraOpThreadPool(args->ctx);
  if (!intra_op_thread_pool.ok()) {
    return new ZKX_FFI_Error{std::move(intra_op_thread_pool).status()};
  }

  *args->num_threads = (*intra_op_thread_pool)->numThreadsInPool();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ZKX FFI Internal Api Implementation
//===----------------------------------------------------------------------===//

ZKX_FFI_Error* ZKX_FFI_INTERNAL_Error_Forward(void* status) {
  auto* absl_status = reinterpret_cast<absl::Status*>(status);
  if (ABSL_PREDICT_TRUE(absl_status->ok())) {
    return nullptr;
  }
  return new ZKX_FFI_Error{std::move(*absl_status)};
}

ZKX_FFI_Future* ZKX_FFI_INTERNAL_Future_Forward(void* async_value) {
  auto* tsl_async_value = reinterpret_cast<tsl::AsyncValue*>(async_value);
  DCHECK(tsl_async_value) << "Async value must not be null";

  return new ZKX_FFI_Future{
      tsl::AsyncValueRef<tsl::Chain>(tsl::TakeRef(tsl_async_value))};
}

void* ZKX_FFI_INTERNAL_Stream_Get(ZKX_FFI_ExecutionContext* ctx) {
  if (auto* gpu = std::get_if<ZKX_FFI_ExecutionContext::GpuContext>(
          &ctx->backend_context)) {
    return gpu->stream;
  }

  return nullptr;
}

int32_t ZKX_FFI_INTERNAL_DeviceOrdinal_Get(ZKX_FFI_ExecutionContext* ctx) {
  return ctx->device_ordinal;
}

int64_t ZKX_FFI_INTERNAL_RunId_Get(ZKX_FFI_ExecutionContext* ctx) {
  return ctx->run_id.ToInt();
}

void* ZKX_FFI_INTERNAL_DeviceMemoryAllocator_Get(
    ZKX_FFI_ExecutionContext* ctx) {
  if (auto* gpu = std::get_if<ZKX_FFI_ExecutionContext::GpuContext>(
          &ctx->backend_context)) {
    return gpu->allocator;
  }

  return nullptr;
}

void* ZKX_FFI_INTERNAL_CalledComputation_Get(ZKX_FFI_ExecutionContext* ctx) {
  return const_cast<HloComputation*>(ctx->called_computation);
}

void* ZKX_FFI_INTERNAL_ExecutionContext_Get(ZKX_FFI_ExecutionContext* ctx) {
  return const_cast<ffi::ExecutionContext*>(ctx->execution_context);
}

void* ZKX_FFI_INTERNAL_ExecutionState_Get(ZKX_FFI_ExecutionContext* ctx) {
  return const_cast<ffi::ExecutionState*>(ctx->execution_state);
}

}  // namespace

void* ZKX_FFI_INTERNAL_IntraOpThreadPool_Get(ZKX_FFI_ExecutionContext* ctx) {
  if (auto* cpu = std::get_if<ZKX_FFI_ExecutionContext::CpuContext>(
          &ctx->backend_context)) {
    return const_cast<Eigen::ThreadPoolDevice*>(cpu->intra_op_thread_pool);
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// ZKX FFI Api access
//===----------------------------------------------------------------------===//

extern "C" const ZKX_FFI_Api* ZKX_FFI_GetApi() { return GetZkxFfiApi(); }

namespace {

ZKX_FFI_InternalApi internal_api = {
    ZKX_FFI_INTERNAL_Error_Forward,
    ZKX_FFI_INTERNAL_Future_Forward,
    ZKX_FFI_INTERNAL_Stream_Get,
    ZKX_FFI_INTERNAL_DeviceOrdinal_Get,
    ZKX_FFI_INTERNAL_RunId_Get,
    ZKX_FFI_INTERNAL_DeviceMemoryAllocator_Get,
    ZKX_FFI_INTERNAL_CalledComputation_Get,
    ZKX_FFI_INTERNAL_ExecutionContext_Get,
    ZKX_FFI_INTERNAL_ExecutionState_Get,
    ZKX_FFI_INTERNAL_IntraOpThreadPool_Get,
};

ZKX_FFI_Api api = {
    ZKX_FFI_Api_STRUCT_SIZE,
    /*extension_start=*/nullptr,

    ZKX_FFI_Api_Version{
        ZKX_FFI_Api_Version_STRUCT_SIZE,
        /*extension_start=*/nullptr,
        ZKX_FFI_API_MAJOR,
        ZKX_FFI_API_MINOR,
    },

    &internal_api,

    ZKX_FFI_Error_Create,
    ZKX_FFI_Error_GetMessage,
    ZKX_FFI_Error_Destroy,
    ZKX_FFI_Handler_Register,
    ZKX_FFI_Stream_Get,
    ZKX_FFI_TypeId_Register,
    ZKX_FFI_ExecutionContext_Get,
    ZKX_FFI_State_Set,
    ZKX_FFI_State_Get,
    ZKX_FFI_DeviceMemory_Allocate,
    ZKX_FFI_DeviceMemory_Free,
    ZKX_FFI_ThreadPool_Schedule,
    ZKX_FFI_ThreadPool_NumThreads,
    ZKX_FFI_Future_Create,
    ZKX_FFI_Future_SetAvailable,
    ZKX_FFI_Future_SetError,
    ZKX_FFI_RunId_Get,
};

}  // namespace

const ZKX_FFI_Api* GetZkxFfiApi() { return &api; }

}  // namespace zkx::ffi
