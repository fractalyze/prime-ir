/* Copyright 2022 The OpenXLA Authors.
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

#include "zkx/pjrt/c/pjrt_c_api_helpers.h"

#include <algorithm>
#include <utility>
#include <variant>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"

#include "stablehlo/dialect/Version.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/lib/connected_traceme.h"
#include "xla/tsl/profiler/lib/context_types.h"
#include "zk_dtypes/include/all_types.h"
#include "zkx/pjrt/c/pjrt_c_api_memory_descriptions_extension.h"
#include "zkx/pjrt/pjrt_device_description.h"
#include "zkx/primitive_util.h"
#include "zkx/shape_util.h"
#include "zkx/util.h"

namespace pjrt {

const std::string_view kHloFormat = "hlo";
const std::string_view kMlirFormat = "mlir";
const std::string_view kHloWithConfigFormat = "hlo_with_config";

PJRT_ClientDeleter MakeClientDeleter(const PJRT_Api* api) {
  return [api](PJRT_Client* client) -> void {
    PJRT_Client_Destroy_Args destroy_args;
    destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    destroy_args.extension_start = nullptr;
    destroy_args.client = client;

    PJRT_Error* error = api->PJRT_Client_Destroy(&destroy_args);
    // TODO(b/236710439): handle the error and remove this CHECK() call
    CHECK(error == nullptr);
  };
}

PJRT_AsyncHostToDeviceTransferManagerDeleter
MakeAsyncHostToDeviceTransferManagerDeleter(const PJRT_Api* api) {
  return [api](
             PJRT_AsyncHostToDeviceTransferManager* transfer_manager) -> void {
    PJRT_AsyncHostToDeviceTransferManager_Destroy_Args destroy_args;
    destroy_args.struct_size =
        PJRT_AsyncHostToDeviceTransferManager_Destroy_Args_STRUCT_SIZE;
    destroy_args.extension_start = nullptr;
    destroy_args.transfer_manager = transfer_manager;
    pjrt::LogFatalIfPjrtError(
        api->PJRT_AsyncHostToDeviceTransferManager_Destroy(&destroy_args), api);
  };
}

PJRT_ErrorDeleter MakeErrorDeleter(const PJRT_Api* api) {
  return [api](PJRT_Error* error) -> void {
    PJRT_Error_Destroy_Args destroy_args;
    destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.extension_start = nullptr;
    destroy_args.error = error;

    api->PJRT_Error_Destroy(&destroy_args);
  };
}

PJRT_BufferDeleter MakeBufferDeleter(const PJRT_Api* api) {
  return [api](PJRT_Buffer* buffer) -> void {
    PJRT_Buffer_Destroy_Args destroy_args;
    destroy_args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    destroy_args.extension_start = nullptr;
    destroy_args.buffer = buffer;

    pjrt::LogFatalIfPjrtError(api->PJRT_Buffer_Destroy(&destroy_args), api);
  };
}

PJRT_ExecutableDeleter MakeExecutableDeleter(const PJRT_Api* api) {
  return [api](PJRT_Executable* executable) -> void {
    PJRT_Executable_Destroy_Args args;
    args.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.executable = executable;
    pjrt::LogFatalIfPjrtError(api->PJRT_Executable_Destroy(&args), api);
  };
}

PJRT_LoadedExecutableDeleter MakeLoadedExecutableDeleter(const PJRT_Api* api) {
  return [api](PJRT_LoadedExecutable* executable) -> void {
    PJRT_LoadedExecutable_Destroy_Args args;
    args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.executable = executable;
    pjrt::LogFatalIfPjrtError(api->PJRT_LoadedExecutable_Destroy(&args), api);
  };
}

absl::Status PjrtErrorToStatus(const PJRT_Error* error, const PJRT_Api* api) {
  absl::Status status;
  if (error != nullptr) {
    status = absl::Status(PjrtErrorToStatusCode(error, api),
                          GetPjrtErrorMessage(error, api));
  }
  return status;
}

PJRT_TopologyDescriptionDeleter MakeTopologyDescriptionDeleter(
    const PJRT_Api* api) {
  return [api](PJRT_TopologyDescription* topology) -> void {
    PJRT_TopologyDescription_Destroy_Args destroy_args;
    destroy_args.struct_size =
        PJRT_TopologyDescription_Destroy_Args_STRUCT_SIZE;
    destroy_args.extension_start = nullptr;
    destroy_args.topology = topology;

    pjrt::LogFatalIfPjrtError(
        api->PJRT_TopologyDescription_Destroy(&destroy_args), api);
  };
}

PJRT_Layouts_MemoryLayoutDeleter MakeMemoryLayoutDeleter(const PJRT_Api* api) {
  PJRT_Layouts_Extension* ext_api =
      FindExtension<PJRT_Layouts_Extension>(api, PJRT_Extension_Type_Layouts);
  CHECK_NE(ext_api, nullptr) << "MakeMemoryLayoutDeleter passed PJRT_Api that "
                                "doesn't support layouts extension";
  return [api, ext_api](PJRT_Layouts_MemoryLayout* layout) -> void {
    PJRT_Layouts_MemoryLayout_Destroy_Args args;
    args.struct_size = PJRT_Layouts_MemoryLayout_Destroy_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.layout = layout;
    pjrt::LogFatalIfPjrtError(ext_api->PJRT_Layouts_MemoryLayout_Destroy(&args),
                              api);
  };
}

PJRT_Error_Code GetErrorCode(const PJRT_Error* error, const PJRT_Api* api) {
  PJRT_Error_GetCode_Args args;
  args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.error = error;
  pjrt::LogFatalIfPjrtError(api->PJRT_Error_GetCode(&args), api);
  return args.code;
}

absl::StatusCode PjrtErrorToStatusCode(const PJRT_Error* error,
                                       const PJRT_Api* api) {
  return PjrtErrorCodeToStatusCode(GetErrorCode(error, api));
}

absl::StatusCode PjrtErrorCodeToStatusCode(PJRT_Error_Code code) {
  switch (code) {
    case PJRT_Error_Code_CANCELLED:
    case PJRT_Error_Code_UNKNOWN:
    case PJRT_Error_Code_INVALID_ARGUMENT:
    case PJRT_Error_Code_DEADLINE_EXCEEDED:
    case PJRT_Error_Code_NOT_FOUND:
    case PJRT_Error_Code_ALREADY_EXISTS:
    case PJRT_Error_Code_PERMISSION_DENIED:
    case PJRT_Error_Code_RESOURCE_EXHAUSTED:
    case PJRT_Error_Code_FAILED_PRECONDITION:
    case PJRT_Error_Code_ABORTED:
    case PJRT_Error_Code_OUT_OF_RANGE:
    case PJRT_Error_Code_UNIMPLEMENTED:
    case PJRT_Error_Code_INTERNAL:
    case PJRT_Error_Code_UNAVAILABLE:
    case PJRT_Error_Code_DATA_LOSS:
    case PJRT_Error_Code_UNAUTHENTICATED:
      return static_cast<absl::StatusCode>(code);
  }
}

// ZKX: Changed - Rewrote to use absl::StatusCode directly instead of
// tsl::error::Code (error_codes.pb.h not available in ZKX).
PJRT_Error_Code StatusCodeToPjrtErrorCode(absl::StatusCode code) {
  switch (code) {
    case absl::StatusCode::kCancelled:
    case absl::StatusCode::kUnknown:
    case absl::StatusCode::kInvalidArgument:
    case absl::StatusCode::kDeadlineExceeded:
    case absl::StatusCode::kNotFound:
    case absl::StatusCode::kAlreadyExists:
    case absl::StatusCode::kPermissionDenied:
    case absl::StatusCode::kUnauthenticated:
    case absl::StatusCode::kResourceExhausted:
    case absl::StatusCode::kFailedPrecondition:
    case absl::StatusCode::kAborted:
    case absl::StatusCode::kOutOfRange:
    case absl::StatusCode::kUnimplemented:
    case absl::StatusCode::kInternal:
    case absl::StatusCode::kUnavailable:
    case absl::StatusCode::kDataLoss:
      return static_cast<PJRT_Error_Code>(code);
    case absl::StatusCode::kOk:
      CHECK(false) << "Status::OK() cannot be converted to PJRT_Error code, "
                      "use nullptr instead";
    default:
      CHECK(false) << "Unexpected status code: " << static_cast<int>(code);
  }
}

std::string_view GetPjrtErrorMessage(const PJRT_Error* error,
                                     const PJRT_Api* api) {
  PJRT_Error_Message_Args message_args;
  message_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  message_args.extension_start = nullptr;
  message_args.error = error;
  api->PJRT_Error_Message(&message_args);
  return std::string_view(message_args.message, message_args.message_size);
}

void LogFatalIfPjrtError(PJRT_Error* error, const PJRT_Api* api) {
  std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> _error(
      error, MakeErrorDeleter(api));
  absl::Status _status = PjrtErrorToStatus(_error.get(), api);
  if (!_status.ok()) {
    LOG(FATAL) << "Unexpected error status " << _status.message();
  }
}

PJRT_EventDeleter MakeEventDeleter(const PJRT_Api* api) {
  CHECK(api != nullptr);
  return [api](PJRT_Event* managed) {
    PJRT_Event_Destroy_Args args;
    args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.event = managed;

    LogFatalIfPjrtError(api->PJRT_Event_Destroy(&args), api);
  };
}

PJRT_Buffer_Type ConvertToPjRtBufferType(zkx::PrimitiveType type) {
  switch (type) {
    case zkx::PrimitiveType::PRIMITIVE_TYPE_INVALID:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_INVALID;
    case zkx::PrimitiveType::PRED:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_PRED;
    case zkx::PrimitiveType::TOKEN:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_TOKEN;
    case zkx::PrimitiveType::S2:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_S2;
    case zkx::PrimitiveType::S4:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_S4;
    case zkx::PrimitiveType::S8:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_S8;
    case zkx::PrimitiveType::S16:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_S16;
    case zkx::PrimitiveType::S32:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_S32;
    case zkx::PrimitiveType::S64:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_S64;
    case zkx::PrimitiveType::U2:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_U2;
    case zkx::PrimitiveType::U4:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_U4;
    case zkx::PrimitiveType::U8:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_U8;
    case zkx::PrimitiveType::U16:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_U16;
    case zkx::PrimitiveType::U32:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_U32;
    case zkx::PrimitiveType::U64:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_U64;
    // ZKX: Removed - floating-point
    // F16, F32, F64, BF16, F4E2M1FN, F8E5M2, F8E4M3, F8E4M3FN,
    // F8E4M3B11FNUZ, F8E5M2FNUZ, F8E4M3FNUZ, F8E3M4, F8E8M0FNU, C64, C128
    // ZKX: Added - Extended unsigned integer types
    case zkx::PrimitiveType::U128:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_U128;
    case zkx::PrimitiveType::U256:
      return PJRT_Buffer_Type::PJRT_Buffer_Type_U256;
// ZKX: Added - ZK types
#define ZK_DTYPES_CASE(cpp_type, unused, enum, lowercase_name) \
  case zkx::PrimitiveType::enum:                               \
    return PJRT_Buffer_Type::PJRT_Buffer_Type_##enum;
      ZK_DTYPES_PUBLIC_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE
    default:
      CHECK(false)
          << "Element type of the shape is not supported in C API layer: "
          << zkx::primitive_util::LowercasePrimitiveTypeName(type);
  }
}

zkx::PrimitiveType ConvertFromPjRtBufferType(PJRT_Buffer_Type type) {
  switch (type) {
    case PJRT_Buffer_Type::PJRT_Buffer_Type_PRED:
      return zkx::PrimitiveType::PRED;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_TOKEN:
      return zkx::PrimitiveType::TOKEN;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_S2:
      return zkx::PrimitiveType::S2;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_S4:
      return zkx::PrimitiveType::S4;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_S8:
      return zkx::PrimitiveType::S8;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_S16:
      return zkx::PrimitiveType::S16;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_S32:
      return zkx::PrimitiveType::S32;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_S64:
      return zkx::PrimitiveType::S64;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_U2:
      return zkx::PrimitiveType::U2;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_U4:
      return zkx::PrimitiveType::U4;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_U8:
      return zkx::PrimitiveType::U8;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_U16:
      return zkx::PrimitiveType::U16;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_U32:
      return zkx::PrimitiveType::U32;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_U64:
      return zkx::PrimitiveType::U64;
    // ZKX: Removed - floating-point
    // F16, F32, F64, BF16, F4E2M1FN, F8E5M2, F8E4M3, F8E4M3FN,
    // F8E4M3B11FNUZ, F8E5M2FNUZ, F8E4M3FNUZ, F8E3M4, F8E8M0FNU, C64, C128
    // ZKX: Added - Extended unsigned integer types
    case PJRT_Buffer_Type::PJRT_Buffer_Type_U128:
      return zkx::PrimitiveType::U128;
    case PJRT_Buffer_Type::PJRT_Buffer_Type_U256:
      return zkx::PrimitiveType::U256;
// ZKX: Added - ZK types
#define ZK_DTYPES_CASE(cpp_type, unused, enum, lowercase_name) \
  case PJRT_Buffer_Type::PJRT_Buffer_Type_##enum:              \
    return zkx::PrimitiveType::enum;
      ZK_DTYPES_PUBLIC_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE
    case PJRT_Buffer_Type::PJRT_Buffer_Type_INVALID:
      CHECK(false) << "Buffer type is not supported in C API layer.";
  }
}

// static
const char* HostBufferSemanticsToString(
    zkx::PjRtClient::HostBufferSemantics h) {
  switch (h) {
    case zkx::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall:
      return "zkx::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall";
    case zkx::PjRtClient::HostBufferSemantics::kImmutableZeroCopy:
      return "zkx::PjRtClient::HostBufferSemantics::kImmutableZeroCopy";
    case zkx::PjRtClient::HostBufferSemantics::kMutableZeroCopy:
      return "zkx::PjRtClient::HostBufferSemantics::kMutableZeroCopy";
    case zkx::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes:
      return "zkx::PjRtClient::HostBufferSemantics::"
             "kImmutableUntilTransferCompletes";
  }
}

PJRT_HostBufferSemantics ConvertToPjRtHostBufferSemantics(
    zkx::PjRtClient::HostBufferSemantics buffer_semantics) {
  switch (buffer_semantics) {
    case zkx::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall:
      return PJRT_HostBufferSemantics::
          PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    case zkx::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes:
      return PJRT_HostBufferSemantics::
          PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
    case zkx::PjRtClient::HostBufferSemantics::kImmutableZeroCopy:
      return PJRT_HostBufferSemantics::
          PJRT_HostBufferSemantics_kImmutableZeroCopy;
    case zkx::PjRtClient::HostBufferSemantics::kMutableZeroCopy:
      return PJRT_HostBufferSemantics::
          PJRT_HostBufferSemantics_kMutableZeroCopy;
    default:
      CHECK(false)
          << "Input host buffer semantics is not supported in C API layer: "
          << HostBufferSemanticsToString(buffer_semantics);
  }
}

zkx::PjRtClient::HostBufferSemantics ConvertFromPjRtHostBufferSemantics(
    PJRT_HostBufferSemantics buffer_semantics) {
  switch (buffer_semantics) {
    case PJRT_HostBufferSemantics::
        PJRT_HostBufferSemantics_kImmutableOnlyDuringCall:
      return zkx::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall;
    case PJRT_HostBufferSemantics::
        PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes:
      return zkx::PjRtClient::HostBufferSemantics::
          kImmutableUntilTransferCompletes;
    case PJRT_HostBufferSemantics::PJRT_HostBufferSemantics_kImmutableZeroCopy:
      return zkx::PjRtClient::HostBufferSemantics::kImmutableZeroCopy;
    case PJRT_HostBufferSemantics::PJRT_HostBufferSemantics_kMutableZeroCopy:
      return zkx::PjRtClient::HostBufferSemantics::kMutableZeroCopy;
  }
}

zkx::PjRtFuture<> ConvertCEventToCppFuture(PJRT_Event* c_event,
                                           const PJRT_Api* c_api) {
  using zkx::PjRtFuture;
  PJRT_Event_OnReady_Args event_onready_args;
  event_onready_args.struct_size = PJRT_Event_OnReady_Args_STRUCT_SIZE;
  event_onready_args.extension_start = nullptr;
  event_onready_args.event = c_event;

  PjRtFuture<>::Promise promise = PjRtFuture<>::CreatePromise();
  event_onready_args.user_arg = new std::function<void(PJRT_Error*)>(
      [promise, c_event, c_api](PJRT_Error* error) mutable {
        if (error != nullptr) {
          promise.Set(::pjrt::PjrtErrorToStatus(error, c_api));
          ::pjrt::MakeErrorDeleter(c_api)(error);
        } else {
          promise.Set();
        }
        ::pjrt::MakeEventDeleter(c_api)(c_event);
      });
  event_onready_args.callback = [](PJRT_Error* error, void* arg) {
    std::function<void(PJRT_Error*)>* set_future =
        reinterpret_cast<std::function<void(PJRT_Error*)>*>(arg);
    (*set_future)(error);
    delete set_future;
  };

  PJRT_Error* error = c_api->PJRT_Event_OnReady(&event_onready_args);
  if (error != nullptr) {
    absl::Status status = ::pjrt::PjrtErrorToStatus(error, c_api);
    ::pjrt::MakeErrorDeleter(c_api)(error);
    delete reinterpret_cast<std::function<void(PJRT_Error*)>*>(
        event_onready_args.user_arg);
    ::pjrt::MakeEventDeleter(c_api)(c_event);
    return PjRtFuture<>(std::move(status));
  }
  return PjRtFuture<>(std::move(promise));
}

namespace {

absl::StatusOr<PJRT_NamedValue> ConvertToPjRtNamedValue(
    const std::string& name, const zkx::PjRtValueType& value) {
  PJRT_NamedValue c_value;
  c_value.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  c_value.extension_start = nullptr;
  c_value.name = name.c_str();
  c_value.name_size = name.size();

  if (std::holds_alternative<std::string>(value)) {
    c_value.type = PJRT_NamedValue_Type::PJRT_NamedValue_kString;
    const std::string& option_string_value = std::get<std::string>(value);
    c_value.string_value = option_string_value.c_str();
    c_value.value_size = option_string_value.size();
  } else if (std::holds_alternative<int64_t>(value)) {
    c_value.type = PJRT_NamedValue_Type::PJRT_NamedValue_kInt64;
    c_value.int64_value = std::get<int64_t>(value);
    c_value.value_size = 1;
  } else if (std::holds_alternative<std::vector<int64_t>>(value)) {
    c_value.type = PJRT_NamedValue_Type::PJRT_NamedValue_kInt64List;
    const std::vector<int64_t>& option_int_list_value =
        std::get<std::vector<int64_t>>(value);
    c_value.int64_array_value = option_int_list_value.data();
    c_value.value_size = option_int_list_value.size();
  } else if (std::holds_alternative<float>(value)) {
    c_value.type = PJRT_NamedValue_Type::PJRT_NamedValue_kFloat;
    c_value.float_value = std::get<float>(value);
    c_value.value_size = 1;
  } else if (std::holds_alternative<bool>(value)) {
    c_value.type = PJRT_NamedValue_Type::PJRT_NamedValue_kBool;
    c_value.bool_value = std::get<bool>(value);
    c_value.value_size = 1;
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unexpected PjRtValueType: '", value.index(), " with name: ", name));
  }

  return c_value;
}

}  // namespace

absl::StatusOr<std::vector<PJRT_NamedValue>> ConvertToPjRtNamedValueList(
    const absl::flat_hash_map<std::string, zkx::PjRtValueType>& cpp_value_map) {
  std::vector<PJRT_NamedValue> c_value_list;
  c_value_list.reserve(cpp_value_map.size());
  for (const auto& [name, value] : cpp_value_map) {
    TF_ASSIGN_OR_RETURN(PJRT_NamedValue c_value,
                        ConvertToPjRtNamedValue(name, value));
    c_value_list.push_back(c_value);
  }
  return c_value_list;
}

absl::flat_hash_map<std::string, zkx::PjRtValueType>
ConvertFromPjRtNamedValueList(const PJRT_NamedValue* c_value_list,
                              size_t list_size) {
  absl::flat_hash_map<std::string, zkx::PjRtValueType> cpp_value_map;
  for (int i = 0; i < list_size; ++i) {
    const PJRT_NamedValue& c_value = c_value_list[i];
    std::string_view name = std::string_view(c_value.name, c_value.name_size);
    switch (c_value.type) {
      case PJRT_NamedValue_Type::PJRT_NamedValue_kString: {
        std::string string_value(c_value.string_value, c_value.value_size);
        cpp_value_map[name] = zkx::PjRtValueType(string_value);
        break;
      }
      case PJRT_NamedValue_Type::PJRT_NamedValue_kInt64: {
        cpp_value_map[name] = zkx::PjRtValueType(c_value.int64_value);
        break;
      }
      case PJRT_NamedValue_Type::PJRT_NamedValue_kInt64List: {
        const int64_t* array_ptr(c_value.int64_array_value);
        std::vector<int64_t> int64_array(array_ptr,
                                         array_ptr + c_value.value_size);
        cpp_value_map[name] = zkx::PjRtValueType(int64_array);
        break;
      }
      case PJRT_NamedValue_Type::PJRT_NamedValue_kFloat: {
        cpp_value_map[name] = zkx::PjRtValueType(c_value.float_value);
        break;
      }
      case PJRT_NamedValue_Type::PJRT_NamedValue_kBool: {
        cpp_value_map[name] = zkx::PjRtValueType(c_value.bool_value);
        break;
      }
      default: {
        LOG(FATAL) << "Unexpected PJRT_NamedValue type: " << c_value.type
                   << " with name: " << name;
        break;
      }
    }
  }
  return cpp_value_map;
}

namespace {

absl::StatusOr<PJRT_NamedValue_Type> GetPjrtNamedValueType(
    zkx::PjRtValueType cpp_value) {
  if (std::holds_alternative<std::string>(cpp_value)) {
    return PJRT_NamedValue_Type::PJRT_NamedValue_kString;
  }
  if (std::holds_alternative<int64_t>(cpp_value)) {
    return PJRT_NamedValue_Type::PJRT_NamedValue_kInt64;
  }
  if (std::holds_alternative<std::vector<int64_t>>(cpp_value)) {
    return PJRT_NamedValue_Type::PJRT_NamedValue_kInt64List;
  }
  if (std::holds_alternative<float>(cpp_value)) {
    return PJRT_NamedValue_Type::PJRT_NamedValue_kFloat;
  }
  if (std::holds_alternative<bool>(cpp_value)) {
    return PJRT_NamedValue_Type::PJRT_NamedValue_kBool;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unexpected PjRtValueType with index", cpp_value.index()));
}

}  // namespace

absl::Status ValidateCreateOptions(
    const absl::flat_hash_map<std::string, zkx::PjRtValueType>& value_map,
    const absl::flat_hash_map<std::string, PJRT_NamedValue_Type>&
        expected_name_and_types) {
  for (const auto& [name, value] : value_map) {
    auto it = expected_name_and_types.find(name);
    if (it == expected_name_and_types.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Unexpected option name passed to PJRT_Client_Create: ", name));
    }
    TF_ASSIGN_OR_RETURN(PJRT_NamedValue_Type type,
                        GetPjrtNamedValueType(value));
    if (type != it->second) {
      return absl::InvalidArgumentError(
          absl::StrCat("Option passed to PJRT_Client_Create with name ", name,
                       " has type index ", value.index(),
                       " but expected type index is ", it->second));
    }
  }
  return absl::OkStatus();
}

namespace {

PJRT_NamedValue ZkxVersion(std::string_view name) {
  PJRT_NamedValue c_value;
  c_value.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  c_value.extension_start = nullptr;
  c_value.name = name.data();
  c_value.name_size = name.size();
  c_value.type = PJRT_NamedValue_Type::PJRT_NamedValue_kInt64;
  // TODO(b/327203806): figure out where to keep the zkx_version.
  c_value.int64_value = 2;
  c_value.value_size = 1;
  return c_value;
}

template <int storage_slot>
PJRT_NamedValue StableHloVersion(std::string_view name,
                                 mlir::vhlo::Version version) {
  PJRT_NamedValue c_value;
  c_value.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  c_value.extension_start = nullptr;
  c_value.name = name.data();
  c_value.name_size = name.size();
  c_value.type = PJRT_NamedValue_Type::PJRT_NamedValue_kInt64List;
  static int64_t triple[3] = {version.getMajor(), version.getMinor(),
                              version.getPatch()};
  c_value.int64_array_value = triple;
  c_value.value_size = 3;
  return c_value;
}

}  // namespace

const std::vector<PJRT_NamedValue>& GetZkxPluginCAttributes() {
  static const std::vector<PJRT_NamedValue>* c_values =
      new std::vector<PJRT_NamedValue>({
          ZkxVersion("zkx_version"),
          StableHloVersion<0>("stablehlo_current_version",
                              mlir::vhlo::Version::getCurrentVersion()),
          StableHloVersion<1>("stablehlo_minimum_version",
                              mlir::vhlo::Version::getMinimumVersion()),
      });
  return *c_values;
}

namespace {

std::string StructSizeErrorMsg(std::string_view struct_name,
                               size_t expected_size, size_t actual_size) {
  std::string error_msg = absl::StrCat(
      "Unexpected ", struct_name, " size: expected ", expected_size, ", got ",
      actual_size,
      ". The plugin is likely built with a later version than the framework.");
#if defined(PJRT_API_MAJOR)
  absl::StrAppend(&error_msg, " This plugin is built with PJRT API version ",
                  PJRT_API_MAJOR, ".", PJRT_API_MINOR, ".");
#endif  // PJRT_API_MAJOR
  return error_msg;
}

}  // namespace

absl::Status ActualStructSizeIsGreaterOrEqual(std::string_view struct_name,
                                              size_t expected_size,
                                              size_t actual_size) {
  if (actual_size < expected_size) {
    return absl::InvalidArgumentError(
        StructSizeErrorMsg(struct_name, expected_size, actual_size));
  }
  if (actual_size > expected_size) {
    VLOG(2) << StructSizeErrorMsg(struct_name, expected_size, actual_size);
  }
  return absl::OkStatus();
}

std::string_view GetPlatformVersion(PJRT_Client* client, const PJRT_Api* api) {
  PJRT_Client_PlatformVersion_Args args;
  args.struct_size = PJRT_Client_PlatformVersion_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = client;
  LogFatalIfPjrtError(api->PJRT_Client_PlatformVersion(&args), api);

  std::string_view platform_version(args.platform_version,
                                    args.platform_version_size);
  return platform_version;
}

std::string_view GetPlatformName(PJRT_Client* client, const PJRT_Api* api) {
  PJRT_Client_PlatformName_Args args;
  args.client = client;
  args.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  pjrt::LogFatalIfPjrtError(api->PJRT_Client_PlatformName(&args), api);

  std::string_view platform_name(args.platform_name, args.platform_name_size);
  return platform_name;
}

absl::StatusOr<PJRT_TopologyDescription*> GetTopologyDescription(
    PJRT_Client* client, const PJRT_Api* api) {
  PJRT_Client_TopologyDescription_Args args;
  args.struct_size = PJRT_Client_TopologyDescription_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.client = client;
  RETURN_STATUS_IF_PJRT_ERROR(api->PJRT_Client_TopologyDescription(&args), api);
  return args.topology;
}

PJRT_Chunk ConvertFromCppChunk(zkx::PjRtChunk chunk) {
  // `deleter_arg` holds a copy of the original zkx::PjRtChunk
  // deleter. The original zkx::PjRtChunk `input` releases its ownership
  // of data, which will subsequently be managed by `deleter` along with
  // `deleter_arg`.
  PJRT_Chunk c_chunk;
  c_chunk.data = chunk.data();
  c_chunk.size = static_cast<size_t>(chunk.size());
  c_chunk.deleter_arg = new std::function(chunk.deleter());
  c_chunk.deleter = [](void* data, void* deleter_arg) {
    auto* deleter = reinterpret_cast<std::function<void(void*)>*>(deleter_arg);
    (*deleter)(data);
    delete deleter;
  };

  // Release the ownership of `chunk.data()`, so it can be managed by `c_chunk`.
  chunk.release();

  return c_chunk;
}

zkx::PjRtChunk ConvertToCppChunk(const PJRT_Chunk& chunk) {
  return zkx::PjRtChunk(
      chunk.data, chunk.size,
      [deleter_arg = chunk.deleter_arg, deleter = chunk.deleter](void* data) {
        deleter(data, deleter_arg);
      });
}

PJRT_DeviceDescription* GetDeviceDescription(const PJRT_Api* api,
                                             PJRT_Device* device) {
  PJRT_Device_GetDescription_Args args;
  args.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.device = device;
  pjrt::LogFatalIfPjrtError(api->PJRT_Device_GetDescription(&args), api);
  return args.device_description;
}

absl::Span<PJRT_Memory* const> GetAddressableMemories(const PJRT_Api* api,
                                                      PJRT_Device* device) {
  PJRT_Device_AddressableMemories_Args args;
  args.struct_size = PJRT_Device_AddressableMemories_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.device = device;
  pjrt::LogFatalIfPjrtError(api->PJRT_Device_AddressableMemories(&args), api);
  return absl::MakeSpan(args.memories, args.num_memories);
}

int GetId(const PJRT_Api* api, PJRT_DeviceDescription* device_desc) {
  PJRT_DeviceDescription_Id_Args args = PJRT_DeviceDescription_Id_Args{
      PJRT_DeviceDescription_Id_Args_STRUCT_SIZE, nullptr, device_desc};
  pjrt::LogFatalIfPjrtError(api->PJRT_DeviceDescription_Id(&args), api);
  return args.id;
}

namespace {

void PjRtValueDeleterCallback(char* value) { delete[] value; }

PJRT_KeyValueGetCFunc ToKVGetCFunc(zkx::KeyValueStoreInterface* kv_store) {
  return [kv_store](PJRT_KeyValueGetCallback_Args* args) -> PJRT_Error* {
    absl::StatusOr<std::string> output =
        kv_store->Get(std::string_view(args->key, args->key_size),
                      absl::Milliseconds(args->timeout_in_ms));
    if (!output.ok()) {
      std::string_view message = output.status().message();
      return (*args->callback_error)(
          StatusCodeToPjrtErrorCode(output.status().code()), message.data(),
          message.size());
    }
    args->value = new char[output->size()];
    std::copy(output->begin(), output->end(), args->value);
    args->value_size = output->size();
    args->value_deleter_callback = &PjRtValueDeleterCallback;
    return nullptr;
  };
}

PJRT_KeyValueTryGetCFunc ToKVTryGetCFunc(
    zkx::KeyValueStoreInterface* kv_store) {
  return [kv_store](PJRT_KeyValueTryGetCallback_Args* args) -> PJRT_Error* {
    absl::StatusOr<std::string> output =
        kv_store->TryGet(std::string_view(args->key, args->key_size));
    if (!output.ok()) {
      std::string_view message = output.status().message();
      return (*args->callback_error)(
          StatusCodeToPjrtErrorCode(output.status().code()), message.data(),
          message.size());
    }
    args->value = new char[output->size()];
    std::copy(output->begin(), output->end(), args->value);
    args->value_size = output->size();
    args->value_deleter_callback = &PjRtValueDeleterCallback;
    return nullptr;
  };
}

PJRT_KeyValuePutCFunc ToKVPutCFunc(zkx::KeyValueStoreInterface* kv_store) {
  return [kv_store](PJRT_KeyValuePutCallback_Args* args) -> PJRT_Error* {
    absl::Status status =
        kv_store->Set(std::string_view(args->key, args->key_size),
                      std::string_view(args->value, args->value_size));
    if (!status.ok()) {
      std::string_view message = status.message();
      return (*args->callback_error)(StatusCodeToPjrtErrorCode(status.code()),
                                     message.data(), message.size());
    }
    return nullptr;
  };
}

PJRT_KeyValueGetCallback ToCKVGetCallback(
    PJRT_KeyValueGetCFunc* kv_get_c_func) {
  return [](PJRT_KeyValueGetCallback_Args* args) -> PJRT_Error* {
    PJRT_KeyValueGetCFunc* kv_get_c_func =
        reinterpret_cast<PJRT_KeyValueGetCFunc*>(args->user_arg);
    if (kv_get_c_func == nullptr) {
      absl::Status status = absl::InvalidArgumentError(
          "got nullptr for PJRT_KeyValueGet_Args.user_arg");
      return (*args->callback_error)(StatusCodeToPjrtErrorCode(status.code()),
                                     status.message().data(),
                                     status.message().size());
    }
    return (*kv_get_c_func)(args);
  };
}

PJRT_KeyValueTryGetCallback ToCKVTryGetCallback(
    PJRT_KeyValueTryGetCFunc* kv_try_get_c_func) {
  return [](PJRT_KeyValueTryGetCallback_Args* args) -> PJRT_Error* {
    PJRT_KeyValueTryGetCFunc* kv_try_get_c_func =
        reinterpret_cast<PJRT_KeyValueTryGetCFunc*>(args->user_arg);
    if (kv_try_get_c_func == nullptr) {
      absl::Status status = absl::InvalidArgumentError(
          "got nullptr for PJRT_KeyValueTryGet_Args.user_arg");
      return (*args->callback_error)(StatusCodeToPjrtErrorCode(status.code()),
                                     status.message().data(),
                                     status.message().size());
    }
    return (*kv_try_get_c_func)(args);
  };
}

PJRT_KeyValuePutCallback ToCKVPutCallback(
    PJRT_KeyValuePutCFunc* kv_put_c_func) {
  return [](PJRT_KeyValuePutCallback_Args* args) -> PJRT_Error* {
    PJRT_KeyValuePutCFunc* kv_put_c_func =
        reinterpret_cast<PJRT_KeyValuePutCFunc*>(args->user_arg);
    if (kv_put_c_func == nullptr) {
      absl::Status status = absl::InvalidArgumentError(
          "got nullptr for PJRT_KeyValuePut_Args.user_arg");
      return (*args->callback_error)(StatusCodeToPjrtErrorCode(status.code()),
                                     status.message().data(),
                                     status.message().size());
    }
    return (*kv_put_c_func)(args);
  };
}

}  // namespace

std::unique_ptr<PJRT_KeyValueCallbackData> ConvertToCKeyValueCallbacks(
    std::shared_ptr<zkx::KeyValueStoreInterface> kv_store) {
  auto kv_callback_data = std::make_unique<PJRT_KeyValueCallbackData>();
  kv_callback_data->kv_get_c_func = ToKVGetCFunc(kv_store.get());
  kv_callback_data->kv_try_get_c_func = ToKVTryGetCFunc(kv_store.get());
  kv_callback_data->kv_put_c_func = ToKVPutCFunc(kv_store.get());
  kv_callback_data->c_kv_get =
      ToCKVGetCallback(&kv_callback_data->kv_get_c_func);
  kv_callback_data->c_kv_try_get =
      ToCKVTryGetCallback(&kv_callback_data->kv_try_get_c_func);
  kv_callback_data->c_kv_put =
      ToCKVPutCallback(&kv_callback_data->kv_put_c_func);
  kv_callback_data->kv_store = std::move(kv_store);
  return kv_callback_data;
}

PJRT_SendCallbackInfo CppSendCallbackToCSendCallback(
    zkx::SendCallback cpp_send_callback,
    PJRT_SendCallbackFunction* send_callback_function) {
  return PJRT_SendCallbackInfo{
      cpp_send_callback.channel_id,
      // this is the void* user_arg to capture `cpp_send_callback.callback`
      send_callback_function,
      // this is the function pointer, PJRT_SendCallback
      [](PJRT_Chunk* chunk, PJRT_CallbackError* callback_error,
         size_t total_size_in_bytes, bool done, void* user_arg) -> PJRT_Error* {
        // PJRT_SendCallback, `send_callback` is internal C interface callback
        // representation that captures the client C++ callback in void*
        // `user_arg` and reinterprets in the lower-level runtime for execution.
        // `user_arg` captures `send_callback_function` which is
        // SendCallbackFunction*.
        PJRT_SendCallbackFunction* send_callback =
            reinterpret_cast<PJRT_SendCallbackFunction*>(user_arg);
        return (*send_callback)(chunk, callback_error, total_size_in_bytes,
                                done);
      }};
}

PJRT_RecvCallbackInfo CppRecvCallbackToCRecvCallback(
    zkx::RecvCallback cpp_recv_callback,
    PJRT_RecvCallbackFunction* recv_callback_function) {
  return PJRT_RecvCallbackInfo{
      cpp_recv_callback.channel_id,
      // this is the void* user_arg to capture `cpp_recv_callback.callback`
      recv_callback_function,
      // this is the function pointer, PJRT_RecvCallback
      [](PJRT_CopyToDeviceStream* stream, void* user_arg) {
        // PJRT_RecvCallback, `recv_callback` is internal C interface callback
        // representation that captures the client C++ callback in void*
        // `user_arg` and reinterprets in the lower-level runtime for execution.
        // `user_arg` captures `recv_callback_function` which is
        // RecvCallbackFunction*.
        auto* recv_callback =
            reinterpret_cast<std::function<void(PJRT_CopyToDeviceStream*)>*>(
                user_arg);
        (*recv_callback)(stream);
      }};
}

absl::StatusOr<BufferMemoryLayoutData> ConvertToBufferMemoryLayoutData(
    const zkx::Layout& cpp_layout) {
  BufferMemoryLayoutData layout_data;
  layout_data.c_layout.type =
      PJRT_Buffer_MemoryLayout_Type::PJRT_Buffer_MemoryLayout_Type_Tiled;

  PJRT_Buffer_MemoryLayout_Tiled c_tiled;
  layout_data.minor_to_major.assign(cpp_layout.minor_to_major().begin(),
                                    cpp_layout.minor_to_major().end());
  c_tiled.minor_to_major = layout_data.minor_to_major.data();
  c_tiled.minor_to_major_size = layout_data.minor_to_major.size();
  c_tiled.num_tiles = cpp_layout.tiles().size();
  if (c_tiled.num_tiles >= 0) {
    layout_data.tile_dim_sizes.reserve(c_tiled.num_tiles);
    for (int i = 0; i < c_tiled.num_tiles; ++i) {
      absl::Span<const int64_t> tile_dim = cpp_layout.tiles()[i].dimensions();
      layout_data.tile_dims.insert(layout_data.tile_dims.end(),
                                   tile_dim.begin(), tile_dim.end());
      layout_data.tile_dim_sizes.push_back(tile_dim.size());
    }
    c_tiled.tile_dims = layout_data.tile_dims.data();
    c_tiled.tile_dim_sizes = layout_data.tile_dim_sizes.data();
  }
  layout_data.c_layout.tiled = c_tiled;
  return layout_data;
}

absl::StatusOr<BufferMemoryLayoutData> ConvertToBufferMemoryLayoutData(
    absl::Span<int64_t const> byte_strides) {
  BufferMemoryLayoutData layout_data;
  layout_data.c_layout.type =
      PJRT_Buffer_MemoryLayout_Type::PJRT_Buffer_MemoryLayout_Type_Strides;
  layout_data.c_layout.strides.byte_strides = byte_strides.data();
  layout_data.c_layout.strides.num_byte_strides = byte_strides.size();
  return layout_data;
}

absl::StatusOr<zkx::Layout> ConvertToLayout(
    const PJRT_Buffer_MemoryLayout_Tiled& c_tiled) {
  absl::Span<const int64_t> minor_to_major(c_tiled.minor_to_major,
                                           c_tiled.minor_to_major_size);
  absl::InlinedVector<zkx::Tile, 1> tiles;
  tiles.reserve(c_tiled.num_tiles);
  const int64_t* current_tile = c_tiled.tile_dims;
  for (int i = 0; i < c_tiled.num_tiles; ++i) {
    tiles.push_back(zkx::Tile(
        absl::Span<const int64_t>(current_tile, c_tiled.tile_dim_sizes[i])));
    current_tile += c_tiled.tile_dim_sizes[i];
  }
  zkx::Layout layout = zkx::Layout(minor_to_major);
  layout.mutable_tiles()->assign(tiles.begin(), tiles.end());
  return layout;
}

PJRT_Buffer_Type GetElementType(const PJRT_Api* api, PJRT_Buffer* buffer) {
  PJRT_Buffer_ElementType_Args args;
  args.struct_size = PJRT_Buffer_ElementType_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer;
  LogFatalIfPjrtError(api->PJRT_Buffer_ElementType(&args), api);
  return args.type;
}

absl::Span<const int64_t> GetDimensions(const PJRT_Api* api,
                                        PJRT_Buffer* buffer) {
  PJRT_Buffer_Dimensions_Args args;
  args.struct_size = PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer;
  LogFatalIfPjrtError(api->PJRT_Buffer_Dimensions(&args), api);
  return {args.dims, args.num_dims};
}

std::unique_ptr<PJRT_Layouts_MemoryLayout, PJRT_Layouts_MemoryLayoutDeleter>
GetMemoryLayout(const PJRT_Api* api, PJRT_Buffer* buffer) {
  PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args args;
  args.struct_size = PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer;
  PJRT_Layouts_Extension* ext_api =
      FindExtension<PJRT_Layouts_Extension>(api, PJRT_Extension_Type_Layouts);
  CHECK_NE(ext_api, nullptr) << "GetMemoryLayout called with PJRT_Api that "
                                "doesn't support layouts extension";
  LogFatalIfPjrtError(ext_api->PJRT_Layouts_PJRT_Buffer_MemoryLayout(&args),
                      api);
  return std::unique_ptr<PJRT_Layouts_MemoryLayout,
                         PJRT_Layouts_MemoryLayoutDeleter>(
      args.layout, MakeMemoryLayoutDeleter(api));
}

absl::StatusOr<zkx::Shape> BuildZkxShapeFromC(
    PJRT_Buffer_Type element_type, const int64_t* dims, size_t num_dims,
    PJRT_Buffer_MemoryLayout* layout) {
  zkx::Shape shape =
      zkx::ShapeUtil::MakeShape(ConvertFromPjRtBufferType(element_type),
                                absl::Span<const int64_t>(dims, num_dims));
  zkx::Layout cpp_layout;
  if (layout != nullptr) {
    switch (layout->type) {
      case PJRT_Buffer_MemoryLayout_Type::PJRT_Buffer_MemoryLayout_Type_Tiled: {
        TF_ASSIGN_OR_RETURN(cpp_layout, ConvertToLayout(layout->tiled));
        break;
      }
      case PJRT_Buffer_MemoryLayout_Type::
          PJRT_Buffer_MemoryLayout_Type_Strides: {
        TF_RETURN_IF_ERROR(absl::InvalidArgumentError(
            "PJRT_Buffer_MemoryLayout_Type_Strides is not supported to be "
            "converted to a zkx::Shape"));
        break;
      }
      default: {
        TF_RETURN_IF_ERROR(absl::InvalidArgumentError(absl::StrCat(
            "Unexpected PJRT_Buffer_MemoryLayout_Type type: ", layout->type)));
      }
    }
    *shape.mutable_layout() = cpp_layout;
  }
  return shape;
}

std::string_view PlatformName(const PJRT_Api* api,
                              const PJRT_TopologyDescription* topo_desc) {
  PJRT_TopologyDescription_PlatformName_Args args;
  args.struct_size = PJRT_TopologyDescription_PlatformName_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.topology = const_cast<PJRT_TopologyDescription*>(topo_desc);
  LogFatalIfPjrtError(api->PJRT_TopologyDescription_PlatformName(&args), api);
  return {args.platform_name, args.platform_name_size};
}

absl::Span<PJRT_DeviceDescription* const> DeviceDescriptions(
    const PJRT_Api* api, const PJRT_TopologyDescription* topo_desc) {
  PJRT_TopologyDescription_GetDeviceDescriptions_Args args;
  args.struct_size =
      PJRT_TopologyDescription_GetDeviceDescriptions_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.topology = const_cast<PJRT_TopologyDescription*>(topo_desc);
  LogFatalIfPjrtError(
      api->PJRT_TopologyDescription_GetDeviceDescriptions(&args), api);
  return {args.descriptions, args.num_descriptions};
}

absl::StatusOr<zkx::CompiledMemoryStats> GetCompiledMemoryStats(
    const PJRT_Api* api, PJRT_Executable* executable) {
  PJRT_Executable_GetCompiledMemoryStats_Args args;
  args.struct_size = PJRT_Executable_GetCompiledMemoryStats_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.executable = executable;
  RETURN_STATUS_IF_PJRT_ERROR(
      api->PJRT_Executable_GetCompiledMemoryStats(&args), api);
  zkx::CompiledMemoryStats results;
  results.generated_code_size_in_bytes = args.generated_code_size_in_bytes;
  results.argument_size_in_bytes = args.argument_size_in_bytes;
  results.output_size_in_bytes = args.output_size_in_bytes;
  results.alias_size_in_bytes = args.alias_size_in_bytes;
  results.temp_size_in_bytes = args.temp_size_in_bytes;
  results.host_generated_code_size_in_bytes =
      args.host_generated_code_size_in_bytes;
  results.host_argument_size_in_bytes = args.host_argument_size_in_bytes;
  results.host_output_size_in_bytes = args.host_output_size_in_bytes;
  results.host_alias_size_in_bytes = args.host_alias_size_in_bytes;
  results.host_temp_size_in_bytes = args.host_temp_size_in_bytes;
  return results;
}

PJRT_Profiler_Extension CreatePjrtProfilerExtension(
    std::string_view traceme_name) {
  tsl::profiler::TraceMeProducer producer(
      traceme_name, tsl::profiler::ContextType::kPjrtLibraryCall);
  int64_t traceme_context_id = producer.GetContextId();
  PJRT_Profiler_Extension profiler_extension{
      /*struct_size=*/PJRT_Profiler_Extension_STRUCT_SIZE,
      /*type=*/PJRT_Extension_Type::PJRT_Extension_Type_Profiler,
      /*next=*/nullptr,
      /*profiler_api=*/nullptr,
      /*traceme_context_id=*/traceme_context_id,
  };
  return profiler_extension;
}

PJRT_ShapeSpec ConvertToPjRtShapeSpec(
    const zkx::PjRtClient::ShapeSpec& shape_spec) {
  PJRT_ShapeSpec c_shape_spec;
  c_shape_spec.struct_size = PJRT_ShapeSpec_STRUCT_SIZE;
  c_shape_spec.extension_start = nullptr;
  c_shape_spec.element_type =
      pjrt::ConvertToPjRtBufferType(shape_spec.element_type);
  c_shape_spec.dims = shape_spec.dims.data();
  c_shape_spec.num_dims = shape_spec.dims.size();
  return c_shape_spec;
}

zkx::PjRtClient::ShapeSpec ConvertFromPjrtShapeSpec(
    PJRT_ShapeSpec c_shape_spec) {
  zkx::PjRtClient::ShapeSpec shape_spec;
  shape_spec.element_type =
      pjrt::ConvertFromPjRtBufferType(c_shape_spec.element_type);

  shape_spec.dims = zkx::DimensionVector(
      c_shape_spec.dims, c_shape_spec.dims + c_shape_spec.num_dims);
  return shape_spec;
}

std::vector<zkx::PjRtMemorySpaceDescription> GetMemorySpaceDescriptions(
    PJRT_DeviceDescription* device_description, const PJRT_Api* c_api,
    absl::StatusOr<zkx::PjRtMemorySpaceDescription*>* default_memory) {
  const PJRT_MemoryDescriptions_Extension* extension =
      pjrt::FindExtension<PJRT_MemoryDescriptions_Extension>(
          c_api, PJRT_Extension_Type::PJRT_Extension_Type_MemoryDescriptions);
  if (!extension) return {};

  PJRT_DeviceDescription_MemoryDescriptions_Args mem_desc_args;
  mem_desc_args.struct_size =
      PJRT_DeviceDescription_MemoryDescriptions_Args_STRUCT_SIZE;
  mem_desc_args.extension_start = nullptr;
  mem_desc_args.device_description = device_description;
  pjrt::LogFatalIfPjrtError(
      extension->PJRT_DeviceDescription_MemoryDescriptions(&mem_desc_args),
      c_api);

  std::vector<zkx::PjRtMemorySpaceDescription> memory_space_descriptions;
  for (int i = 0; i < mem_desc_args.num_memory_descriptions; i++) {
    PJRT_MemoryDescription_Kind_Args kind_args;
    kind_args.struct_size = PJRT_MemoryDescription_Kind_Args_STRUCT_SIZE;
    kind_args.extension_start = nullptr;
    kind_args.memory_description = mem_desc_args.memory_descriptions[i];
    pjrt::LogFatalIfPjrtError(
        extension->PJRT_MemoryDescription_Kind(&kind_args), c_api);
    zkx::PjRtMemorySpaceDescription description(
        std::string(kind_args.kind, kind_args.kind_size), kind_args.kind_id);
    memory_space_descriptions.push_back(description);
  }
  if (default_memory) {
    *default_memory = {};
    for (int i = 0; i < mem_desc_args.num_memory_descriptions; i++) {
      if (mem_desc_args.default_memory_index == i) {
        *default_memory = &memory_space_descriptions[i];
      }
    }
  }
  return memory_space_descriptions;
}
PJRT_Error* InvokePjRtEventWhenReady(
    const PJRT_Api* api, PJRT_Event* event,
    absl::AnyInvocable<void() &&> on_done_with_event) {
  if (on_done_with_event) {
    PJRT_Event_OnReady_Args event_args;
    event_args.struct_size = PJRT_Event_OnReady_Args_STRUCT_SIZE;
    event_args.extension_start = nullptr;
    event_args.event = event;
    event_args.user_arg = new absl::AnyInvocable<void(PJRT_Error*)>(
        [on_done_with_event = std::move(on_done_with_event),
         c_api = api](PJRT_Error* error) mutable {
          if (error) {
            ::pjrt::MakeErrorDeleter(c_api)(error);
          }
          std::move(on_done_with_event)();
        });
    event_args.callback = [](PJRT_Error* error, void* args) {
      auto* on_done_with_event =
          reinterpret_cast<absl::AnyInvocable<void(PJRT_Error*)>*>(args);
      (*on_done_with_event)(error);
      delete on_done_with_event;
    };
    return api->PJRT_Event_OnReady(&event_args);
  }
  return nullptr;
}

}  // namespace pjrt
