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

#include "zkx/pjrt/c/pjrt_c_api_cpu_internal.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"

#include "zkx/pjrt/c/pjrt_c_api_ffi_extension.h"
#include "zkx/pjrt/c/pjrt_c_api_ffi_internal.h"
#include "zkx/pjrt/c/pjrt_c_api_helpers.h"
#include "zkx/pjrt/c/pjrt_c_api_layouts_extension.h"
#include "zkx/pjrt/c/pjrt_c_api_memory_descriptions_extension.h"
#include "zkx/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "zkx/pjrt/cpu/cpu_client.h"
#include "zkx/pjrt/pjrt_client.h"
#include "zkx/pjrt/pjrt_executable.h"
#include "zkx/pjrt/plugin/cpu/cpu_client_options.h"

namespace pjrt {
namespace cpu_plugin {

PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_Create_Args", PJRT_Client_Create_Args_STRUCT_SIZE,
      args->struct_size));

  // TODO(b/263170683): cpu_device_count should be configurable after config
  // options can be passed to PJRT_Client_Create.
  zkx::CpuClientOptions options;
  options.cpu_device_count = 4;
  // ZKX: Changed - use zkx::GetTfrtCpuClient instead of
  // xla::GetXlaPjrtCpuClient
  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<zkx::PjRtClient> client,
                        zkx::GetTfrtCpuClient(std::move(options)));
  args->client = pjrt::CreateWrapperClient(std::move(client));
  return nullptr;
}

PJRT_Error* PJRT_ExecuteContext_Create(PJRT_ExecuteContext_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_ExecuteContext_Create_Args",
      PJRT_ExecuteContext_Create_Args_STRUCT_SIZE, args->struct_size));
  auto execute_context = std::make_unique<zkx::ExecuteContext>();
  args->context = pjrt::CreateWrapperExecuteContext(std::move(execute_context));
  return nullptr;
}

PJRT_Error* PJRT_CpuDeviceTopology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  return new PJRT_Error{
      absl::UnimplementedError("Topology not supported for CPU compilation.")};
}

const PJRT_Api* GetCpuPjrtApi() {
  static PJRT_Layouts_Extension layouts_extension =
      pjrt::CreateLayoutsExtension(nullptr);

  static PJRT_MemoryDescriptions_Extension memory_descriptions_extension =
      pjrt::CreateMemoryDescriptionsExtension(
          reinterpret_cast<PJRT_Extension_Base*>(&layouts_extension));

  static PJRT_FFI_Extension ffi_extension = pjrt::CreateFfiExtension(
      reinterpret_cast<PJRT_Extension_Base*>(&memory_descriptions_extension));

  static const PJRT_Api pjrt_api = pjrt::CreatePjrtApi(
      pjrt::cpu_plugin::PJRT_Client_Create,
      pjrt::cpu_plugin::PJRT_ExecuteContext_Create,
      pjrt::cpu_plugin::PJRT_CpuDeviceTopology_Create,
      pjrt::PJRT_Plugin_Initialize_NoOp,
      reinterpret_cast<PJRT_Extension_Base*>(&ffi_extension),
      pjrt::PJRT_Plugin_Attributes_Zkx);

  return &pjrt_api;
}

}  // namespace cpu_plugin
}  // namespace pjrt
