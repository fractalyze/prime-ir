/* Copyright 2023 The OpenXLA Authors.
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

#include "zkx/python/pjrt_ifrt/zkx_compiler.h"

#include <string>

#include "absl/status/status.h"
#include "llvm/Support/Casting.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/python/ifrt/client.h"
#include "zkx/python/ifrt/device.h"
#include "zkx/python/ifrt/serdes.h"
#include "zkx/python/pjrt_ifrt/zkx_compiler.pb.h"
#include "zkx/service/computation_placer.h"

namespace zkx::ifrt {
namespace {

class ZkxCompileOptionsSerDes
    : public llvm::RTTIExtends<ZkxCompileOptionsSerDes, SerDes> {
 public:
  std::string_view type_name() const override {
    return "zkx::ifrt::ZkxCompileOptions";
  }

  absl::StatusOr<std::string> Serialize(
      const Serializable& serializable,
      std::unique_ptr<SerializeOptions>) override {
    const auto& options = llvm::cast<ZkxCompileOptions>(serializable);

    ZkxCompileOptionsProto proto;
    TF_ASSIGN_OR_RETURN(*proto.mutable_compile_options(),
                        options.compile_options.ToProto());
    if (!options.loaded_host_callbacks.empty()) {
      return absl::UnimplementedError(
          "zkx::ifrt::ZkxCompileOptions with loaded_host_callbacks is not "
          "serializable");
    }
    return proto.SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions>) override {
    ZkxCompileOptionsProto proto;
    if (!proto.ParseFromString(serialized)) {
      return absl::DataLossError(
          "Unable to parse serialized ZkxCompileOptionsProto");
    }

    auto options = std::make_unique<ZkxCompileOptions>();
    TF_ASSIGN_OR_RETURN(
        options->compile_options,
        zkx::CompileOptions::FromProto(proto.compile_options()));
    return options;
  }

  static char ID;
};

char ZkxCompileOptionsSerDes::ID = 0;

bool register_zkx_compile_options_serdes = ([]{
  RegisterSerDes<ZkxCompileOptions>(
      std::make_unique<ZkxCompileOptionsSerDes>());
}(), true);

}  // namespace

char ZkxCompileOptions::ID = 0;
char ZkxDeserializeExecutableOptions::ID = 0;

absl::StatusOr<std::unique_ptr<ZkxCompileOptions>> GetZkxCompileOptions(
    std::unique_ptr<CompileOptions> options) {
  if (!llvm::isa<ZkxCompileOptions>(options.get())) {
    return absl::InvalidArgumentError("options must be ZkxCompileOptions");
  }
  return std::unique_ptr<ZkxCompileOptions>(
      static_cast<ZkxCompileOptions*>(options.release()));
}

absl::StatusOr<std::unique_ptr<ZkxDeserializeExecutableOptions>>
GetZkxDeserializeExecutableOptions(
    std::unique_ptr<DeserializeExecutableOptions> options) {
  if (!llvm::isa<ZkxDeserializeExecutableOptions>(options.get())) {
    return absl::InvalidArgumentError(
        "options must be ZkxDeserializeExecutableOptions");
  }
  return std::unique_ptr<ZkxDeserializeExecutableOptions>(
      static_cast<ZkxDeserializeExecutableOptions*>(options.release()));
}

absl::StatusOr<DeviceListRef> GetDeviceListFromDeviceAssignment(
    Client* ifrt_client, const zkx::DeviceAssignment& device_assignment) {
  std::vector<Device*> devices;
  devices.reserve(device_assignment.replica_count() *
                  device_assignment.computation_count());
  for (int64_t i = 0; i < device_assignment.replica_count(); ++i) {
    for (int64_t j = 0; j < device_assignment.computation_count(); ++j) {
      TF_ASSIGN_OR_RETURN(
          Device * device,
          ifrt_client->LookupDevice(DeviceId(device_assignment(i, j))));
      devices.push_back(device);
    }
  }
  return ifrt_client->MakeDeviceList(devices);
}

absl::StatusOr<DeviceListRef> GetDeviceListFromZkxCompileOptions(
    Client* ifrt_client, const zkx::CompileOptions& compile_options) {
  if (compile_options.executable_build_options.has_device_assignment()) {
    return GetDeviceListFromDeviceAssignment(
        ifrt_client,
        compile_options.executable_build_options.device_assignment());
  }
  if (compile_options.compile_portable_executable) {
    return ifrt_client->MakeDeviceList(
        {ifrt_client->addressable_devices().front()});
  }
  auto& build_options = compile_options.executable_build_options;
  if (build_options.device_ordinal() >= 0) {
    TF_ASSIGN_OR_RETURN(
        Device * device,
        ifrt_client->LookupDevice(DeviceId(build_options.device_ordinal())));
    return ifrt_client->MakeDeviceList({device});
  }
  TF_ASSIGN_OR_RETURN(
      zkx::DeviceAssignment default_da,
      ifrt_client->GetDefaultDeviceAssignment(build_options.num_replicas(),
                                              build_options.num_partitions()));
  TF_ASSIGN_OR_RETURN(DeviceListRef devices, GetDeviceListFromDeviceAssignment(
                                                 ifrt_client, default_da));
  return devices;
}

}  // namespace zkx::ifrt
