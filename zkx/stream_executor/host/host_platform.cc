/* Copyright 2016 The OpenXLA Authors.
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

#include "zkx/stream_executor/host/host_platform.h"

#include <thread>  // NOLINT(build/c++11)
#include <utility>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"

#include "zkx/stream_executor/host/host_executor.h"
#include "zkx/stream_executor/host/host_platform_id.h"
#include "zkx/stream_executor/platform/initialize.h"
#include "zkx/stream_executor/platform_manager.h"

namespace stream_executor::host {

HostPlatform::HostPlatform() : name_("Host") {}

HostPlatform::~HostPlatform() {}

Platform::Id HostPlatform::id() const { return kHostPlatformId; }

int HostPlatform::VisibleDeviceCount() const {
  return std::thread::hardware_concurrency();
}

std::string_view HostPlatform::Name() const { return name_; }

absl::StatusOr<std::unique_ptr<DeviceDescription>>
HostPlatform::DescriptionForDevice(int ordinal) const {
  return HostExecutor::CreateDeviceDescription(ordinal);
}

absl::StatusOr<StreamExecutor*> HostPlatform::ExecutorForDevice(int ordinal) {
  return executor_cache_.GetOrCreate(
      ordinal, [this, ordinal]() { return GetUncachedExecutor(ordinal); });
}

absl::StatusOr<std::unique_ptr<StreamExecutor>>
HostPlatform::GetUncachedExecutor(int ordinal) {
  auto executor = std::make_unique<HostExecutor>(this, ordinal);
  auto init_status = executor->Init();
  if (!init_status.ok()) {
    return absl::InternalError(absl::StrFormat(
        "failed initializing StreamExecutor for device ordinal %d: %s", ordinal,
        init_status.ToString().c_str()));
  }

  return std::move(executor);
}

namespace {

void InitializeHostPlatform() {
  std::unique_ptr<Platform> platform(new host::HostPlatform);
  CHECK_OK(PlatformManager::RegisterPlatform(std::move(platform)));
}

}  // namespace

}  // namespace stream_executor::host

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    host_platform, stream_executor::host::InitializeHostPlatform());
