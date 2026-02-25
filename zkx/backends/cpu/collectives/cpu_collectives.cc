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

#include "zkx/backends/cpu/collectives/cpu_collectives.h"

#include "absl/log/check.h"
#include "absl/log/log.h"

#include "xla/tsl/platform/casts.h"
#include "zkx/core/collectives/collectives_registry.h"

namespace zkx::cpu {

// static
CpuCollectives* CpuCollectives::Default() {
  absl::StatusOr<Collectives*> collectives =
      CollectivesRegistry::Default("host");
  CHECK_OK(collectives) << "Failed to get CPU collectives";  // Crash OK

  if (auto* cpu_collectives = tsl::down_cast<CpuCollectives*>(*collectives)) {
    return cpu_collectives;
  }

  LOG(FATAL) << "Unsupported collectives implementation for CPU";
}

absl::StatusOr<const CpuCollectives::Device*> CpuCollectives::TryCast(
    const Collectives::Device* device) {
  if (auto* cpu_device = tsl::down_cast<const Device*>(device)) {
    return cpu_device;
  }
  return absl::InvalidArgumentError("Collectives device is not a CPU device");
}

absl::StatusOr<const CpuCollectives::Executor*> CpuCollectives::TryCast(
    const Communicator::Executor* executor) {
  if (auto* cpu_executor = tsl::down_cast<const Executor*>(executor)) {
    return cpu_executor;
  }
  return absl::InvalidArgumentError(
      "Collectives executor is not a CPU executor");
}

}  // namespace zkx::cpu
