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

#ifndef ZKX_PJRT_CPU_CPU_DEVICE_H_
#define ZKX_PJRT_CPU_CPU_DEVICE_H_

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

#include "zkx/literal.h"
#include "zkx/pjrt/pjrt_client.h"
#include "zkx/pjrt/pjrt_common.h"
#include "zkx/pjrt/pjrt_future.h"
#include "zkx/pjrt/plugin/cpu/cpu_device_description.h"
#include "zkx/pjrt/semaphore.h"

namespace zkx {

class TfrtCpuDevice final : public PjRtDevice {
 public:
  explicit TfrtCpuDevice(int process_id, int local_device_id,
                         int max_inflight_computations = 32);

  const CpuDeviceDescription& description() const override {
    return description_;
  }

  void SetClient(PjRtClient* client) {
    CHECK_EQ(client_, nullptr);
    client_ = client;
  }

  PjRtClient* client() const override { return client_; }

  bool IsAddressable() const override {
    return process_index() == client()->process_index();
  }

  PjRtLocalDeviceId local_device_id() const override {
    return PjRtLocalDeviceId(local_hardware_id().value());
  }

  PjRtLocalHardwareId local_hardware_id() const override {
    return PjRtLocalHardwareId(description_.local_hardware_id());
  }

  absl::Status TransferToInfeed(const LiteralSlice& literal) override;

  absl::Status TransferFromOutfeed(MutableBorrowingLiteral literal) override;

  void AttachMemorySpace(PjRtMemorySpace* memory_space);

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override {
    return memory_spaces_;
  }

  absl::StatusOr<PjRtMemorySpace*> default_memory_space() const override;

  absl::StatusOr<PjRtMemorySpace*> memory_space_by_kind(
      std::string_view memory_space_kind) const override;

  absl::StatusOr<PjRtMemorySpace*> memory_space_by_kind_id(int id) const;

  // Returns a semaphore for admission control on inflight computations.
  Semaphore& max_inflight_computations_semaphore() {
    return max_inflight_computations_semaphore_;
  }

  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      std::string_view description) const override {
    return nullptr;
  }

 private:
  PjRtClient* client_ = nullptr;  // not owned
  CpuDeviceDescription description_;
  absl::InlinedVector<PjRtMemorySpace*, 1> memory_spaces_;
  absl::flat_hash_map<int, PjRtMemorySpace*> memory_spaces_by_id_;

  // TODO(zhangqiaorjc): Optimize semaphore related overhead.
  // Semaphore used to limit how many programs can be enqueued by the host
  // ahead of the device.
  Semaphore max_inflight_computations_semaphore_;
};

}  // namespace zkx

#endif  // ZKX_PJRT_CPU_CPU_DEVICE_H_
