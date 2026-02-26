/* Copyright 2020 The OpenXLA Authors.
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

#ifndef ZKX_PJRT_GPU_GPU_HELPERS_H_
#define ZKX_PJRT_GPU_GPU_HELPERS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

#include "xla/tsl/framework/bfc_allocator.h"
#include "zkx/client/local_client.h"
#include "zkx/stream_executor/stream_executor.h"

namespace zkx {

// Builds an zkx::LocalClient for the GPU platform.
absl::StatusOr<LocalClient*> GetGpuZkxClient(
    const std::optional<std::string>& platform_name,
    const std::optional<std::set<int>>& allowed_devices);

// Enables peer access between all pairs of GPUs where possible.
void EnablePeerAccess(absl::Span<se::StreamExecutor* const> executors);

// Returns a GPU pinned host memory allocator to use when staging host->GPU
// transfers. Uses a fixed pool of pinned memory.
//
// The pool size is controlled by ZKX_PJRT_GPU_HOST_MEMORY_LIMIT_GB environment
// variable, which defaults to 64GB.
//
// If ZKX_PJRT_GPU_HOST_MEMORY_PREALLOCATE is set to true, the pool will be
// preallocated, and the preallocated size is controlled by
// ZKX_PJRT_GPU_HOST_MEMORY_LIMIT_GB environment variable, which defaults to
// 16GB in this case.
absl::StatusOr<std::unique_ptr<tsl::BFCAllocator>> GetGpuHostAllocator(
    se::StreamExecutor* executor);

// Builds a BFCAllocator for all local GPUs.
absl::StatusOr<std::unique_ptr<tsl::BFCAllocator>> CreateBFCAllocator(
    se::StreamExecutor* executor, double memory_fraction, bool preallocate,
    std::optional<int64_t> gpu_system_memory_size);

// Builds a BFCAllocator for all local GPUs that uses collective memory.
absl::StatusOr<std::unique_ptr<tsl::BFCAllocator>> CreateCollectiveBFCAllocator(
    se::StreamExecutor* executor, double memory_fraction,
    size_t collective_memory_size);

// Represents topology of devices.
struct TopologySizes {
  int num_slices = 0;
  int num_hosts_per_slice = 0;
  int num_devices_per_host = 0;

  // Returns number of devices in the topology.
  int GetDeviceCount();
  // Parses the topology description of the form
  // "<num_slices> x <num_hosts_per_slice> x <num_devices_per_host>"
  // and returns the parsed components on success.
  static absl::StatusOr<TopologySizes> FromString(
      std::string_view topology_string);
};

}  // namespace zkx

#endif  // ZKX_PJRT_GPU_GPU_HELPERS_H_
