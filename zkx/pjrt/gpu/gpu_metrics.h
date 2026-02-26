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

#ifndef ZKX_PJRT_GPU_GPU_METRICS_H_
#define ZKX_PJRT_GPU_GPU_METRICS_H_

#include <cstdint>
#include <string_view>

namespace zkx {
namespace gpu_metrics {

inline constexpr std::string_view freeGpuSystemMemoryMetricName =
    "/pjrt/gpu/free_gpu_system_memory";

void RecordFreeGpuSystemMemory(int device_ordinal, int64_t free_memory);

int64_t GetFreeGpuSystemMemory(int gpu_id);

}  // namespace gpu_metrics
}  // namespace zkx

#endif  // ZKX_PJRT_GPU_GPU_METRICS_H_
