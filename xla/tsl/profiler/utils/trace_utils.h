/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

#ifndef XLA_TSL_PROFILER_UTILS_TRACE_UTILS_H_
#define XLA_TSL_PROFILER_UTILS_TRACE_UTILS_H_

#include <cstdint>
#include <optional>
#include <string_view>

#include "absl/strings/numbers.h"

namespace tsl::profiler {

// Constants used as trace_viewer PID (device_id in trace_events.proto).
// PID 0 is unused.
// Support up to 500 accelerator devices.
constexpr uint32_t kFirstDeviceId = 1;
constexpr uint32_t kLastDeviceId = 500;
// Support Upto 200 custom planes as fake devices (i.e., planes with a
// "/custom:" prefix). See `<project_name>::kCustomPlanePrefix` for more
// information
constexpr uint32_t kFirstCustomPlaneDeviceId = kLastDeviceId + 1;
constexpr uint32_t kMaxCustomPlaneDevicesPerHost = 200;
constexpr uint32_t kLastCustomPlaneDeviceId =
    kFirstCustomPlaneDeviceId + kMaxCustomPlaneDevicesPerHost - 1;
// Host threads are shown as a single fake device.
constexpr uint32_t kHostThreadsDeviceId = kLastCustomPlaneDeviceId + 1;

// Constants used as trace_viewer TID (resource_id in trace_events.proto).
constexpr int kThreadIdDerivedMin = 0xdeadbeef;
constexpr int kThreadIdStepInfo = kThreadIdDerivedMin;
constexpr int kThreadIdKernelLaunch = kThreadIdDerivedMin + 1;
constexpr int kThreadIdTfNameScope = kThreadIdDerivedMin + 2;
constexpr int kThreadIdTfOp = kThreadIdDerivedMin + 3;
constexpr int kThreadIdHloModule = kThreadIdDerivedMin + 4;
constexpr int kThreadIdHloOp = kThreadIdDerivedMin + 5;
constexpr int kThreadIdOverhead = kThreadIdDerivedMin + 6;
constexpr int kThreadIdSource = kThreadIdDerivedMin + 7;
constexpr int kThreadIdHostOffloadOpStart = kThreadIdDerivedMin + 8;
constexpr int kThreadIdHostOffloadOpEnd = kThreadIdDerivedMin + 48;
// Space for derived lines for host ZKX Ops
constexpr int kThreadIdHostZkxRegionStart = kThreadIdDerivedMin + 49;
constexpr int kThreadIdHostZkxRegionEnd = kThreadIdHostZkxRegionStart + 240;
constexpr int kThreadIdDerivedMax = kThreadIdHostZkxRegionEnd;

static inline bool IsDerivedThreadId(int thread_id) {
  return thread_id >= kThreadIdDerivedMin && thread_id <= kThreadIdDerivedMax;
}

// Parses the device ordinal (N) from device names that use TensorFlow
// convention: "hostname /device:xPU:N".
static inline std::optional<uint32_t> ParseDeviceOrdinal(
    std::string_view device_name) {
  if (auto pos = device_name.find_last_of(':'); pos != std::string_view::npos) {
    device_name.remove_prefix(pos + 1);
  }
  if (auto pos = device_name.find_first_of(' ');
      pos != std::string_view::npos) {
    device_name.remove_suffix(device_name.size() - pos);
  }
  uint32_t device_id;
  if (absl::SimpleAtoi(device_name, &device_id)) return device_id;
  return std::nullopt;
}

}  // namespace tsl::profiler

#endif  // XLA_TSL_PROFILER_UTILS_TRACE_UTILS_H_
