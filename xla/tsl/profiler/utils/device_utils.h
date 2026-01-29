/* Copyright 2024 The TensorFlow Authors All Rights Reserved.
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

#ifndef XLA_TSL_PROFILER_UTILS_DEVICE_UTILS_H_
#define XLA_TSL_PROFILER_UTILS_DEVICE_UTILS_H_

#include "xla/tsl/profiler/protobuf/xplane.pb.h"

namespace tsl::profiler {

using ::tensorflow::profiler::XPlane;

enum class DeviceType {
  kUnknown,
  kCpu,
  kTpu,
  kGpu,
};

// Gets DeviceType from XPlane.
DeviceType GetDeviceType(const XPlane& plane);
// Gets DeviceType from XPlane name.
DeviceType GetDeviceType(std::string_view plane_name);

}  // namespace tsl::profiler

#endif  // XLA_TSL_PROFILER_UTILS_DEVICE_UTILS_H_
