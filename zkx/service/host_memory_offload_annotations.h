/* Copyright 2024 The OpenXLA Authors.
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

#ifndef ZKX_SERVICE_HOST_MEMORY_OFFLOAD_ANNOTATIONS_H_
#define ZKX_SERVICE_HOST_MEMORY_OFFLOAD_ANNOTATIONS_H_

#include <string_view>

namespace zkx {
namespace host_memory_offload_annotations {

// External annotations:
inline constexpr std::string_view kDevicePlacement =
    "annotate_device_placement";
inline constexpr std::string_view kMemoryTargetPinnedHost = "pinned_host";
inline constexpr std::string_view kMemoryTargetUnpinnedHost = "unpinned_host";
inline constexpr std::string_view kMemoryTargetDevice = "device";
inline constexpr std::string_view kMemoryTargetDeviceSram = "device_sram";
inline constexpr std::string_view kMemoryTargetPinnedDevice = "pinned_device";

// Internal annotations:
inline constexpr std::string_view kMoveToHostCustomCallTarget = "MoveToHost";
inline constexpr std::string_view kMoveToDeviceCustomCallTarget =
    "MoveToDevice";
inline constexpr std::string_view kPinToDeviceCustomCallTarget = "PinToDevice";
inline constexpr std::string_view kPinToDeviceSramCustomCallTarget =
    "PinToDeviceSram";

}  // namespace host_memory_offload_annotations
}  // namespace zkx

#endif  // ZKX_SERVICE_HOST_MEMORY_OFFLOAD_ANNOTATIONS_H_
