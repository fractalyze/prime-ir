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

#ifndef ZKX_BACKENDS_PROFILER_PLUGIN_PLUGIN_TRACER_IMPL_H_
#define ZKX_BACKENDS_PROFILER_PLUGIN_PLUGIN_TRACER_IMPL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "xla/tsl/profiler/lib/profiler_interface.h"
#include "xla/tsl/profiler/protobuf/xplane.pb.h"
#include "zkx/backends/profiler/plugin/profiler_c_api.h"

struct PLUGIN_Profiler {
  std::optional<tensorflow::profiler::XSpace> space;
  std::unique_ptr<std::vector<uint8_t>> buffer;
  size_t byte_size;
  std::unique_ptr<tsl::profiler::ProfilerInterface> impl;
  bool stopped;
};

namespace zkx::profiler {

PLUGIN_Profiler_Error* PLUGIN_Profiler_Create(
    PLUGIN_Profiler_Create_Args* args);

PLUGIN_Profiler_Error* PLUGIN_Profiler_Destroy(
    PLUGIN_Profiler_Destroy_Args* args);

PLUGIN_Profiler_Error* PLUGIN_Profiler_Start(PLUGIN_Profiler_Start_Args* args);

PLUGIN_Profiler_Error* PLUGIN_Profiler_Stop(PLUGIN_Profiler_Stop_Args* args);

PLUGIN_Profiler_Error* PLUGIN_Profiler_CollectData(
    PLUGIN_Profiler_CollectData_Args* args);

}  // namespace zkx::profiler

#endif  // ZKX_BACKENDS_PROFILER_PLUGIN_PLUGIN_TRACER_IMPL_H_
