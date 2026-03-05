/* Copyright 2020 The OpenXLA Authors.
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

#include "zkx/python/profiler_utils.h"

#include "absl/log/log.h"

namespace zkx {

// TODO(chokobole): Implement RegisterProfiler when PJRT C API plugin profiler
// infrastructure is ported to ZKX. This requires:
//   - xla/pjrt/c/pjrt_c_api.h
//   - xla/backends/profiler/plugin/plugin_tracer.h
//   - xla/pjrt/c/pjrt_c_api_profiler_extension.h
void RegisterProfiler(const void* pjrt_api) {
  LOG(WARNING) << "Plugin profiler registration is not yet supported in ZKX.";
}

}  // namespace zkx
