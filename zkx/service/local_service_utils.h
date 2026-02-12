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

#ifndef ZKX_SERVICE_LOCAL_SERVICE_UTILS_H_
#define ZKX_SERVICE_LOCAL_SERVICE_UTILS_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

#include "zkx/client/executable_build_options.h"
#include "zkx/hlo/builder/zkx_computation.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/service/backend.h"
#include "zkx/service/hlo_module_config.h"
#include "zkx/service/service.h"
#include "zkx/shape.h"

namespace zkx {

// Validates the computation argument layouts, and returns the corresponding
// HloModuleConfig.
absl::StatusOr<std::unique_ptr<HloModuleConfig>> GetHloModuleConfig(
    const ZkxComputation& computation,
    absl::Span<const Shape* const> argument_layouts,
    const ExecutableBuildOptions& build_options,
    ServiceOptions* options = nullptr, Backend* backend = nullptr);

}  // namespace zkx

#endif  // ZKX_SERVICE_LOCAL_SERVICE_UTILS_H_
