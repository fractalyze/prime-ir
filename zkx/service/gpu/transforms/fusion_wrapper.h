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

#ifndef ZKX_SERVICE_GPU_TRANSFORMS_FUSION_WRAPPER_H_
#define ZKX_SERVICE_GPU_TRANSFORMS_FUSION_WRAPPER_H_

#include <string_view>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"

#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/pass/hlo_pass_interface.h"
#include "zkx/stream_executor/device_description.h"

namespace zkx::gpu {

// Wraps leftover unfused instructions that are in the entry computation that
// have no LHLO equivalent in fusions containing just that instruction.
//
// In XLA, this inherits from emitters::FusionWrapperBase which provides the
// Run() logic. Since FusionWrapperBase is not yet ported to ZKX, the run
// logic is inlined here.
class FusionWrapper : public HloModulePass {
 public:
  explicit FusionWrapper(const se::DeviceDescription& device_description)
      : device_description_(device_description) {}

  std::string_view name() const override { return "fusion-wrapper"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<std::string_view>& execution_threads) override;

 private:
  const se::DeviceDescription& device_description_;
};

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_TRANSFORMS_FUSION_WRAPPER_H_
