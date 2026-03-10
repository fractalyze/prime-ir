/* Copyright 2026 The ZKX Authors.

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

#ifndef ZKX_SERVICE_GPU_TRANSFORMS_MSM_BATCH_FUSION_H_
#define ZKX_SERVICE_GPU_TRANSFORMS_MSM_BATCH_FUSION_H_

#include <string_view>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"

#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/pass/hlo_pass_interface.h"

namespace zkx::gpu {

// Fuses consecutive MSM instructions that share the same bases operand into a
// single batched MSM instruction.
//
// Pattern: find MSM instructions in the same computation where:
//   - Same bases operand (pointer identity after CSE)
//   - Same window_bits, precompute_factor, bitsize config
//   - No data dependency between them
//
// Fuse by:
//   1. Concatenate scalar operands: [N] + [N] → [B*N]
//   2. Create single kMsm with batch_size=B, are_points_shared=true
//   3. Result shape becomes [B] (B point results)
//   4. Add slice ops to extract individual results
class MsmBatchFusion : public HloModulePass {
 public:
  std::string_view name() const override { return "msm-batch-fusion"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<std::string_view>& execution_threads) override;
};

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_TRANSFORMS_MSM_BATCH_FUSION_H_
