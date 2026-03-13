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

#ifndef ZKX_SERVICE_GPU_TRANSFORMS_POSEIDON2_FUSION_REWRITER_H_
#define ZKX_SERVICE_GPU_TRANSFORMS_POSEIDON2_FUSION_REWRITER_H_

#include <string_view>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"

#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/pass/hlo_pass_interface.h"

namespace zkx::gpu {

// Detects Poseidon2 permutation patterns in HLO and wraps them into a single
// kCustom fusion with custom_fusion_config.name = "poseidon2_permutation".
//
// The pattern is identified by finding 3 consecutive kWhile instructions in a
// computation:
//   while #1 (external init): trip count = external_rounds (typically 4)
//   while #2 (internal):      trip count = internal_rounds (13, 20, 21, or 23)
//   while #3 (external term): trip count = external_rounds (same as #1)
//
// The rewriter extracts constants from while bodies and passes them as
// additional fusion operands. Configuration (width, rounds, sbox_degree) is
// encoded in the fusion name string.
//
// Must run before PriorityFusion.
class Poseidon2FusionRewriter : public HloModulePass {
 public:
  std::string_view name() const override { return "poseidon2-fusion-rewriter"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<std::string_view>& execution_threads) override;
};

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_TRANSFORMS_POSEIDON2_FUSION_REWRITER_H_
