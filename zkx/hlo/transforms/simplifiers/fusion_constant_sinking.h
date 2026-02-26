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

#ifndef ZKX_HLO_TRANSFORMS_SIMPLIFIERS_FUSION_CONSTANT_SINKING_H_
#define ZKX_HLO_TRANSFORMS_SIMPLIFIERS_FUSION_CONSTANT_SINKING_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"

#include "zkx/hlo/pass/hlo_pass_interface.h"

namespace zkx {

// A pass which sinks constants into fusion computations.
class FusionConstantSinking : public HloModulePass {
 public:
  std::string_view name() const override { return "fusion_constant_sinking"; }

  // Run fusion constant sinking operations on the given module. Returns whether
  // the module was changed (constant expressions folded).
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<std::string_view>& execution_threads) override;
};

}  // namespace zkx

#endif  // ZKX_HLO_TRANSFORMS_SIMPLIFIERS_FUSION_CONSTANT_SINKING_H_
