/* Copyright 2022 The OpenXLA Authors.
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

#ifndef ZKX_SERVICE_LAYOUT_NORMALIZATION_H_
#define ZKX_SERVICE_LAYOUT_NORMALIZATION_H_

#include <functional>
#include <optional>

#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/pass/hlo_pass_interface.h"

namespace zkx {

using CustomCallTransformer =
    std::function<absl::StatusOr<std::optional<HloInstruction*>>(
        HloCustomCallInstruction*)>;

// Normalize shapes for some subsets of HLOs.
//
// A shape is called "normalized" when it's layout is descending, and no
// degenerate dimensions are present.
//
// The normalization pass converts shapes to physically-equivalent but
// normalized ones, e.g. f32[5,1,4]{0,1,2} is converted to f32[4,5]{1,0}.
class LayoutNormalization : public HloModulePass {
 public:
  // The provided custom_call_transformer allows backend to specify custom-call
  // transformation rules.
  explicit LayoutNormalization(
      const CustomCallTransformer& custom_call_transformer = nullptr)
      : custom_call_transformer_(custom_call_transformer) {}

  std::string_view name() const override { return "layout_normalization"; }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<std::string_view>& execution_threads) override;

 private:
  CustomCallTransformer custom_call_transformer_;
};

}  // namespace zkx

#endif  // ZKX_SERVICE_LAYOUT_NORMALIZATION_H_
