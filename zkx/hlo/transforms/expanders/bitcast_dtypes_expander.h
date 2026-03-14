/* Copyright 2021 The OpenXLA Authors.
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

#ifndef ZKX_HLO_TRANSFORMS_EXPANDERS_BITCAST_DTYPES_EXPANDER_H_
#define ZKX_HLO_TRANSFORMS_EXPANDERS_BITCAST_DTYPES_EXPANDER_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"

#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/transforms/expanders/op_expander_pass.h"

namespace zkx {

// A pass which expands bitcast-convert between differently sized dtypes to a
// reduction.
class BitcastDtypesExpander : public OpExpanderPass {
 public:
  std::string_view name() const override { return "bitcast_dtypes_expander"; }

 protected:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;

  absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;

 private:
  absl::flat_hash_map<std::string, HloComputation*>
      computation_cache_;  // not owned
};

}  // namespace zkx

#endif  // ZKX_HLO_TRANSFORMS_EXPANDERS_BITCAST_DTYPES_EXPANDER_H_
