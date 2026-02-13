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

#ifndef ZKX_SERVICE_GPU_HLO_FUSION_STATS_H_
#define ZKX_SERVICE_GPU_HLO_FUSION_STATS_H_

#include <cstdint>
#include <map>
#include <set>
#include <string>

#include "absl/status/status.h"

#include "zkx/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_module.h"

// Read-only pass logging statistics about HLO fusion ops in the module. Enabled
// at VLOG level 1 only.
namespace zkx::gpu {

class HloOpcodeHistogram : public std::map<std::set<std::string>, int64_t> {
 public:
  std::string ToString();
};

class HloFusionStatsVisitor : public ConstDfsHloVisitorWithDefault {
 public:
  std::string ToString();

 protected:
  absl::Status DefaultAction(const HloInstruction* instr) final;

  absl::Status HandleFusion(const HloInstruction* fusion) override;

 private:
  int64_t num_fusions_ = 0;
  int64_t num_loop_fusions_ = 0;
  int64_t num_input_fusions_ = 0;
  HloOpcodeHistogram loop_fusion_opcode_histogram_;
  HloOpcodeHistogram input_fusion_opcode_histogram_;
};

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_HLO_FUSION_STATS_H_
