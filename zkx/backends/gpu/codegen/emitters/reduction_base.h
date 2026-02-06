/* Copyright 2024 The OpenXLA Authors.
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
#ifndef ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_REDUCTION_BASE_H_
#define ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_REDUCTION_BASE_H_

#include <cstdint>
#include <vector>

#include "zkx/hlo/analysis/indexing_map.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/service/gpu/hlo_fusion_analysis.h"
#include "zkx/service/gpu/reduction_utils.h"
#include "zkx/util.h"

namespace zkx::gpu {

struct ReductionGroups {
  std::vector<std::vector<const HloInstruction*>> grouped_roots;

  // For each root of the fusion, returns the index of the group it was placed
  // in.
  std::vector<int> group_id_per_root;

  // For each root of the fusion, returns whether it is a reduction root, or
  // an additional output.
  std::vector<bool> is_reduction_root;
};

ReductionGroups GroupDisjointReductions(const HloFusionAnalysis& analysis);

int GetVectorSizeForMlir(const HloFusionAnalysis& analysis, int64_t minor_dim,
                         int num_threads);

void AddGroupIdConstraint(IndexingMap& map, int64_t root_index,
                          const ReductionGroups& groups);

}  // namespace zkx::gpu

#endif  // ZKX_BACKENDS_GPU_CODEGEN_EMITTERS_REDUCTION_BASE_H_
