/* Copyright 2017 The OpenXLA Authors.
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

#include "zkx/service/cpu/dynamic_update_slice_util.h"

#include "zkx/service/cpu/backend_config.pb.h"

namespace zkx::llvm_ir {

bool MayBeImplementedAsInPlaceDynamicUpdateSlice(const HloInstruction* instr) {
  // Today we can't emit a dynamic-update-slice if the DUS node is parallelized;
  // the emitter will not emit correct code.  It's possible to change this, but
  // then ParallelTaskAssigner would have to somehow know whether a node *will*
  // be emitted as an in-place DUS, and it can't, because it doesn't have a
  // buffer assignment when it runs.
  auto cpu_backend_config_or = instr->backend_config<cpu::BackendConfig>();
  if (cpu_backend_config_or.ok() &&
      !cpu_backend_config_or->outer_dimension_partitions().empty()) {
    return false;
  }

  // Until we know the final buffer assignment, any unfused dynamic-update-slice
  // might be implementable as an in-place DUS.
  if (instr->opcode() == HloOpcode::kDynamicUpdateSlice) {
    return true;
  }

  // A fusion may be implementable as an in-place dynamic update slice if
  //  - it's a loop fusion,
  //  - dynamic-update-slice is the root of the fusion, and
  //  - operand 0 of the dynamic-update-slice is a parameter to the fusion
  //    (ignoring any get-tuple-element operations in the way).
  if (instr->IsLoopFusion()) {
    const HloInstruction* fused_root = instr->fused_expression_root();
    return fused_root->opcode() == HloOpcode::kDynamicUpdateSlice &&
           fused_root->operand(0)->LatestNonGteAncestor()->opcode() ==
               HloOpcode::kParameter;
  }

  return false;
}

}  // namespace zkx::llvm_ir
