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
#include "zkx/backends/gpu/codegen/copy.h"

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/backends/gpu/codegen/fusion_emitter.h"
#include "zkx/backends/gpu/runtime/copy_thunk.h"
#include "zkx/backends/gpu/runtime/thunk.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/hlo/utils/hlo_traversal.h"
#include "zkx/service/buffer_assignment.h"
#include "zkx/service/gpu/ir_emitter_context.h"
#include "zkx/shape.h"
#include "zkx/shape_util.h"

namespace zkx::gpu {

absl::StatusOr<FusionEmissionResult> MemcpyFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  std::vector<BufferAllocation::Slice> src_buffers;
  for (const HloInstructionAdaptor& root_adaptor : analysis_.fusion_roots()) {
    const HloInstruction* root = &root_adaptor.instruction();
    const HloInstruction* src_instr =
        fusion.operand(root->operand(0)->parameter_number());
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                        buffer_assignment_->GetUniqueSlice(src_instr, {}));
    src_buffers.push_back(slice);
  }

  std::vector<BufferAllocation::Slice> dst_buffers;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      fusion.shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (!subshape.IsArray()) {
          return absl::OkStatus();
        }
        TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                            buffer_assignment_->GetUniqueSlice(&fusion, index));
        dst_buffers.push_back(slice);
        return absl::OkStatus();
      }));

  FusionEmissionResult result;
  for (size_t i = 0; i < src_buffers.size(); ++i) {
    if (src_buffers[i] != dst_buffers[i]) {
      result.thunks.emplace_back(std::make_unique<DeviceToDeviceCopyThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(&fusion),
          /*source_buffer=*/src_buffers[i],
          /*destination_buffer=*/dst_buffers[i],
          /*mem_size=*/src_buffers[i].size()));
    }
  }
  return result;
}

}  // namespace zkx::gpu
