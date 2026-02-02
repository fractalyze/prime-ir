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

#ifndef ZKX_SERVICE_GPU_TRANSFORMS_LAYOUT_ASSIGNMENT_H_
#define ZKX_SERVICE_GPU_TRANSFORMS_LAYOUT_ASSIGNMENT_H_

#include <cstdint>
#include <initializer_list>

#include "absl/status/status.h"
#include "absl/types/span.h"

#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/service/computation_layout.h"
#include "zkx/service/layout_assignment.h"
#include "zkx/stream_executor/device_description.h"

namespace zkx::gpu {

// GPU-specific layout assignment pass which preassigns layouts to satisfy
// layout constraints for operands and results of library calls.
// NOTE: XLA version also has gpu_version_, dnn_version_ members and
// AddBackendConstraintsToDnnConvCustomCall method for DNN convolution support.
class GpuLayoutAssignment : public LayoutAssignment {
 public:
  explicit GpuLayoutAssignment(
      ComputationLayout* entry_computation_layout,
      const se::DeviceDescription& device_description,
      ChannelLayoutConstraints* channel_constraints = nullptr)
      : LayoutAssignment(entry_computation_layout, channel_constraints),
        device_description_(device_description) {}
  ~GpuLayoutAssignment() override = default;

 protected:
  absl::Status AddBackendConstraints(LayoutConstraints* constraints) override;

 private:
  // dim_groups are ordered from major to minor dimensions.
  absl::Status SetOperandMajorToMinorLayout(
      const HloInstruction* instruction, int64_t operand,
      std::initializer_list<absl::Span<const int64_t>> dim_groups);

  absl::Status SetDotOperandLayout(const HloInstruction* instruction,
                                   int64_t operand,
                                   absl::Span<const int64_t> batch_dims,
                                   absl::Span<const int64_t> row_dims,
                                   absl::Span<const int64_t> col_dims);

  absl::Status SetDotOperandLayoutToMinorContracting(
      const HloInstruction* instruction, int64_t operand,
      absl::Span<const int64_t> batch_dims,
      absl::Span<const int64_t> contracting_dims,
      absl::Span<const int64_t> noncontracting_dims);

  absl::Status SetDotLayout(const HloInstruction* instruction,
                            LayoutConstraints* constraints);

  bool PropagateReductionLayoutToOperand(const HloInstruction* user) override;

  bool InstructionCanChangeLayoutInstance(
      const HloInstruction* instruction) override;

  absl::Status AddDotBackendConstraints(LayoutConstraints* constraints,
                                        HloDotInstruction* instruction);

  const se::DeviceDescription& device_description_;
};

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_TRANSFORMS_LAYOUT_ASSIGNMENT_H_
