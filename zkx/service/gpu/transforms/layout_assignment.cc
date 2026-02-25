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

#include "zkx/service/gpu/transforms/layout_assignment.h"

#include <cstddef>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/layout.h"
#include "zkx/layout_util.h"
#include "zkx/primitive_util.h"
#include "zkx/service/gpu/ir_emission_utils.h"
#include "zkx/service/gpu/matmul_indexing_utils.h"
#include "zkx/service/gpu/matmul_utils.h"
#include "zkx/service/gpu/reduction_utils.h"
#include "zkx/service/host_memory_offload_annotations.h"
#include "zkx/service/logical_buffer.h"
#include "zkx/shape.h"
#include "zkx/shape_layout.h"
#include "zkx/shape_util.h"

namespace zkx::gpu {
namespace {

// Checks if the instruction operates on sub-byte types (e.g., S4, U4).
// These require special layout handling for packed operations.
bool IsPackedInstruction(const HloInstruction* instruction) {
  return primitive_util::IsSubByteNonPredType(
             instruction->shape().element_type()) ||
         (instruction->opcode() == HloOpcode::kConvert &&
          primitive_util::IsSubByteNonPredType(
              instruction->operand(0)->shape().element_type()));
}

bool DotCanSupportShapeWithLayout(const HloInstruction* dot,
                                  const Shape& shape) {
  const DotDimensionNumbers& dot_dims = dot->dot_dimension_numbers();
  // If we are able to construct a `MatrixLayout` then the dot can support
  // this layout.
  return MatrixLayout::For(shape, dot_dims.lhs_batch_dimensions().size(),
                           dot->operand(0)->shape().rank() -
                               dot_dims.lhs_contracting_dimensions().size() -
                               dot_dims.lhs_batch_dimensions().size(),
                           dot_dims.rhs_batch_dimensions().size(),
                           dot->operand(1)->shape().rank() -
                               dot_dims.rhs_contracting_dimensions().size() -
                               dot_dims.rhs_batch_dimensions().size())
      .ok();
}

}  // namespace

absl::Status GpuLayoutAssignment::AddDotBackendConstraints(
    LayoutConstraints* constraints, HloDotInstruction* instruction) {
  struct Side {
    size_t operand_no;
    const HloInstruction* operand;
    absl::Span<const int64_t> batch_dims;
    absl::Span<const int64_t> contracting_dims;
    PrimitiveType type;
    std::vector<int64_t> non_contracting_dims;
  };
  auto make_side =
      [&](size_t operand_no, absl::Span<const int64_t> batch_dims,
          absl::Span<const int64_t> contracting_dims) -> absl::StatusOr<Side> {
    Side side = {operand_no, instruction->operand(operand_no), batch_dims,
                 contracting_dims};
    side.type = side.operand->shape().element_type();
    TF_ASSIGN_OR_RETURN(
        side.non_contracting_dims,
        GetNonContractingDims(side.operand->shape(), side.batch_dims,
                              side.contracting_dims));
    return side;
  };
  const DotDimensionNumbers& dot_dims = instruction->dot_dimension_numbers();
  TF_ASSIGN_OR_RETURN(const Side lhs,
                      make_side(0, dot_dims.lhs_batch_dimensions(),
                                dot_dims.lhs_contracting_dimensions()));
  TF_ASSIGN_OR_RETURN(const Side rhs,
                      make_side(1, dot_dims.rhs_batch_dimensions(),
                                dot_dims.rhs_contracting_dimensions()));

  const PrimitiveType& output_type = instruction->shape().element_type();

  // S8 to S32 dot operations require contracting dims to be minor.
  const bool is_s8_to_s32 = output_type == PrimitiveType::S32 &&
                            lhs.type == PrimitiveType::S8 &&
                            rhs.type == PrimitiveType::S8;

  for (const Side& side : {lhs, rhs}) {
    // Packed instructions (sub-byte types like S4, U4) and S8->S32 operations
    // require contracting dims to be in minor positions.
    if (IsPackedInstruction(side.operand) || is_s8_to_s32) {
      TF_RETURN_IF_ERROR(SetDotOperandLayoutToMinorContracting(
          instruction, side.operand_no, side.batch_dims, side.contracting_dims,
          side.non_contracting_dims));
    } else if (!side.batch_dims.empty() || side.contracting_dims.size() > 1 ||
               side.non_contracting_dims.size() > 1) {
      TF_RETURN_IF_ERROR(SetDotOperandLayout(
          instruction, side.operand_no, side.batch_dims, side.contracting_dims,
          side.non_contracting_dims));
    }
  }

  if (!lhs.batch_dims.empty() || lhs.non_contracting_dims.size() > 1 ||
      rhs.non_contracting_dims.size() > 1) {
    TF_RETURN_IF_ERROR(SetDotLayout(instruction, constraints));
  }

  return absl::OkStatus();
}

absl::Status GpuLayoutAssignment::AddBackendConstraints(
    LayoutConstraints* constraints) {
  auto post_order = constraints->computation()->MakeInstructionPostOrder();
  for (auto iterator = post_order.rbegin(); iterator != post_order.rend();
       ++iterator) {
    HloInstruction* instruction = *iterator;

    if (HloPredicateIsOp<HloOpcode::kDot>(instruction)) {
      TF_RETURN_IF_ERROR(AddDotBackendConstraints(
          constraints, Cast<HloDotInstruction>(instruction)));
    } else if (HloPredicateIsOp<HloOpcode::kTranspose>(instruction)) {
      const HloInstruction* operand = instruction->operand(0);
      if ((HloPredicateIsNotOp<HloOpcode::kDot>(operand)) ||
          (operand->user_count() > 1)) {
        continue;
      }

      Shape shape = operand->shape();
      *shape.mutable_layout() =
          LayoutUtil::MakeLayoutFromMajorToMinor(instruction->dimensions());

      if (DotCanSupportShapeWithLayout(operand, shape)) {
        TF_RETURN_IF_ERROR(
            SetOperandLayout(shape, instruction, /*operand_no=*/0));
      }
    } else if (HloPredicateIsOp<HloOpcode::kFft>(instruction)) {
      // cuFFT requires a dim0 major layout.
      Shape op0_shape = instruction->operand(0)->shape();
      LayoutUtil::SetToDefaultLayout(&op0_shape);
      Shape output_shape = instruction->shape();
      LayoutUtil::SetToDefaultLayout(&output_shape);
      TF_RETURN_IF_ERROR(SetOperandLayout(op0_shape, instruction, 0));
      TF_RETURN_IF_ERROR(SetInstructionLayout(output_shape, instruction));
    } else if ((HloPredicateIsOp<HloOpcode::kSort>(instruction) ||
                IsCubDeviceRadixSort(*instruction)) &&
               instruction->operand(0)->shape().rank() > 1) {
      Shape keys_shape = instruction->operand(0)->shape();
      Layout keys_layout =
          LayoutUtil::GetDefaultLayoutForRank(keys_shape.rank());
      for (int64_t i = 0; i < instruction->operand_count(); ++i) {
        Shape shape = instruction->operand(i)->shape();
        *shape.mutable_layout() = keys_layout;
        TF_RETURN_IF_ERROR(SetOperandLayout(shape, instruction, i));
        const LogicalBuffer* output_buffer;
        if (instruction->shape().IsArray()) {
          TF_ASSIGN_OR_RETURN(
              output_buffer,
              points_to_analysis_->GetBufferDefinedAt(instruction, {}));
        } else {
          TF_ASSIGN_OR_RETURN(
              output_buffer,
              points_to_analysis_->GetBufferDefinedAt(instruction, {i}));
        }
        TF_RETURN_IF_ERROR(SetBufferLayout(keys_layout, *output_buffer));
      }
    } else if (HloPredicateIsOp<HloOpcode::kReduceScatter>(instruction)) {
      auto ars = Cast<HloReduceScatterInstruction>(instruction);
      TF_RETURN_IF_ERROR(SetInstructionLayout(
          ShapeUtil::MoveDimToMajor(ars->shape(), ars->scatter_dimension()),
          ars));
    } else if (HloPredicateIsOp<HloOpcode::kAllGather>(instruction)) {
      auto ag = Cast<HloAllGatherInstruction>(instruction);
      TF_RETURN_IF_ERROR(SetInstructionLayout(
          ShapeUtil::MoveDimToMajor(ag->shape(), ag->all_gather_dimension()),
          ag));
    } else if (HloPredicateIsOp<HloOpcode::kAllToAll>(instruction) &&
               instruction->shape().IsArray()) {
      auto* all_to_all = Cast<HloAllToAllInstruction>(instruction);
      TF_RETURN_IF_ERROR(SetInstructionLayout(
          ShapeUtil::MoveDimToMajor(all_to_all->shape(),
                                    *all_to_all->split_dimension()),
          all_to_all));
    } else if (HloPredicateIsOp<HloOpcode::kSend>(instruction)) {
      Shape s = instruction->operand(0)->shape();
      LayoutUtil::SetToDefaultLayout(&s);
      TF_RETURN_IF_ERROR(SetInstructionLayout(s, instruction->operand(0)));
      TF_RETURN_IF_ERROR(
          SetArrayOperandLayout(s.layout(), instruction->operand(0), 0));
    } else if (HloPredicateIsOp<HloOpcode::kRecv>(instruction)) {
      Shape s = instruction->shape();
      ShapeUtil::ForEachMutableSubshape(
          &s, [&](Shape* subshape, const ShapeIndex& index) {
            LayoutUtil::SetToDefaultLayout(subshape);
          });
      TF_RETURN_IF_ERROR(SetInstructionLayout(s, instruction));
    }
  }
  return absl::OkStatus();
}

absl::Status GpuLayoutAssignment::SetDotOperandLayout(
    const HloInstruction* instruction, int64_t operand,
    absl::Span<const int64_t> batch_dims, absl::Span<const int64_t> row_dims,
    absl::Span<const int64_t> col_dims) {
  Shape shape = instruction->operand(operand)->shape();

  // First, try to use the existing layout, if present.
  if (shape.has_layout() &&
      MatrixLayout::For(shape, batch_dims, row_dims, col_dims).ok())
    // Re-set the operand layout, so it becomes mandatory.
    return SetOperandLayout(shape, instruction, operand);

  // Next, try the default layout (for the sake of everybody's sanity).
  LayoutUtil::SetToDefaultLayout(&shape);
  if (MatrixLayout::For(shape, batch_dims, row_dims, col_dims).ok())
    return SetOperandLayout(shape, instruction, operand);

  // Otherwise, fallback to forcing (batch, rows, cols) layout.
  return SetOperandMajorToMinorLayout(
      instruction, operand,
      /*dim_groups=*/{batch_dims, row_dims, col_dims});
}

absl::Status GpuLayoutAssignment::SetDotOperandLayoutToMinorContracting(
    const HloInstruction* instruction, int64_t operand,
    absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> contracting_dims,
    absl::Span<const int64_t> noncontracting_dims) {
  Shape shape = instruction->operand(operand)->shape();

  if (shape.has_layout() &&
      shape.layout().minor_to_major_size() >= contracting_dims.size()) {
    // Check that the contracting dimensions are physically minor, i.e. check
    // that minor physical dimensions all point to contracting logical
    // dimensions.
    bool contracting_dims_are_minor = true;
    const auto& minor_to_major = shape.layout().minor_to_major();
    for (int64_t i = 0; i < contracting_dims.size(); ++i) {
      if (!absl::c_linear_search(contracting_dims, minor_to_major[i])) {
        contracting_dims_are_minor = false;
        break;
      }
    }

    // If contracting dims are already minor, and the layout is valid, keep it.
    if (contracting_dims_are_minor &&
        MatrixLayout::For(shape, batch_dims, noncontracting_dims,
                          contracting_dims)
            .ok()) {
      // Re-set the operand layout, so it becomes mandatory.
      return SetOperandLayout(shape, instruction, operand);
    }
  }
  return SetOperandMajorToMinorLayout(
      instruction, operand,
      /*dim_groups=*/
      {batch_dims, noncontracting_dims, contracting_dims});
}

absl::Status GpuLayoutAssignment::SetOperandMajorToMinorLayout(
    const HloInstruction* instruction, int64_t operand,
    std::initializer_list<absl::Span<const int64_t>> dim_groups) {
  size_t size = 0;
  for (auto group : dim_groups) size += group.size();
  std::vector<int64_t> major_to_minor;
  major_to_minor.reserve(size);
  for (const auto& group : dim_groups) {
    major_to_minor.insert(major_to_minor.end(), group.begin(), group.end());
  }

  Shape shape = instruction->operand(operand)->shape();
  *shape.mutable_layout() =
      LayoutUtil::MakeLayoutFromMajorToMinor(major_to_minor);
  return SetOperandLayout(shape, instruction, operand);
}

absl::Status GpuLayoutAssignment::SetDotLayout(
    const HloInstruction* instruction, LayoutConstraints* constraints) {
  // If a user has requested a layout that we can support, use that.
  for (const HloInstruction* user : instruction->users()) {
    for (int64_t i = 0; i < user->operand_count(); ++i) {
      if (user->operand(i) != instruction) {
        continue;
      }

      const ShapeLayout* constraint = constraints->OperandLayout(user, i);
      if ((constraint != nullptr) &&
          DotCanSupportShapeWithLayout(instruction, constraint->shape())) {
        return SetInstructionLayout(constraint->shape(), instruction);
      }
    }
  }

  // Otherwise, use the default layout.
  return SetInstructionLayout(
      LayoutUtil::GetWithDefaultLayout(instruction->shape()), instruction);
}

bool GpuLayoutAssignment::PropagateReductionLayoutToOperand(
    const HloInstruction* user) {
  int64_t reduction_size = 1;
  for (int64_t reduction_dim : user->dimensions()) {
    reduction_size *= user->operand(0)->shape().dimensions(reduction_dim);
  }
  int64_t kept_dimension_size = ShapeUtil::ElementsIn(user->shape());
  return IsUnnestedReductionFasterThanElemental(
      {/*is_row_reduction=*/true, {1, kept_dimension_size, reduction_size}},
      device_description_);
}

bool GpuLayoutAssignment::InstructionCanChangeLayoutInstance(
    const HloInstruction* instruction) {
  // The host offloading custom calls will be eventually removed
  // by the offloader, so we need to make sure that the calls do not change
  // the layout and thus cause layout mismatches after the removal.
  // NOTE: XLA also checks kTopKCustomCallTarget here. Add if needed.
  const HloCustomCallInstruction* custom_call =
      DynCast<HloCustomCallInstruction>(instruction);
  if (custom_call != nullptr &&
      (custom_call->custom_call_target() ==
           host_memory_offload_annotations::kMoveToHostCustomCallTarget ||
       custom_call->custom_call_target() ==
           host_memory_offload_annotations::kMoveToDeviceCustomCallTarget)) {
    return false;
  }

  return LayoutAssignment::InstructionCanChangeLayoutInstance(instruction);
}

}  // namespace zkx::gpu
