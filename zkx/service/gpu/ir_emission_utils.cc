/* Copyright 2017 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

#include "zkx/service/gpu/ir_emission_utils.h"

#include <queue>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/primitive_util.h"
#include "zkx/util.h"

namespace zkx::gpu {

bool IsSliceWithUnitStrides(const HloInstruction* instr) {
  auto slice = DynCast<HloSliceInstruction>(instr);
  return slice && absl::c_all_of(slice->slice_strides(),
                                 [](int64_t stride) { return stride == 1; });
}

bool IsCubDeviceRadixSort(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kCubDeviceRadixSortTarget;
}

absl::StatusOr<BufferAllocation::Slice> GetAllocationSlice(
    const BufferAssignment& buffer_assignment, const HloInstruction* instr,
    const ShapeIndex& index) {
  return buffer_assignment.GetUniqueSlice(instr, index);
}

namespace {

template <typename T>
absl::InlinedVector<const HloInstruction*, 4> GetStartIndices(T instr) {
  absl::InlinedVector<const HloInstruction*, 4> result;
  for (int i = instr->first_index_operand_number(); i < instr->operand_count();
       i++) {
    const HloInstruction* index = instr->operand(i);
    result.push_back(index);
  }
  return result;
}

}  // namespace

absl::StatusOr<bool> CanEmitFusedDynamicUpdateSliceInPlaceForGpu(
    const HloFusionAdaptor& fusion_adaptor,
    std::function<absl::StatusOr<BufferAllocation::Slice>(
        const HloInstruction* instr, const ShapeIndex& index)>
        get_allocation_slice,
    const HloInstruction* fusion) {
  std::vector<HloInstructionAdaptor> dus_instrs =
      GetOutputDefiningDynamicUpdateSlices(fusion_adaptor.GetRoots());

  // This check could probably be relaxed: if code generation is made to use a
  // separate parallel loop for each dynamic slice update, then it shouldn't be
  // necessary for every output to be a dynamic slice update, nor to have the
  // same shape.
  if (dus_instrs.empty()) {
    return false;
  }

  if (dus_instrs.size() != fusion_adaptor.GetRoots().size()) {
    return false;
  }

  Shape update_shape = dus_instrs[0].GetOperand(1).shape();

  for (int i = 0; i < dus_instrs.size(); ++i) {
    const auto& dus = dus_instrs[i];

    // DynamicUpdateSlice ops should have a single path to the root to avoid
    // allowing a dynamic slice update to depend on another, as this would not
    // be guaranteed to work with the current codegen.
    // We follow DUS users until we find an instruction without users. We
    // support only few patterns:
    //
    //   (1) ROOT dynamic-update-slice
    //   (2) ROOT tuple(dynamic-update-slice)
    //   (3) ROOT bitcast(dynamic-update-slice)
    //   (4) ROOT tuple(bitcast(dynamic-update-slice))
    //
    // In case there is a root tuple, the search will stop at the tuple operand,
    // as the root tuple is not considered a real user by HloInstructionAdaptor.
    // Note that due to AlgebraicSimplifier we will never have a chain of
    // bitcasts.
    HloInstructionAdaptor real_root = dus;
    auto users = real_root.GetUsers();
    while (!users.empty()) {
      if (users.size() > 1) {
        return false;
      }
      real_root = users.front();
      if (real_root.opcode() != HloOpcode::kBitcast) {
        return false;
      }
      users = real_root.GetUsers();
    }

    // Find "real" DUS operand by skipping bitcasted operands.
    HloInstructionAdaptor operand = dus.GetOperand(0);
    if (fusion_adaptor.ContainsInstruction(operand) &&
        operand.opcode() == HloOpcode::kBitcast) {
      operand = operand.GetOperand(0);
    }

    // Operand to a DUS (or Bitcast) must be a fusion parameter.
    // HloInstructionAdaptor skips parameters, so we need to check whether
    // 'operand' is outside of the fusion.
    if (fusion_adaptor.ContainsInstruction(operand)) {
      return false;
    }

    // We require that the parameter being updated is only read at the same
    // index positions by all users, since we otherwise risk a race condition
    // when updating the parameter inplace.
    std::queue<HloInstructionAdaptor> q;
    absl::flat_hash_set<const HloInstruction*> visited;
    q.push(operand);
    visited.insert(&operand.instruction());
    // We have already checked above that the DUS only has one user. So we don't
    // need to visit it during the breadth-first search.
    visited.insert(&dus.instruction());
    while (!q.empty()) {
      HloInstructionAdaptor instr = q.front();
      q.pop();
      for (const HloInstructionAdaptor& user : instr.GetUsers()) {
        if (user.opcode() == HloOpcode::kDynamicSlice &&
            dus.GetOperand(0) == user.GetOperand(0) &&
            update_shape == user.shape()) {
          // We can still emit in-place in this case if the same slice is
          // accessed by the DUS and the DS. If they don't access the same
          // slice, the two slices might partially overlap and read/write the
          // same index at different times, and then we cannot guarantee that we
          // read before it is overwritten. However if both access only a single
          // element, there also can be no race condition.
          absl::InlinedVector<const HloInstruction*, 4> user_start_indices =
              GetStartIndices(
                  Cast<HloDynamicSliceInstruction>(&user.instruction()));
          absl::InlinedVector<const HloInstruction*, 4> dus_start_indices =
              GetStartIndices(
                  Cast<HloDynamicUpdateSliceInstruction>(&dus.instruction()));
          if (ShapeUtil::ElementsIn(update_shape) != 1 &&
              user_start_indices != dus_start_indices) {
            return false;
          }
        } else if (user != dus &&
                   user.opcode() == HloOpcode::kDynamicUpdateSlice) {
          return false;
        } else if (user != dus && !user.instruction().IsElementwise() &&
                   user.opcode() != HloOpcode::kBitcast &&
                   user.opcode() != HloOpcode::kTuple) {
          return false;
        }
        if (visited.insert(&user.instruction()).second) {
          q.push(user);
        }
      }
    }

    // This check could probably be relaxed: if code generation is made to use a
    // separate parallel loop for each dynamic slice update, then it shouldn't
    // be necessary for the shape to be the same for all the dynamic slice
    // updates. Note that this equality check purposefully ignores the element
    // type.
    if (Cast<HloDynamicUpdateSliceInstruction>(&dus.instruction())
            ->update()
            ->shape() != update_shape) {
      return false;
    }

    if (fusion != nullptr) {
      ShapeIndex root_index = {};
      if (fusion->IsMultiOutputFusion()) {
        root_index = {i};
      }
      // Get output buffer for the fusion root.
      TF_ASSIGN_OR_RETURN(BufferAllocation::Slice output_buffer,
                          get_allocation_slice(fusion, root_index));

      TF_ASSIGN_OR_RETURN(BufferAllocation::Slice lhs_buffer,
                          get_allocation_slice(&operand.instruction(), {}));
      if (lhs_buffer != output_buffer) {
        return false;
      }
    }
  }

  return true;
}

std::vector<HloInstructionAdaptor> GetOutputDefiningDynamicUpdateSlices(
    absl::Span<HloInstructionAdaptor const> roots) {
  std::vector<HloInstructionAdaptor> dus_ops;
  for (HloInstructionAdaptor root : roots) {
    while (root.opcode() == HloOpcode::kBitcast) {
      root = root.GetOperand(0);
    }

    if (root.opcode() == HloOpcode::kDynamicUpdateSlice) {
      dus_ops.push_back(root);
    }
  }
  return dus_ops;
}

std::optional<TransposeDescription> GetDescriptionForTiledTransposeEmitter(
    const HloInstruction& hero) {
  if (hero.opcode() != HloOpcode::kTranspose) {
    return std::nullopt;
  }

  // We can assume that TransposeDimensionGrouper pass has run, so no need to
  // call GetNormalizedLogicalTransposeShape here.
  absl::InlinedVector<int64_t, 3> permutation(hero.dimensions().begin(),
                                              hero.dimensions().end());
  // A real transpose needs at least 2 transpose dimensions.
  if (permutation.size() < 2) {
    return std::nullopt;
  }
  absl::InlinedVector<int64_t, 3> dimensions(hero.shape().dimensions().begin(),
                                             hero.shape().dimensions().end());
  int64_t operand_most_minor_dim = hero.operand(0)->shape().dimensions().back();
  if (permutation.back() == dimensions.size() - 1) {
    operand_most_minor_dim =
        hero.operand(0)->shape().dimensions(dimensions.size() - 2);
    auto byte_width = primitive_util::ByteWidth(hero.shape().element_type());
    if (byte_width * dimensions.back() <= kMaxBytesInMostMinorDimension &&
        byte_width * dimensions.back() *
                std::min(operand_most_minor_dim,
                         dimensions[dimensions.size() - 2]) >=
            kMinDimensionToTransposeTiled) {
      return TransposeDescription{&hero, dimensions, permutation};
    }
  } else if ((operand_most_minor_dim >= kMinDimensionToTransposeTiled &&
              dimensions.back() >= kMinDimensionToTransposeTiled) ||
             (operand_most_minor_dim >= kMinDimensionToTransposeTiled2 &&
              dimensions.back() >= kMinDimensionToTransposeTiled2 &&
              operand_most_minor_dim * dimensions.back() >=
                  kMinTotalDimensionsToTransposeTiled)) {
    return TransposeDescription{&hero, dimensions, permutation};
  }
  return std::nullopt;
}

bool IsIntermediate(const HloInstruction* instr, int allowed_operand_count) {
  // Number of operands should be in range [1, allowed_operand_count].
  if (instr->operand_count() == 0 ||
      instr->operand_count() > allowed_operand_count) {
    return false;
  }

  if (instr->IsElementwise()) {
    // All elementwise ops are considered intermediate, except for copies that
    // modify the layout. Copies that do not modify the layout are used in
    // CopyFusion.
    if (instr->opcode() == HloOpcode::kCopy) {
      return instr->shape() == instr->operand(0)->shape();
    }
    return true;
  }

  // `instr` is a bitcast or a bitcast-like operation.
  switch (instr->opcode()) {
    case HloOpcode::kBitcast:
      return true;
    case HloOpcode::kReshape:
      return ShapeUtil::ReshapeIsBitcast(instr->operand(0)->shape(),
                                         instr->shape());
    case HloOpcode::kTranspose:
      return ShapeUtil::TransposeIsBitcast(instr->operand(0)->shape(),
                                           instr->shape(), instr->dimensions());
    default:
      return false;
  }
}

namespace {

std::optional<HloInstructionAdaptor> FindNonTrivialHero(
    const HloInstructionAdaptor& root,
    const std::function<bool(const HloInstruction&)>& predicate) {
  std::optional<HloInstructionAdaptor> hero = std::nullopt;
  auto visitor = [&](HloInstructionAdaptor node) {
    if (predicate(node.instruction())) {
      if (hero) {  // Bail out if we found multiple potential heroes.
        hero = std::nullopt;
        return TraversalResult::kInterrupt;
      }
      hero = node;
      return TraversalResult::kSkip;
    }

    if (!IsIntermediate(&node.instruction(), /*allowed_operand_count=*/3)) {
      return TraversalResult::kSkip;
    }
    return TraversalResult::kAdvance;
  };
  HloBfsConsumersFirstTraversal({root}, root.parent(), visitor);
  if (!hero) {
    return std::nullopt;
  }

  // Make sure that no non-elementwise op is reachable from the transpose.
  auto is_nontrivial = [](HloInstructionAdaptor node) {
    return node.instruction().opcode() != HloOpcode::kTuple &&
           node.instruction().opcode() != HloOpcode::kParameter &&
           !IsIntermediate(&node.instruction(),
                           /*allowed_operand_count=*/3);
  };
  bool visit_operands = false;
  if (HloBfsAnyOf(hero->GetUsers(), hero->parent(), is_nontrivial,
                  visit_operands)) {
    return std::nullopt;
  }

  return hero;
}

}  // namespace

HloInstructionAdaptor FindNonTrivialHero(const HloInstructionAdaptor& instr) {
  HloInstructionAdaptor hero = instr;

  // Go up the chain of trivial element-wise(+bitcast, -copy) operations. Note
  // that no memoization is needed due to number of operands constraints: we
  // never have to revisit same nodes.
  while (IsIntermediate(&hero.instruction(), /*allowed_operand_count=*/1) &&
         hero.parent().ContainsInstruction(hero.GetOperand(0))) {
    hero = hero.GetOperand(0);
  }

  // Try a bit harder to find a transpose or concat hero. The shared memory
  // transpose and concat emitters also work if there are elementwise ops with
  // more than 1 operand on the path between root and the root op.
  auto is_transpose = [](const HloInstruction& node) {
    return GetDescriptionForTiledTransposeEmitter(node).has_value();
  };
  if (auto transpose = FindNonTrivialHero(hero, is_transpose)) {
    return *transpose;
  }
  auto is_concatenate = [](const HloInstruction& node) {
    return node.opcode() == HloOpcode::kConcatenate;
  };
  if (auto concatenate = FindNonTrivialHero(hero, is_concatenate)) {
    return *concatenate;
  }
  if (hero.opcode() != HloOpcode::kReduce) {
    return instr;
  }
  return hero;
}

const HloInstruction& FindNonTrivialHero(const HloInstruction& instr) {
  CHECK_NE(instr.opcode(), HloOpcode::kFusion);
  auto fusion_adaptor = HloFusionAdaptor::ForComputation(instr.parent());
  HloInstructionAdaptor instr_adaptor(instr, fusion_adaptor.get());
  return FindNonTrivialHero(instr_adaptor).instruction();
}

absl::StatusOr<DenseDataIntermediate> LiteralToZkxFormat(
    const Literal& literal) {
  PrimitiveType element_type = literal.shape().element_type();
  if (!primitive_util::IsArrayType(element_type)) {
    return absl::InternalError("Unsupported type in LiteralToZkxFormat");
  }

  int64_t byte_size = literal.size_bytes();
  if (primitive_util::IsSubByteNonPredType(element_type)) {
    auto bit_width = primitive_util::BitWidth(element_type);
    std::vector<uint8_t> output(CeilOfRatio<int64_t>(byte_size, 8 / bit_width));
    absl::Span<char> output_span =
        absl::MakeSpan(reinterpret_cast<char*>(output.data()), output.size());
    PackIntN(
        bit_width,
        absl::MakeSpan(reinterpret_cast<const char*>(literal.untyped_data()),
                       byte_size),
        output_span);
    return DenseDataIntermediate::Own(std::move(output));
  }

  return DenseDataIntermediate::Alias(absl::MakeSpan(
      reinterpret_cast<const uint8_t*>(literal.untyped_data()), byte_size));
}

}  // namespace zkx::gpu
