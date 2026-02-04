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

#include "zkx/hlo/ir/hlo_instruction.h"

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/escaping.h"

#include "xla/tsl/lib/gtl/map_util.h"
#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/ir/hlo_op_metadata.h"

namespace zkx {

// static
const HloInstruction::Rare* const HloInstruction::kEmptyRare =
    new HloInstruction::Rare;

namespace {
// Specialization for erasing from PtrVec<T>.
template <typename T>
absl::Status EraseElementFromVector(PtrVec<T>* container, T value) {
  // absl::c_find returns a const_iterator which does not seem to work on
  // gcc 4.8.4, and this breaks the ubuntu/xla_gpu build bot.
  auto it = std::find(container->begin(), container->end(), value);
  TF_RET_CHECK(it != container->end());
  container->erase(it);
  return absl::OkStatus();
}

std::string CustomCallScheduleToString(const CustomCallSchedule& schedule) {
  return absl::AsciiStrToLower(CustomCallSchedule_Name(schedule));
}

std::string CustomCallApiVersionToString(const CustomCallApiVersion& version) {
  return absl::AsciiStrToLower(CustomCallApiVersion_Name(version));
}
}  // namespace

void HloInstruction::Users::Clear() {
  users_.clear();
  user_map_.reset(nullptr);
  DCHECK(CheckInvariants());
}

bool HloInstruction::Users::Contains(const HloInstruction* instruction) const {
  if (user_map_ == nullptr) {
    return std::find(users_.begin(), users_.end(), instruction) != users_.end();
  } else {
    return user_map_->contains(instruction);
  }
}

void HloInstruction::Users::AddUser(HloInstruction* user) {
  if (!Contains(user)) {
    // Create hash table if user list is large.
    if (user_map_ == nullptr && users_.size() >= kMapThreshold) {
      user_map_ =
          std::make_unique<absl::flat_hash_map<const HloInstruction*, int64_t>>(
              users_.size());
      RebuildMap();
      DCHECK(CheckInvariants());
    }

    if (user_map_ != nullptr) {
      user_map_->emplace(user, users_.size());
    }
    users_.push_back(user);
    DCHECK(CheckInvariants());
  }
}

int64_t HloInstruction::Users::UserId(HloInstruction* user) {
  if (user_map_ == nullptr) {
    auto it = std::find(users_.begin(), users_.end(), user);
    CHECK(it != users_.end());
    return it - users_.begin();
  } else {
    auto result = user_map_->find(user);
    CHECK(result != user_map_->end());
    return result->second;
  }
}

void HloInstruction::Users::MaybeRemoveUser(HloInstruction* user) {
  if (Contains(user)) {
    RemoveUser(user);
    DCHECK(CheckInvariants());
  }
}

void HloInstruction::Users::RemoveUser(HloInstruction* user) {
  const int64_t index = UserId(user);
  CHECK_EQ(users_[index], user);

  // Move the last user into the position of the removed user.
  HloInstruction* last = users_.back();

  // Update map if allocated.
  if (user_map_ != nullptr) {
    (*user_map_)[last] = index;
    user_map_->erase(user);
  }

  // Replace found user with last slot from the vector.
  users_[index] = last;
  users_.pop_back();

  DCHECK(CheckInvariants());
}

void HloInstruction::Users::SortInstructionUsers(
    const MappedPtrContainerSorter<HloInstruction>::MapPtrFn& map_fn,
    const Users& sorted_instruction_users) {
  using Sorter = MappedPtrContainerSorter<HloInstruction>;
  auto status = Sorter::Sort(map_fn, Sorter::IndexAfterMappedElementsFn(),
                             sorted_instruction_users.users_, users_);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to sort instruction users: " << status;
  }
  if (user_map_ != nullptr) {
    user_map_->clear();
    RebuildMap();
  }
  DCHECK(CheckInvariants());
}

void HloInstruction::Users::RebuildMap() {
  for (uint64_t i = 0; i < users_.size(); ++i) {
    (*user_map_)[users_[i]] = i;
  }
}

bool HloInstruction::Users::CheckInvariants() {
  if (user_map_ != nullptr) {
    // Avoid quadratic behavior by doing a quick and dirty check on
    // size instead of actually comparing mapped indices.
    CHECK_EQ(users_.size(), user_map_->size());
  }
  return true;
}

void HloInstruction::AppendComputation(HloComputation* computation) {
  // In .cc file since PtrVec<T*>::push_back() wants to check the alignment
  // of T and hlo_instruction.h does not include hlo_computation.h.
  mutable_rare()->called_computations.push_back(computation);
}

HloInstruction* HloInstruction::AddInstruction(
    std::unique_ptr<HloInstruction> derived_instruction) {
  HloInstruction* derived =
      parent()->AddInstruction(std::move(derived_instruction));
  const bool has_prior_sharding = derived->has_sharding();
  SetupDerivedInstruction(derived);
  if (!has_prior_sharding && (derived->opcode() == HloOpcode::kReshape ||
                              derived->opcode() == HloOpcode::kTranspose)) {
    derived->clear_sharding();
  }
  return derived;
}

// static
absl::StatusOr<std::unique_ptr<HloInstruction>> HloInstruction::CreateFromProto(
    const HloInstructionProto& proto,
    const absl::flat_hash_map<int64_t, HloInstruction*>& instruction_map,
    const absl::flat_hash_map<int64_t, HloComputation*>& computation_map,
    bool prohibit_empty_literal) {
  TF_RET_CHECK(!proto.opcode().empty());
  HloOpcode opcode;
  auto opcode_or = StringToHloOpcode(proto.opcode());
  std::optional<ComparisonDirection> comparison_direction;
  if (opcode_or.ok()) {
    opcode = std::move(opcode_or).value();
  } else {
    // Unknown opcode. Try auto-upgrading deprecated "less-than",
    // "greater-than", etc opcodes, which are now rolled into the kCompare
    // opcode.
    if (proto.opcode() == "equal-to") {
      comparison_direction = ComparisonDirection::kEq;
    } else if (proto.opcode() == "not-equal-to") {
      comparison_direction = ComparisonDirection::kNe;
    } else if (proto.opcode() == "greater-than-or-equal-to") {
      comparison_direction = ComparisonDirection::kGe;
    } else if (proto.opcode() == "greater-than") {
      comparison_direction = ComparisonDirection::kGt;
    } else if (proto.opcode() == "less-than-or-equal-to") {
      comparison_direction = ComparisonDirection::kLe;
    } else if (proto.opcode() == "less-than") {
      comparison_direction = ComparisonDirection::kLt;
    }
    if (comparison_direction) {
      opcode = HloOpcode::kCompare;
    } else {
      return absl::InvalidArgumentError(
          absl::StrFormat("Unknown opcode: %s", proto.opcode()));
    }
  }

  TF_RET_CHECK(proto.has_shape());

  std::unique_ptr<HloInstruction> instruction;
  const auto operands = [&instruction_map, &proto](int index) {
    return instruction_map.at(proto.operand_ids(index));
  };
  const auto all_operands = [&instruction_map, &proto]() {
    std::vector<HloInstruction*> result(proto.operand_ids_size());
    std::transform(proto.operand_ids().begin(), proto.operand_ids().end(),
                   result.begin(), [&instruction_map](int64_t operand_id) {
                     return instruction_map.at(operand_id);
                   });
    return result;
  };
  const auto output_to_operand_aliasing = [&proto]() {
    std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
        output_to_operand_aliasing;
    for (const auto& aliasing : proto.output_operand_aliasing()) {
      output_to_operand_aliasing.emplace_back(
          ShapeIndex(aliasing.output_shape_index().begin(),
                     aliasing.output_shape_index().end()),
          std::make_pair(aliasing.operand_index(),
                         ShapeIndex(aliasing.operand_shape_index().begin(),
                                    aliasing.operand_shape_index().end())));
    }
    return output_to_operand_aliasing;
  };
  const auto computations = [&computation_map, &proto](int index) {
    return computation_map.at(proto.called_computation_ids(index));
  };
  const auto all_computations = [&computation_map, &proto]() {
    std::vector<HloComputation*> result(proto.called_computation_ids_size());
    std::transform(proto.called_computation_ids().begin(),
                   proto.called_computation_ids().end(), result.begin(),
                   [&computation_map](int64_t computation_id) {
                     return computation_map.at(computation_id);
                   });
    return result;
  };

  TF_RET_CHECK(
      absl::c_all_of(proto.operand_ids(),
                     [&](int64_t id) { return instruction_map.contains(id); }))
      << proto.name() << " instruction contains invalid operand id(s)";

  TF_RET_CHECK(
      absl::c_all_of(proto.called_computation_ids(),
                     [&](int64_t id) { return computation_map.contains(id); }))
      << proto.name() << " instruction references invalid computation id(s)";

  Shape shape(proto.shape());
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShapeWithOptionalLayout(shape));

  std::optional<int> arity = HloOpcodeArity(opcode);
  if (arity) {
    TF_RET_CHECK(proto.operand_ids_size() == *arity)
        << proto.opcode() << " instruction should have " << *arity
        << " operands but sees " << proto.operand_ids_size();
  }

  std::optional<int64_t> channel_id;
  if (proto.channel_id() > 0) {
    channel_id = proto.channel_id();
  }

  switch (opcode) {
    case HloOpcode::kFft: {
      instruction = CreateFft(shape, all_operands(), proto.fft_type(),
                              proto.fft_length(), proto.fft_do_bit_reverse());
      break;
    }
    case HloOpcode::kMsm: {
      instruction =
          CreateMsm(shape, operands(0), operands(1), proto.window_bits());
      break;
    }
    case HloOpcode::kAsyncStart: {
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "Async start instruction should have 1 called computation but "
             "sees "
          << proto.called_computation_ids_size();
      instruction = CreateAsyncStart(shape, all_operands(), computations(0),
                                     proto.async_execution_thread().empty()
                                         ? kMainExecutionThread
                                         : proto.async_execution_thread());
      break;
    }
    case HloOpcode::kAsyncUpdate: {
      TF_RET_CHECK(proto.operand_ids_size() == 1)
          << "Async update requires one singular operand";
      HloInstruction* prev_op = operands(0);
      TF_RET_CHECK(prev_op->IsAsynchronous())
          << "Async update requires its operand to be an asynchronous op";
      if (!proto.async_execution_thread().empty()) {
        TF_RET_CHECK(proto.async_execution_thread() ==
                     prev_op->async_execution_thread())
            << "Async update should have " << prev_op->async_execution_thread()
            << " async_execution_thread, but sees "
            << proto.async_execution_thread();
      }
      if (!proto.called_computation_ids().empty()) {
        TF_RET_CHECK(computations(0) == prev_op->async_wrapped_computation())
            << "Async update should have "
            << prev_op->async_wrapped_computation()->name()
            << " async_wrapped_computation, but sees "
            << computations(0)->name();
      }
      instruction = CreateAsyncUpdate(shape, prev_op);
      break;
    }
    case HloOpcode::kAsyncDone: {
      TF_RET_CHECK(proto.operand_ids_size() == 1)
          << "Async done requires one singular operand";
      HloInstruction* prev_op = operands(0);
      TF_RET_CHECK(prev_op->IsAsynchronous())
          << "Async done requires its operand to be an asynchronous op";
      if (!proto.async_execution_thread().empty()) {
        TF_RET_CHECK(proto.async_execution_thread() ==
                     prev_op->async_execution_thread())
            << "Async done should have " << prev_op->async_execution_thread()
            << " async_execution_thread, but sees "
            << proto.async_execution_thread();
      }
      if (!proto.called_computation_ids().empty()) {
        TF_RET_CHECK(computations(0) == prev_op->async_wrapped_computation())
            << "Async done should have "
            << prev_op->async_wrapped_computation()->name()
            << " async_wrapped_computation, but sees "
            << computations(0)->name();
      }
      instruction = CreateAsyncDone(shape, prev_op);
      break;
    }
    case HloOpcode::kCopyStart: {
      std::optional<int> cross_program_prefetch_index;
      if (proto.optional_cross_program_prefetch_index_case() ==
          HloInstructionProto::kCrossProgramPrefetchIndex) {
        cross_program_prefetch_index =
            std::make_optional(proto.cross_program_prefetch_index());

        // Silently upgrade HLO protos using the old field.
      } else if (proto.is_cross_program_prefetch()) {
        cross_program_prefetch_index = 0;
      }

      instruction =
          CreateCopyStart(shape, operands(0), cross_program_prefetch_index);
      break;
    }
    case HloOpcode::kCompare: {
      // Auto-upgraded from deprecated opcode skips the following.
      if (!comparison_direction) {
        TF_ASSIGN_OR_RETURN(
            comparison_direction,
            StringToComparisonDirection(proto.comparison_direction()));
      }
      if (proto.has_comparison_primitive_type()) {
        instruction = CreateCompare(shape, operands(0), operands(1),
                                    *comparison_direction,
                                    proto.comparison_primitive_type());
      } else {
        // Allow the specify of comparison type to be optional.
        // The comparison type will be determined by the types of the operands.
        instruction = CreateCompare(shape, operands(0), operands(1),
                                    *comparison_direction);
      }
      break;
    }
    case HloOpcode::kSend:
      instruction = CreateSend(operands(0), operands(1), channel_id,
                               proto.is_host_transfer());
      break;
    case HloOpcode::kSendDone:
      TF_RET_CHECK(DynCast<HloSendInstruction>(operands(0)) != nullptr)
          << "SendDone must take the context operand from Send";
      instruction =
          CreateSendDone(operands(0), channel_id, proto.is_host_transfer());
      break;
    case HloOpcode::kRecv:
      instruction = CreateRecv(shape.tuple_shapes(0), operands(0), channel_id,
                               proto.is_host_transfer());
      break;
    case HloOpcode::kRecvDone:
      TF_RET_CHECK(DynCast<HloRecvInstruction>(operands(0)) != nullptr)
          << "RecvDone must take the context operand from Recv";
      instruction =
          CreateRecvDone(operands(0), channel_id, proto.is_host_transfer());
      break;
    case HloOpcode::kReverse:
      instruction =
          CreateReverse(shape, operands(0),
                        std::vector<int64_t>(proto.dimensions().begin(),
                                             proto.dimensions().end()));
      break;
    case HloOpcode::kBitReverse:
      instruction =
          CreateBitReverse(shape, operands(0),
                           std::vector<int64_t>(proto.dimensions().begin(),
                                                proto.dimensions().end()));
      break;
    case HloOpcode::kConcatenate:
      TF_RET_CHECK(proto.dimensions_size() == 1)
          << "Concatenate instruction should have 1 dimension but sees "
          << proto.dimensions_size();
      instruction =
          CreateConcatenate(shape, all_operands(), proto.dimensions(0));
      break;
    case HloOpcode::kConditional: {
      TF_RET_CHECK(proto.called_computation_ids_size() > 0)
          << "conditional should have at least 1 called computation";
      if (operands(0)->shape().element_type() == PRED) {
        TF_RET_CHECK(proto.called_computation_ids_size() == 2)
            << "conditional should have exactly 2 called computations but got "
            << proto.called_computation_ids_size();
      }
      TF_RET_CHECK(proto.operand_ids_size() ==
                   proto.called_computation_ids_size() + 1)
          << "conditional should have one branch_index operand plus one "
             "operand per called computation but got "
          << proto.operand_ids_size() << " operands for "
          << proto.called_computation_ids_size() << " branch computations";
      auto cond_operands = all_operands();
      instruction =
          CreateConditional(shape, cond_operands[0], all_computations(),
                            absl::MakeSpan(cond_operands).subspan(1));
      break;
    }
    case HloOpcode::kReduce:
      TF_RET_CHECK(proto.operand_ids_size() % 2 == 0)
          << "Reduce instruction should have an even number of operands but "
             "sees "
          << proto.operand_ids_size();
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "Reduce instruction should have 1 called computation but sees "
          << proto.called_computation_ids_size();
      {
        const auto reduce_operands = all_operands();
        auto inputs = absl::MakeSpan(reduce_operands)
                          .subspan(0, reduce_operands.size() / 2);
        auto init_values =
            absl::MakeSpan(reduce_operands)
                .subspan(reduce_operands.size() / 2, reduce_operands.size());
        instruction =
            CreateReduce(shape, inputs, init_values,
                         std::vector<int64_t>(proto.dimensions().begin(),
                                              proto.dimensions().end()),
                         computations(0));
      }
      break;
    case HloOpcode::kSort: {
      TF_RET_CHECK(proto.operand_ids_size() >= 1)
          << "Sort instruction should have at least 1 operand but has "
          << proto.operand_ids_size();
      TF_RET_CHECK(proto.dimensions().size() == 1)
          << "Sort instruction should have 1 dimension";
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "Sort instruction should one called computation but sees "
          << proto.called_computation_ids_size();
      auto sort_operands = all_operands();
      instruction = CreateSort(shape, proto.dimensions(0), all_operands(),
                               computations(0), proto.is_stable());
      break;
    }
    case HloOpcode::kTranspose:
      instruction =
          CreateTranspose(shape, operands(0),
                          std::vector<int64_t>(proto.dimensions().begin(),
                                               proto.dimensions().end()));
      break;
    case HloOpcode::kBroadcast:
      instruction =
          CreateBroadcast(shape, operands(0),
                          std::vector<int64_t>(proto.dimensions().begin(),
                                               proto.dimensions().end()));
      break;
    case HloOpcode::kMap:
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "Map instruction should have 1 called computation but sees "
          << proto.called_computation_ids_size();
      instruction = CreateMap(shape, all_operands(), computations(0));
      break;
    case HloOpcode::kSlice: {
      std::vector<int64_t> slice_starts, slice_limits, slice_strides;
      for (const HloInstructionProto::SliceDimensions& slice_dimensions :
           proto.slice_dimensions()) {
        slice_starts.push_back(slice_dimensions.start());
        slice_limits.push_back(slice_dimensions.limit());
        slice_strides.push_back(slice_dimensions.stride());
      }
      instruction = CreateSlice(shape, operands(0), slice_starts, slice_limits,
                                slice_strides);
      break;
    }
    case HloOpcode::kConstant: {
      // TODO(b/110214922): Revert this to CHECK(proto.has_literal()).
      if (proto.has_literal()) {
        TF_ASSIGN_OR_RETURN(
            auto literal,
            Literal::CreateFromProto(proto.literal(), prohibit_empty_literal));
        instruction = CreateConstant(std::move(literal));
        // Literal's shape may have no/different tiling info.
        TF_RET_CHECK(Shape::Equal().MinorToMajorOnlyInLayout()(
            instruction->shape(), shape))
            << instruction->shape().ToString(true) << " vs "
            << shape.ToString(true);
        *instruction->mutable_shape() = shape;
      } else {
        instruction = std::make_unique<HloConstantInstruction>(shape);
      }
      break;
    }
    case HloOpcode::kFusion: {
      // In the proto, fused computations are held exclusively within the
      // HloInstructionProto and do not appear as an HloComputationProto within
      // the HloModuleProto.
      TF_RET_CHECK(!proto.fusion_kind().empty());
      TF_ASSIGN_OR_RETURN(FusionKind fusion_kind,
                          StringToFusionKind(proto.fusion_kind()));

      // Find the fused computation and set its fusion instruction.
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "Expect 1 called computation for fusion instruction but sees "
          << proto.called_computation_ids_size();
      const int64_t fusion_id = proto.called_computation_ids(0);
      auto* fused_computation =
          tsl::gtl::FindPtrOrNull(computation_map, fusion_id);
      TF_RET_CHECK(fused_computation != nullptr)
          << "No fusion computation with id " << fusion_id;
      instruction =
          CreateFusion(shape, fusion_kind, all_operands(), fused_computation);
      auto fusion_instr = DynCast<HloFusionInstruction>(instruction.get());
      fusion_instr->set_output_to_operand_aliasing(
          output_to_operand_aliasing());
      break;
    }
    case HloOpcode::kParameter:
      instruction =
          CreateParameter(proto.parameter_number(), shape, proto.name());
      if (!proto.parameter_replication().replicated_at_leaf_buffers().empty()) {
        instruction->set_parameter_replicated_at_leaf_buffers(
            proto.parameter_replication().replicated_at_leaf_buffers());
      }
      break;
    case HloOpcode::kGetTupleElement:
      instruction =
          CreateGetTupleElement(shape, operands(0), proto.tuple_index());
      break;
    case HloOpcode::kInfeed: {
      TF_RET_CHECK(shape.IsTuple() &&
                   (ShapeUtil::TupleElementCount(shape) == 2))
          << "Infeed should have a tuple shape with 2 operands, but has: "
          << shape;
      const Shape& data_shape = ShapeUtil::GetTupleElementShape(shape, 0);
      instruction =
          CreateInfeed(data_shape, operands(0), proto.infeed_config());
    } break;
    case HloOpcode::kOutfeed: {
      Shape outfeed_shape(proto.outfeed_shape());
      TF_RETURN_IF_ERROR(
          ShapeUtil::ValidateShapeWithOptionalLayout(outfeed_shape));
      instruction = CreateOutfeed(outfeed_shape, operands(0), operands(1),
                                  proto.outfeed_config());
      break;
    }
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart: {
      std::optional<int64_t> channel_id;
      if (proto.channel_id() > 0) {
        channel_id = proto.channel_id();
      }

      TF_RET_CHECK(proto.dimensions_size() == 1)
          << "AllGather cannot have more than 1 all-gather dimensions";
      int64_t all_gather_dimension = proto.dimensions(0);
      if (opcode == HloOpcode::kAllGather) {
        instruction = CreateAllGather(
            shape, all_operands(), all_gather_dimension,
            CollectiveDeviceList::FromProto(proto), proto.constrain_layout(),
            channel_id, proto.use_global_device_ids());
      } else {
        instruction = CreateAllGatherStart(
            shape, all_operands(), all_gather_dimension,
            CollectiveDeviceList::FromProto(proto), proto.constrain_layout(),
            channel_id, proto.use_global_device_ids());
      }
      break;
    }
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kReduceScatter: {
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "AllReduce should have 1 called computation but sees "
          << proto.called_computation_ids_size();
      TF_RET_CHECK(proto.channel_id() <= 0 || proto.all_reduce_id() <= 0)
          << "AllReduce cannot have both channel_id() and all_reduce_id()";
      std::optional<int64_t> channel_id;
      if (proto.channel_id() > 0) {
        channel_id = proto.channel_id();
      }
      if (proto.all_reduce_id() > 0) {
        channel_id = proto.all_reduce_id();
      }
      CollectiveDeviceList device_list = CollectiveDeviceList::FromProto(proto);
      if (opcode == HloOpcode::kAllReduce) {
        instruction =
            CreateAllReduce(shape, all_operands(), computations(0), device_list,
                            proto.constrain_layout(), channel_id,
                            proto.use_global_device_ids());
      } else if (opcode == HloOpcode::kReduceScatter) {
        TF_RET_CHECK(proto.dimensions_size() == 1)
            << "ReduceScatter cannot have more than 1 scatter dimensions";
        int64_t scatter_dimension = proto.dimensions(0);
        instruction = CreateReduceScatter(
            shape, all_operands(), computations(0), device_list,
            proto.constrain_layout(), channel_id, proto.use_global_device_ids(),
            scatter_dimension);
      } else {
        instruction =
            CreateAllReduceStart(shape, all_operands(), computations(0),
                                 device_list, proto.constrain_layout(),
                                 channel_id, proto.use_global_device_ids());
      }
      break;
    }
    case HloOpcode::kAllToAll: {
      std::optional<int64_t> channel_id;
      if (proto.channel_id() > 0) {
        channel_id = proto.channel_id();
      }
      std::optional<int64_t> split_dimension;
      if (proto.dimensions_size() > 0) {
        TF_RET_CHECK(proto.dimensions_size() == 1)
            << "AllToAll cannot have more than 1 dimension (split dimension)";
        TF_RET_CHECK(all_operands().size() == 1)
            << "AllToAll must have a single operand when the split dimension "
               "is specified";
        split_dimension = proto.dimensions(0);
      }
      instruction = CreateAllToAll(
          shape, all_operands(), CollectiveDeviceList::FromProto(proto),
          proto.constrain_layout(), channel_id, split_dimension);
      break;
    }
    case HloOpcode::kRaggedAllToAll: {
      std::optional<int64_t> channel_id;
      if (proto.channel_id() > 0) {
        channel_id = proto.channel_id();
      }
      TF_RET_CHECK(all_operands().size() == 6)
          << "RaggedAllToAll must have 6 operands";
      instruction = CreateRaggedAllToAll(shape, all_operands(),
                                         CollectiveDeviceList::FromProto(proto),
                                         channel_id);
      break;
    }
    case HloOpcode::kCollectiveBroadcast: {
      std::optional<int64_t> channel_id;
      if (proto.channel_id() > 0) {
        channel_id = proto.channel_id();
      }
      instruction = CreateCollectiveBroadcast(
          shape, all_operands(), CollectiveDeviceList::FromProto(proto), false,
          channel_id);
      break;
    }
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteStart: {
      std::vector<std::pair<int64_t, int64_t>> source_target_pairs(
          proto.source_target_pairs_size());
      std::optional<int64_t> channel_id;
      if (proto.channel_id() > 0) {
        channel_id = proto.channel_id();
      }
      for (int i = 0; i < source_target_pairs.size(); ++i) {
        source_target_pairs[i].first = proto.source_target_pairs(i).source();
        source_target_pairs[i].second = proto.source_target_pairs(i).target();
      }
      if (proto.dynamic_slice_sizes_size() == 0) {
        if (opcode == HloOpcode::kCollectivePermute) {
          instruction = CreateCollectivePermute(
              shape, all_operands(), source_target_pairs, channel_id);
        } else if (opcode == HloOpcode::kCollectivePermuteStart) {
          instruction = CreateCollectivePermuteStart(
              shape, all_operands(), source_target_pairs, channel_id);
        } else {
          LOG(FATAL) << "Expect CollectivePermute or CollectivePermuteStart, "
                     << "but got " << opcode;
        }
      } else {
        std::vector<std::vector<int64_t>> slice_sizes;
        HloInstruction* input = operands(0);
        HloInstruction* input_start_indices = operands(2);
        if (input->shape().IsTuple() &&
            input->shape().tuple_shapes_size() > 1) {
          slice_sizes.resize(input->shape().tuple_shapes_size());
        } else {
          slice_sizes.resize(1);
        }
        int proto_index = 0;
        if (input->shape().IsTuple()) {
          if (input_start_indices->shape()
                  .tuple_shapes(0)
                  .tuple_shapes(0)
                  .IsArray()) {
            slice_sizes.resize(input->shape().tuple_shapes_size());
            for (int i = 0; i < input->shape().tuple_shapes_size(); ++i) {
              slice_sizes[i].resize(
                  input->shape().tuple_shapes(i).dimensions_size());
              for (int j = 0;
                   j < input->shape().tuple_shapes(i).dimensions_size(); ++j) {
                CHECK_GE(proto.dynamic_slice_sizes_size(), proto_index);
                slice_sizes[i][j] = proto.dynamic_slice_sizes(proto_index);
                proto_index += 1;
              }
            }
          } else {
            slice_sizes.resize(
                input->shape().tuple_shapes_size() *
                ShapeUtil::TupleElementCount(
                    input_start_indices->shape().tuple_shapes(0)));
            int slice_sizes_count = 0;
            for (int i = 0; i < input->shape().tuple_shapes_size(); ++i) {
              for (int j = 0;
                   j < ShapeUtil::TupleElementCount(
                           input_start_indices->shape().tuple_shapes(i));
                   ++j) {
                slice_sizes[slice_sizes_count].resize(
                    input->shape().tuple_shapes(i).rank());
                for (int k = 0; k < input->shape().tuple_shapes(i).rank();
                     ++k) {
                  CHECK_GE(proto.dynamic_slice_sizes_size(), proto_index);
                  slice_sizes[slice_sizes_count][k] =
                      proto.dynamic_slice_sizes(proto_index);
                  proto_index += 1;
                }
                slice_sizes_count += 1;
              }
            }
          }
        } else {
          slice_sizes.resize(
              ShapeUtil::TupleElementCount(input_start_indices->shape()));
          if (input_start_indices->shape().tuple_shapes(0).IsTuple()) {
            for (int i = 0;
                 i < ShapeUtil::TupleElementCount(input_start_indices->shape());
                 ++i) {
              slice_sizes[i].resize(input->shape().dimensions_size());
              for (int j = 0; j < input->shape().dimensions_size(); ++j) {
                slice_sizes[i][j] = proto.dynamic_slice_sizes(proto_index);
                proto_index += 1;
              }
            }
          } else {
            slice_sizes.resize(1);
            slice_sizes[0].resize(input->shape().dimensions_size());
            for (int j = 0; j < input->shape().dimensions_size(); ++j) {
              slice_sizes[0][j] = proto.dynamic_slice_sizes(proto_index);
              proto_index += 1;
            }
          }
        }
        if (opcode == HloOpcode::kCollectivePermute) {
          instruction = CreateCollectivePermute(
              shape, operands(0), operands(1), operands(2), operands(3),
              source_target_pairs, slice_sizes, channel_id);
        } else if (opcode == HloOpcode::kCollectivePermuteStart) {
          instruction = CreateCollectivePermuteStart(
              shape, operands(0), operands(1), operands(2), operands(3),
              source_target_pairs, slice_sizes, channel_id);
        } else {
          LOG(FATAL) << "Expect CollectivePermute or CollectivePermuteStart, "
                     << "but got " << opcode;
        }
      }
      break;
    }
    case HloOpcode::kReplicaId: {
      instruction = CreateReplicaId(shape);
      break;
    }
    case HloOpcode::kPartitionId: {
      instruction = CreatePartitionId(shape);
      break;
    }
    case HloOpcode::kReduceWindow: {
      TF_RET_CHECK(proto.operand_ids_size() % 2 == 0)
          << "Reduce window should have an even number of operands but "
             "sees "
          << proto.operand_ids_size();
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "ReduceWindow should have 1 called computation but sees "
          << proto.called_computation_ids_size();
      {
        const auto reduce_operands = all_operands();
        auto inputs = absl::MakeSpan(reduce_operands)
                          .subspan(0, reduce_operands.size() / 2);
        auto init_values =
            absl::MakeSpan(reduce_operands)
                .subspan(reduce_operands.size() / 2, reduce_operands.size());
        instruction = CreateReduceWindow(shape, inputs, init_values,
                                         proto.window(), computations(0));
      }
      break;
    }
    case HloOpcode::kCustomCall: {
      if (proto.constrain_layout()) {
        // A proto RepeatedPtrField cannot be converted to a Span (it is a
        // vector of pointers essentially) so create a vector of shapes to pass
        // in.
        std::vector<Shape> operand_shapes;
        const auto& operand_shapes_with_layout =
            proto.operand_shapes_with_layout();
        operand_shapes.reserve(operand_shapes_with_layout.size());
        for (const ShapeProto& shape_proto : operand_shapes_with_layout) {
          operand_shapes.emplace_back(shape_proto);
        }
        instruction =
            CreateCustomCall(shape, all_operands(), proto.custom_call_target(),
                             operand_shapes, proto.backend_config());
      } else {
        if (proto.called_computation_ids_size() == 1) {
          instruction = CreateCustomCall(shape, all_operands(), computations(0),
                                         proto.custom_call_target(),
                                         proto.backend_config());
        } else if (proto.called_computation_ids_size() > 1) {
          instruction = CreateCustomCall(
              shape, all_operands(), all_computations(),
              proto.custom_call_target(), proto.backend_config());
        } else {
          instruction = CreateCustomCall(shape, all_operands(),
                                         proto.custom_call_target(),
                                         proto.backend_config());
        }
      }
      auto* custom_call_instr =
          Cast<HloCustomCallInstruction>(instruction.get());
      // Note: window_, convolution_dimension_numbers_, feature_group_count_,
      // batch_group_count_, padding_type_, and precision_config_ are omitted
      // as ZKX does not support convolution operations.
      if (proto.has_literal()) {
        TF_ASSIGN_OR_RETURN(
            auto literal,
            Literal::CreateFromProto(proto.literal(), prohibit_empty_literal));
        custom_call_instr->set_literal(std::move(literal));
      }
      custom_call_instr->set_custom_call_has_side_effect(
          proto.custom_call_has_side_effect());
      custom_call_instr->set_output_to_operand_aliasing(
          output_to_operand_aliasing());
      custom_call_instr->set_custom_call_schedule(proto.custom_call_schedule());
      custom_call_instr->set_api_version(proto.custom_call_api_version());
      break;
    }
    case HloOpcode::kPad:
      TF_RET_CHECK(proto.has_padding_config());
      instruction =
          CreatePad(shape, operands(0), operands(1), proto.padding_config());
      break;
    case HloOpcode::kDynamicSlice: {
      std::vector<int64_t> slice_sizes(proto.dynamic_slice_sizes_size());
      absl::c_copy(proto.dynamic_slice_sizes(), slice_sizes.begin());
      TF_RET_CHECK(proto.operand_ids_size() >= 1)
          << "DynamicSlice instruction should have at least 1 operands but "
             "sees "
          << proto.operand_ids_size();
      // TODO(b/118437727): Old form, make the check unconditional.
      if (proto.operand_ids_size() != 2 || operands(1)->shape().rank() != 1) {
        auto expected_operands = 1 + operands(0)->shape().rank();
        TF_RET_CHECK(proto.operand_ids_size() == expected_operands)
            << "DynamicSlice instruction should have " << expected_operands
            << " operands, but has " << proto.operand_ids_size();
      }
      const auto& operand_vector = all_operands();
      instruction = CreateDynamicSlice(
          shape, operands(0), absl::MakeSpan(operand_vector).subspan(1),
          slice_sizes);
      break;
    }
    case HloOpcode::kDynamicUpdateSlice: {
      TF_RET_CHECK(proto.operand_ids_size() >= 2)
          << "DynamicUpdateSlice instruction should have at least 2 operands "
             "but sees "
          << proto.operand_ids_size();
      // TODO(b/118437727): Old form, make the check unconditional.
      if (proto.operand_ids_size() != 3 || operands(2)->shape().rank() != 1) {
        auto expected_operands = 2 + operands(0)->shape().rank();
        TF_RET_CHECK(proto.operand_ids_size() == expected_operands)
            << "DynamicUpdateSlice instruction should have "
            << expected_operands << " operands, but has "
            << proto.operand_ids_size();
      }
      const auto& operand_vector = all_operands();
      instruction =
          CreateDynamicUpdateSlice(shape, operands(0), operands(1),
                                   absl::MakeSpan(operand_vector).subspan(2));
      break;
    }
    case HloOpcode::kGather: {
      TF_RET_CHECK(proto.has_gather_dimension_numbers())
          << "Gather instruction should have GatherDimensionNumbers set.";
      auto gather_dimension_numbers = std::make_unique<GatherDimensionNumbers>(
          proto.gather_dimension_numbers());
      std::vector<int64_t> gather_slice_sizes;
      const auto& slice_sizes = proto.gather_slice_sizes();
      gather_slice_sizes.reserve(slice_sizes.size());
      for (int64_t bound : slice_sizes) {
        gather_slice_sizes.push_back(bound);
      }
      instruction = CreateGather(shape, operands(0), operands(1),
                                 *gather_dimension_numbers, gather_slice_sizes,
                                 proto.indices_are_sorted());
      break;
    }
    case HloOpcode::kScatter: {
      TF_RET_CHECK(proto.has_scatter_dimension_numbers())
          << "Scatter instruction should have ScatterDimensionNumbers set.";
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "Scatter instruction should have 1 called computation but sees "
          << proto.called_computation_ids_size();
      auto scatter_dimension_numbers =
          std::make_unique<ScatterDimensionNumbers>(
              proto.scatter_dimension_numbers());
      auto operands = all_operands();
      auto operand_span = absl::MakeConstSpan(operands);
      auto input_count = operands.size() / 2;
      instruction =
          CreateScatter(shape, operand_span.first(input_count),
                        operands[input_count], operand_span.last(input_count),
                        computations(0), *scatter_dimension_numbers,
                        proto.indices_are_sorted(), proto.unique_indices());
      break;
    }
    case HloOpcode::kIota:
      TF_RET_CHECK(proto.dimensions_size() == 1)
          << "Iota instruction should have 1 dimension but sees "
          << proto.dimensions_size();
      instruction = CreateIota(shape, proto.dimensions(0));
      break;
    case HloOpcode::kDot: {
      int expected_operands =
          HloDotInstruction::kOperands + proto.dot_sparsity_size();
      TF_RET_CHECK(proto.dot_sparsity_size() <= HloDotInstruction::kOperands)
          << "Too many sparse dot descriptors: " << proto.dot_sparsity_size();
      TF_RET_CHECK(proto.operand_ids_size() == expected_operands)
          << proto.opcode() << " instruction should have " << expected_operands
          << " operands but sees " << proto.operand_ids_size();
      TF_RET_CHECK(proto.has_dot_dimension_numbers())
          << "Dot instruction should have dot_dimension_numbers.";
      std::vector<SparsityDescriptor> sparsity(proto.dot_sparsity().begin(),
                                               proto.dot_sparsity().end());
      auto operand_vector = all_operands();
      instruction = std::make_unique<HloDotInstruction>(
          shape, operands(0), operands(1), proto.dot_dimension_numbers(),
          std::move(sparsity),
          absl::MakeSpan(operand_vector).subspan(HloDotInstruction::kOperands));
      break;
    }
    case HloOpcode::kRaggedDot: {
      // TODO(chokobole): Uncomment this. Dependency: HloRaggedDotInstruction
      // int expected_operands = HloRaggedDotInstruction::kOperands;
      // TF_RET_CHECK(proto.operand_ids_size() == expected_operands)
      //     << proto.opcode() << " instruction should have " <<
      //     expected_operands
      //     << " operands but sees " << proto.operand_ids_size();
      // TF_RET_CHECK(proto.has_ragged_dot_dimension_numbers())
      //     << "RaggedDot instruction should have
      //     ragged_dot_dimension_numbers.";
      // TF_RET_CHECK(absl::c_all_of(proto.precision_config().operand_precision(),
      //                             PrecisionConfig::Precision_IsValid));
      // PrecisionConfig precision_config = proto.precision_config();
      // // Only the lhs and rhs have precisions.
      // precision_config.mutable_operand_precision()->Resize(
      //     HloRaggedDotInstruction::kOperands - 1, PrecisionConfig::DEFAULT);
      // auto operand_vector = all_operands();
      // instruction = std::make_unique<HloRaggedDotInstruction>(
      //     shape, operands(0), operands(1), operands(2),
      //     proto.ragged_dot_dimension_numbers(), precision_config);
      return absl::UnimplementedError(
          "HloInstruction::CreateFromProto: RaggedDot not implemented");
      break;
    }
    case HloOpcode::kDomain: {
      // TODO(chokobole): Uncomment this. Dependency: HloDomainInstruction
      // std::shared_ptr<const HloSharding> entry_hlo_sharding;
      // std::shared_ptr<const HloSharding> exit_hlo_sharding;
      // if (proto.has_domain_entry_sharding()) {
      //   TF_ASSIGN_OR_RETURN(
      //       HloSharding sharding,
      //       HloSharding::FromProto(proto.domain_entry_sharding()));
      //   entry_hlo_sharding = std::make_shared<const HloSharding>(sharding);
      // }
      // if (proto.has_domain_exit_sharding()) {
      //   TF_ASSIGN_OR_RETURN(
      //       HloSharding sharding,
      //       HloSharding::FromProto(proto.domain_exit_sharding()));
      //   exit_hlo_sharding = std::make_shared<const HloSharding>(sharding);
      // }
      // instruction = std::make_unique<HloDomainInstruction>(
      //     shape, operands(0),
      //     std::make_unique<ShardingMetadata>(entry_hlo_sharding),
      //     std::make_unique<ShardingMetadata>(exit_hlo_sharding));
      return absl::UnimplementedError(
          "HloInstruction::CreateFromProto: Domain not implemented");
      break;
    }
    case HloOpcode::kGetDimensionSize:
      TF_RET_CHECK(proto.dimensions_size() == 1);
      instruction =
          CreateGetDimensionSize(shape, operands(0), proto.dimensions(0));
      break;
    case HloOpcode::kSetDimensionSize:
      TF_RET_CHECK(proto.dimensions_size() == 1);
      instruction = CreateSetDimensionSize(shape, operands(0), operands(1),
                                           proto.dimensions(0));
      break;
    case HloOpcode::kReshape: {
      int64_t inferred_dimension = -1;
      if (!proto.dimensions().empty()) {
        inferred_dimension = proto.dimensions()[0];
      }
      TF_RET_CHECK(shape.IsArray() && operands(0)->shape().IsArray() &&
                   (operands(0)->shape().is_unbounded_dynamic() ||
                    ShapeUtil::StaticExtentProduct(shape) ==
                        ShapeUtil::StaticExtentProduct(operands(0)->shape())))
          << "shape: " << ShapeUtil::HumanString(shape)
          << " operand: " << ShapeUtil::HumanString(operands(0)->shape());
      instruction = CreateReshape(shape, operands(0), inferred_dimension);
      break;
    }
    case HloOpcode::kDynamicReshape: {
      TF_RET_CHECK(shape.IsArray() && operands(0)->shape().IsArray() &&
                   ShapeUtil::StaticExtentProduct(shape) ==
                       ShapeUtil::StaticExtentProduct(operands(0)->shape()))
          << "shape: " << ShapeUtil::HumanString(shape)
          << " operand: " << ShapeUtil::HumanString(operands(0)->shape());
      const auto& operand_vector = all_operands();
      instruction = CreateDynamicReshape(
          shape, operands(0), absl::MakeSpan(operand_vector).subspan(1));
      break;
    }
    case HloOpcode::kCall: {
      TF_RET_CHECK(proto.called_computation_ids_size() == 1)
          << "Call should have 1 called computation but has "
          << proto.called_computation_ids_size();
      TF_RET_CHECK(!proto.has_dot_dimension_numbers()) << instruction->opcode();

      if (proto.is_composite()) {
        TF_RET_CHECK(proto.has_frontend_attributes())
            << "A composite call op must have frontend attributes";
        auto map = proto.frontend_attributes().map();
        auto name = map.find("composite.name");
        TF_RET_CHECK(name != map.end() && !name->second.empty())
            << "A composite call op must have frontend attributes with key "
               "composite.name whose value is non-empty";

        auto attributes = map.find("composite.attributes");
        TF_RET_CHECK(attributes == map.end() || !attributes->second.empty())
            << "A composite call op must have frontend attributes with key "
               "composite.attributes whose value is default: {} or non-empty";

        auto version_str = map.find("composite.version");
        int64_t version = 0;
        TF_RET_CHECK(
            version_str == map.end() ||
            (absl::SimpleAtoi(version_str->second, &version) && version >= 0))
            << "A composite call op must have frontend attributes with a "
               "composite.version whose value is a non-negative integer but "
               "got: "
            << version_str->second;

        instruction = CreateCompositeCall(
            shape, all_operands(),
            computation_map.at(proto.called_computation_ids()[0]), name->second,
            attributes == map.end() ? "{}" : attributes->second, version);
        instruction->set_output_to_operand_aliasing(
            output_to_operand_aliasing());
      } else {
        instruction = std::make_unique<HloCallInstruction>(
            shape, all_operands(),
            computation_map.at(proto.called_computation_ids()[0]));
        instruction->set_output_to_operand_aliasing(
            output_to_operand_aliasing());
      }
      break;
    }
    default: {
      instruction = absl::WrapUnique(new HloInstruction(opcode, shape));
      if (instruction->opcode() == HloOpcode::kWhile) {
        TF_RET_CHECK(proto.called_computation_ids_size() == 2)
            << "While should have 2 called computation but has "
            << proto.called_computation_ids_size();
        computation_map.at(proto.called_computation_ids(0))
            ->SetWhileCallInstruction(instruction.get());
      }

      for (const int64_t operand_id : proto.operand_ids()) {
        instruction->AppendOperand(instruction_map.at(operand_id));
      }
      for (const int64_t computation_id : proto.called_computation_ids()) {
        instruction->AppendComputation(computation_map.at(computation_id));
      }
      if (instruction->opcode() == HloOpcode::kWhile) {
        instruction->while_body()->SetWhileCallInstruction(instruction.get());
      }

      // clang-format off
      // TODO(chokobole): Uncomment this. Dependency: HloInstructionProto::dot_dimension_numbers
      // clang-format on
      // TF_RET_CHECK(!proto.has_dot_dimension_numbers()) <<
      // instruction->opcode();
      break;
    }
  }

  for (const int64_t predecessor_id : proto.control_predecessor_ids()) {
    TF_RET_CHECK(ContainsKey(instruction_map, predecessor_id))
        << "No instruction with id " << predecessor_id;
    TF_RETURN_IF_ERROR(instruction_map.at(predecessor_id)
                           ->AddControlDependencyTo(instruction.get()));
  }

  TF_RET_CHECK(!proto.name().empty());
  instruction->SetAndSanitizeName(proto.name());
  *instruction->metadata_ = proto.metadata();
  instruction->backend_config_ = BackendConfigWrapper(proto.backend_config());

  TF_RET_CHECK(proto.id() >= 0)
      << "Instruction with negative id: " << proto.id();
  TF_RET_CHECK(proto.id() <= INT_MAX)
      << "Instruction with id > INT_MAX: " << proto.id();
  instruction->unique_id_ = proto.id();

  if (proto.has_sharding()) {
    TF_ASSIGN_OR_RETURN(HloSharding sharding,
                        HloSharding::FromProto(proto.sharding()));
    // To allow for existing Hlo protos to not fail verification, apply tuple
    // sharding normalization.
    sharding = sharding.NormalizeTupleSharding(instruction->shape());
    instruction->set_sharding(sharding);
  }

  if (proto.has_frontend_attributes()) {
    instruction->set_frontend_attributes(proto.frontend_attributes());
  }

  if (proto.has_statistics_viz()) {
    instruction->set_statistics_viz(proto.statistics_viz());
  }

  if (proto.has_original_value()) {
    const OriginalValueProto& original_value_proto = proto.original_value();
    auto original_value = std::make_shared<OriginalValue>(shape);

    for (const auto& leaf : original_value_proto.leaves()) {
      *original_value->mutable_element(ShapeIndex(leaf.leaf_shape_index())) = {
          leaf.instruction_name(), ShapeIndex(leaf.shape_index())};
    }

    instruction->set_original_value(original_value);
  }

  return std::move(instruction);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateParameter(
    int64_t parameter_number, const Shape& shape, std::string_view name) {
  return std::make_unique<HloParameterInstruction>(parameter_number, shape,
                                                   name);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateConstant(
    Literal literal) {
  return std::make_unique<HloConstantInstruction>(std::move(literal));
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateIota(
    const Shape& shape, int64_t iota_dimension) {
  return std::make_unique<HloIotaInstruction>(shape, iota_dimension);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateConstant(
    Literal literal, const Shape& shape) {
  return std::make_unique<HloConstantInstruction>(std::move(literal), shape);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateGetTupleElement(
    const Shape& shape, HloInstruction* operand, int64_t index) {
  return std::make_unique<HloGetTupleElementInstruction>(shape, operand, index);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateGetTupleElement(
    HloInstruction* operand, int64_t index) {
  return std::make_unique<HloGetTupleElementInstruction>(
      operand->shape().tuple_shapes(index), operand, index);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateNary(
    const Shape& shape, HloOpcode opcode,
    absl::Span<HloInstruction* const> operands) {
  if (opcode == HloOpcode::kCopy) {
    // It is impossible to copy an opaque shape, we don't know how big it is.
    CHECK(!shape.IsOpaque());
  }
  auto instruction = absl::WrapUnique(new HloInstruction(opcode, shape));
  for (auto operand : operands) {
    instruction->AppendOperand(operand);
  }
  return instruction;
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateUnary(
    const Shape& shape, HloOpcode opcode, HloInstruction* operand) {
  // Only certain opcodes are supported with CreateUnary: opcodes of unary
  // instructions with no auxiliary fields.
  switch (opcode) {
    case HloOpcode::kAbs:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kBitcast:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kClz:
    case HloOpcode::kCopy:
    case HloOpcode::kCopyDone:
    case HloOpcode::kInverse:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kSign:
      return CreateNary(shape, opcode, {operand});
    default:
      LOG(FATAL) << "Invalid unary instruction opcode " << opcode;
  }
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateBinary(
    const Shape& shape, HloOpcode opcode, HloInstruction* lhs,
    HloInstruction* rhs) {
  // Only certain opcodes are supported with CreateBinary: opcodes of binary
  // instructions with no auxiliary fields.
  switch (opcode) {
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kDivide:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kMsm:
    case HloOpcode::kPower:
    case HloOpcode::kOr:
    case HloOpcode::kRemainder:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kSubtract:
    case HloOpcode::kXor:
      break;
    default:
      LOG(FATAL) << "Invalid binary instruction opcode " << opcode;
  }
  return CreateNary(shape, opcode, {lhs, rhs});
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateTernary(
    const Shape& shape, HloOpcode opcode, HloInstruction* lhs,
    HloInstruction* rhs, HloInstruction* ehs) {
  // Only certain opcodes are supported with CreateTernary: opcodes of ternary
  // instructions with no auxiliary fields.
  switch (opcode) {
    case HloOpcode::kClamp:
    case HloOpcode::kSelect:
      break;
    default:
      LOG(FATAL) << "Invalid ternary instruction opcode " << opcode;
  }
  return CreateNary(shape, opcode, {lhs, rhs, ehs});
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateVariadic(
    const Shape& shape, HloOpcode opcode,
    absl::Span<HloInstruction* const> operands) {
  std::optional<int> arity = HloOpcodeArity(opcode);
  CHECK(!arity.has_value() || arity.value() == operands.size());
  return CreateNary(shape, opcode, operands);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateMap(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* map_computation) {
  return std::make_unique<HloMapInstruction>(shape, operands, map_computation);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateFft(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    FftType fft_type, int64_t fft_length, bool fft_do_bit_reverse) {
  return std::make_unique<HloFftInstruction>(shape, operands, fft_type,
                                             fft_length, fft_do_bit_reverse);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateMsm(
    const Shape& shape, HloInstruction* scalars, HloInstruction* bases,
    uint32_t window_bits) {
  return std::make_unique<HloMsmInstruction>(shape, scalars, bases,
                                             window_bits);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateAsyncStart(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* async_computation,
    std::string_view async_execution_thread) {
  return std::make_unique<HloAsyncStartInstruction>(
      HloOpcode::kAsyncStart, shape, operands, async_computation,
      async_execution_thread);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateAsyncUpdate(
    const Shape& shape, HloInstruction* operand) {
  return std::make_unique<HloAsyncInstruction>(HloOpcode::kAsyncUpdate, shape,
                                               operand);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateAsyncDone(
    const Shape& shape, HloInstruction* operand) {
  return std::make_unique<HloAsyncInstruction>(HloOpcode::kAsyncDone, shape,
                                               operand);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateCopyStart(
    const Shape& shape, HloInstruction* operand,
    std::optional<int> cross_program_prefetch) {
  return std::make_unique<HloCopyStartInstruction>(shape, operand,
                                                   cross_program_prefetch);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateCompare(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    ComparisonDirection direction, std::optional<PrimitiveType> type) {
  return std::make_unique<HloCompareInstruction>(shape, lhs, rhs, direction,
                                                 type);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateDot(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    const DotDimensionNumbers& dimension_numbers,
    std::vector<SparsityDescriptor> sparsity,
    absl::Span<HloInstruction* const> sparse_meta) {
  return std::make_unique<HloDotInstruction>(shape, lhs, rhs, dimension_numbers,
                                             std::move(sparsity), sparse_meta);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateAllGather(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    int64_t all_gather_dimension, const CollectiveDeviceList& device_list,
    bool constrain_layout, const std::optional<int64_t>& channel_id,
    bool use_global_device_ids) {
  return std::make_unique<HloAllGatherInstruction>(
      HloOpcode::kAllGather, shape, operands, all_gather_dimension, device_list,
      constrain_layout, channel_id, use_global_device_ids);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateAllGatherStart(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    int64_t all_gather_dimension, const CollectiveDeviceList& device_list,
    bool constrain_layout, const std::optional<int64_t>& channel_id,
    bool use_global_device_ids) {
  return std::make_unique<HloAllGatherInstruction>(
      HloOpcode::kAllGatherStart, shape, operands, all_gather_dimension,
      device_list, constrain_layout, channel_id, use_global_device_ids);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateAllReduce(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* reduce_computation, const CollectiveDeviceList& device_list,
    bool constrain_layout, const std::optional<int64_t>& channel_id,
    bool use_global_device_ids) {
  return std::make_unique<HloAllReduceInstruction>(
      HloOpcode::kAllReduce, shape, operands, reduce_computation, device_list,
      constrain_layout, channel_id, use_global_device_ids);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateReduceScatter(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* reduce_computation, const CollectiveDeviceList& device_list,
    bool constrain_layout, const std::optional<int64_t>& channel_id,
    bool use_global_device_ids, int64_t scatter_dimension) {
  return std::make_unique<HloReduceScatterInstruction>(
      shape, operands, reduce_computation, device_list, constrain_layout,
      channel_id, use_global_device_ids, scatter_dimension);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateAllReduceStart(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* reduce_computation, const CollectiveDeviceList& device_list,
    bool constrain_layout, const std::optional<int64_t>& channel_id,
    bool use_global_device_ids) {
  return std::make_unique<HloAllReduceInstruction>(
      HloOpcode::kAllReduceStart, shape, operands, reduce_computation,
      device_list, constrain_layout, channel_id, use_global_device_ids);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateAllToAll(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    const CollectiveDeviceList& device_list, bool constrain_layout,
    const std::optional<int64_t>& channel_id,
    const std::optional<int64_t>& split_dimension) {
  return std::make_unique<HloAllToAllInstruction>(shape, operands, device_list,
                                                  constrain_layout, channel_id,
                                                  split_dimension);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateRaggedAllToAll(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    const CollectiveDeviceList& device_list,
    const std::optional<int64_t>& channel_id) {
  return std::make_unique<HloRaggedAllToAllInstruction>(
      shape, operands, device_list, channel_id);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateCollectiveBroadcast(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    const CollectiveDeviceList& device_list, bool constrain_layout,
    const std::optional<int64_t>& channel_id) {
  return std::make_unique<HloCollectiveBroadcastInstruction>(
      HloOpcode::kCollectiveBroadcast, shape, operands, device_list,
      constrain_layout, channel_id);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateCollectivePermute(
    const Shape& shape, HloInstruction* operand,
    const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
    const std::optional<int64_t>& channel_id) {
  return std::make_unique<HloCollectivePermuteInstruction>(
      HloOpcode::kCollectivePermute, shape,
      absl::Span<HloInstruction* const>(&operand, 1), source_target_pairs,
      channel_id);
}

// overloaded function of above CreateCollectivePermute for multiple operands
// static
std::unique_ptr<HloInstruction> HloInstruction::CreateCollectivePermute(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
    const std::optional<int64_t>& channel_id) {
  return std::make_unique<HloCollectivePermuteInstruction>(
      HloOpcode::kCollectivePermute, shape, operands, source_target_pairs,
      channel_id);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateCollectivePermute(
    const Shape& shape, HloInstruction* input, HloInstruction* output,
    HloInstruction* input_start_indices, HloInstruction* output_start_indices,
    absl::Span<const std::pair<int64_t, int64_t>> source_target_pairs,
    absl::Span<const std::vector<int64_t>> slice_sizes,
    const std::optional<int64_t>& channel_id) {
  return std::make_unique<HloCollectivePermuteInstruction>(
      HloOpcode::kCollectivePermute, shape, input, output, input_start_indices,
      output_start_indices, source_target_pairs, slice_sizes, channel_id);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateCollectivePermuteStart(
    const Shape& shape, HloInstruction* operand,
    const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
    const std::optional<int64_t>& channel_id) {
  return std::make_unique<HloCollectivePermuteInstruction>(
      HloOpcode::kCollectivePermuteStart, shape,
      absl::Span<HloInstruction* const>(&operand, 1), source_target_pairs,
      channel_id);
}

// overloaded function of above CreateCollectivePermuteStart for multiple
// operands
// static
std::unique_ptr<HloInstruction> HloInstruction::CreateCollectivePermuteStart(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
    const std::optional<int64_t>& channel_id) {
  return std::make_unique<HloCollectivePermuteInstruction>(
      HloOpcode::kCollectivePermuteStart, shape, operands, source_target_pairs,
      channel_id);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateCollectivePermuteStart(
    const Shape& shape, HloInstruction* input, HloInstruction* output,
    HloInstruction* input_start_indices, HloInstruction* output_start_indices,
    absl::Span<const std::pair<int64_t, int64_t>> source_target_pairs,
    absl::Span<const std::vector<int64_t>> slice_sizes,
    const std::optional<int64_t>& channel_id) {
  return std::make_unique<HloCollectivePermuteInstruction>(
      HloOpcode::kCollectivePermuteStart, shape, input, output,
      input_start_indices, output_start_indices, source_target_pairs,
      slice_sizes, channel_id);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateReplicaId(
    const Shape& shape) {
  CHECK(Shape::Equal().IgnoreLayout()(shape, ShapeUtil::MakeShape(U32, {})))
      << "HloInstruction replica-id must have a shape of u32[], but "
      << shape.ToString() << " is specified";
  return absl::WrapUnique(new HloInstruction(HloOpcode::kReplicaId, shape));
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreatePartitionId(
    const Shape& shape) {
  CHECK(Shape::Equal().IgnoreLayout()(shape, ShapeUtil::MakeShape(U32, {})))
      << "HloInstruction partition-id must have a shape of u32[], but "
      << shape.ToString() << " is specified";
  return absl::WrapUnique(new HloInstruction(HloOpcode::kPartitionId, shape));
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateBitcast(
    const Shape& shape, HloInstruction* operand) {
  auto instruction =
      absl::WrapUnique(new HloInstruction(HloOpcode::kBitcast, shape));
  instruction->AppendOperand(operand);
  return instruction;
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateBitcastConvert(
    const Shape& shape, HloInstruction* operand) {
  auto instruction =
      absl::WrapUnique(new HloInstruction(HloOpcode::kBitcastConvert, shape));
  instruction->AppendOperand(operand);
  return instruction;
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateReduce(
    const Shape& shape, HloInstruction* operand, HloInstruction* init_value,
    absl::Span<const int64_t> dimensions_to_reduce,
    HloComputation* reduce_computation) {
  return absl::WrapUnique(new HloReduceInstruction(
      shape, {operand, init_value}, dimensions_to_reduce, reduce_computation));
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateReduce(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::Span<HloInstruction* const> init_values,
    absl::Span<const int64_t> dimensions_to_reduce,
    HloComputation* reduce_computation) {
  std::vector<HloInstruction*> all_args;
  all_args.reserve(operands.size() * 2);
  all_args.insert(all_args.end(), operands.begin(), operands.end());
  all_args.insert(all_args.end(), init_values.begin(), init_values.end());
  return std::make_unique<HloReduceInstruction>(
      shape, all_args, dimensions_to_reduce, reduce_computation);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateReduce(
    const Shape& shape, HloInstruction* tuple_of_instructions,
    absl::Span<HloInstruction* const> init_values,
    absl::Span<const int64_t> dimensions_to_reduce,
    HloComputation* reduce_computation) {
  if (!tuple_of_instructions->shape().IsTuple()) {
    CHECK_EQ(init_values.size(), 1)
        << "The first input has to be a tuple, or the number of init values "
           "has to be one.";
    return CreateReduce(shape, tuple_of_instructions, init_values[0],
                        dimensions_to_reduce, reduce_computation);
  }
  absl::InlinedVector<HloInstruction*, 4> inputs;
  for (int idx = 0; idx < tuple_of_instructions->shape().tuple_shapes_size();
       idx++) {
    std::unique_ptr<HloInstruction> gte =
        HloInstruction::CreateGetTupleElement(tuple_of_instructions, idx);
    inputs.push_back(
        tuple_of_instructions->parent()->AddInstruction(std::move(gte)));
  }
  return CreateReduce(shape, inputs, init_values, dimensions_to_reduce,
                      reduce_computation);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateReduceWindow(
    const Shape& shape, HloInstruction* operand, HloInstruction* init_value,
    const Window& window, HloComputation* reduce_computation) {
  return std::make_unique<HloReduceWindowInstruction>(
      shape, operand, init_value, window, reduce_computation);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateReduceWindow(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::Span<HloInstruction* const> init_values, const Window& window,
    HloComputation* reduce_computation) {
  return std::make_unique<HloReduceWindowInstruction>(
      shape, operands, init_values, window, reduce_computation);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateInfeed(
    const Shape& infeed_shape, HloInstruction* token_operand,
    const std::string& config) {
  return std::make_unique<HloInfeedInstruction>(infeed_shape, token_operand,
                                                config);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateOutfeed(
    const Shape& outfeed_shape, HloInstruction* operand,
    HloInstruction* token_operand, std::string_view outfeed_config) {
  return std::make_unique<HloOutfeedInstruction>(outfeed_shape, operand,
                                                 token_operand, outfeed_config);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateSend(
    HloInstruction* operand, HloInstruction* token,
    std::optional<int64_t> channel_id, bool is_host_transfer) {
  return std::make_unique<HloSendInstruction>(operand, token, channel_id,
                                              is_host_transfer);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateSendDone(
    HloInstruction* operand, std::optional<int64_t> channel_id,
    bool is_host_transfer) {
  return std::make_unique<HloSendDoneInstruction>(operand, channel_id,
                                                  is_host_transfer);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateRecv(
    const Shape& shape, HloInstruction* token,
    std::optional<int64_t> channel_id, bool is_host_transfer) {
  return std::make_unique<HloRecvInstruction>(shape, token, channel_id,
                                              is_host_transfer);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateRecvDone(
    HloInstruction* operand, std::optional<int64_t> channel_id,
    bool is_host_transfer) {
  return std::make_unique<HloRecvDoneInstruction>(operand, channel_id,
                                                  is_host_transfer);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateReverse(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64_t> dimensions) {
  return std::make_unique<HloReverseInstruction>(shape, operand, dimensions);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateBitReverse(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64_t> dimensions) {
  return std::make_unique<HloBitReverseInstruction>(shape, operand, dimensions);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateTuple(
    absl::Span<HloInstruction* const> elements) {
  std::vector<const Shape*> element_shapes;
  element_shapes.reserve(elements.size());
  for (auto element : elements) {
    element_shapes.push_back(&element->shape());
  }
  Shape tuple_shape = ShapeUtil::MakeTupleShapeWithPtrs(element_shapes);
  return CreateVariadic(tuple_shape, HloOpcode::kTuple, elements);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateGather(
    const Shape& shape, HloInstruction* operand, HloInstruction* start_indices,
    const GatherDimensionNumbers& gather_dim_numbers,
    absl::Span<const int64_t> slice_sizes, bool indices_are_sorted) {
  return std::make_unique<HloGatherInstruction>(shape, operand, start_indices,
                                                gather_dim_numbers, slice_sizes,
                                                indices_are_sorted);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateScatter(
    const Shape& shape, HloInstruction* operand,
    HloInstruction* scatter_indices, HloInstruction* updates,
    HloComputation* update_computation,
    const ScatterDimensionNumbers& scatter_dim_numbers, bool indices_are_sorted,
    bool unique_indices) {
  return absl::WrapUnique(new HloScatterInstruction(
      shape, {operand, scatter_indices, updates}, update_computation,
      scatter_dim_numbers, indices_are_sorted, unique_indices));
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateScatter(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloInstruction* scatter_indices, absl::Span<HloInstruction* const> updates,
    HloComputation* update_computation,
    const ScatterDimensionNumbers& scatter_dim_numbers, bool indices_are_sorted,
    bool unique_indices) {
  absl::InlinedVector<HloInstruction*, 3> args;
  args.reserve(operands.size() + updates.size() + 1);
  absl::c_copy(operands, std::back_inserter(args));
  args.push_back(scatter_indices);
  absl::c_copy(updates, std::back_inserter(args));
  return std::make_unique<HloScatterInstruction>(
      shape, args, update_computation, scatter_dim_numbers, indices_are_sorted,
      unique_indices);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateAfterAll(
    absl::Span<HloInstruction* const> operands) {
  CHECK(!operands.empty());
  auto instruction = absl::WrapUnique(
      new HloInstruction(HloOpcode::kAfterAll, ShapeUtil::MakeTokenShape()));
  for (auto operand : operands) {
    instruction->AppendOperand(operand);
  }
  return instruction;
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateToken() {
  return absl::WrapUnique(
      new HloInstruction(HloOpcode::kAfterAll, ShapeUtil::MakeTokenShape()));
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateWhile(
    const Shape& shape, HloComputation* condition, HloComputation* body,
    HloInstruction* init) {
  auto instruction =
      absl::WrapUnique(new HloInstruction(HloOpcode::kWhile, shape));
  instruction->AppendOperand(init);
  // Body comes before condition computation in the vector.
  instruction->AppendComputation(body);
  instruction->AppendComputation(condition);
  // Set back pointer from body computation to the while call instruction.
  body->SetWhileCallInstruction(instruction.get());
  return instruction;
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateConditional(
    const Shape& shape, HloInstruction* pred,
    HloInstruction* true_computation_arg, HloComputation* true_computation,
    HloInstruction* false_computation_arg, HloComputation* false_computation) {
  auto instruction =
      absl::WrapUnique(new HloInstruction(HloOpcode::kConditional, shape));
  instruction->AppendOperand(pred);
  instruction->AppendOperand(true_computation_arg);
  instruction->AppendOperand(false_computation_arg);
  // In called_computations, the index of true_computation must be 0 and that
  // of false computation must be 1, as defined by kTrueComputationIndex and
  // kFalseComputationIndex.
  instruction->AppendComputation(true_computation);
  instruction->AppendComputation(false_computation);
  // Set back pointer from computations to the conditional instruction.
  true_computation->SetConditionalCallInstruction(instruction.get());
  false_computation->SetConditionalCallInstruction(instruction.get());
  return instruction;
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateConditional(
    const Shape& shape, HloInstruction* branch_index,
    absl::Span<HloComputation* const> branch_computations,
    absl::Span<HloInstruction* const> branch_computation_args) {
  auto instruction =
      absl::WrapUnique(new HloInstruction(HloOpcode::kConditional, shape));
  instruction->AppendOperand(branch_index);
  CHECK_EQ(branch_computations.size(), branch_computation_args.size());
  for (int i = 0; i < branch_computations.size(); ++i) {
    instruction->AppendComputation(branch_computations[i]);
    instruction->AppendOperand(branch_computation_args[i]);
    // Set back pointer from the computation to the conditional instruction.
    branch_computations[i]->SetConditionalCallInstruction(instruction.get());
  }
  return instruction;
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateSlice(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64_t> start_indices,
    absl::Span<const int64_t> limit_indices,
    absl::Span<const int64_t> strides) {
  return std::make_unique<HloSliceInstruction>(shape, operand, start_indices,
                                               limit_indices, strides);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateDynamicSlice(
    const Shape& shape, HloInstruction* operand,
    absl::Span<HloInstruction* const> start_indices,
    absl::Span<const int64_t> slice_sizes) {
  return std::make_unique<HloDynamicSliceInstruction>(
      shape, operand, start_indices, slice_sizes);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateDynamicUpdateSlice(
    const Shape& shape, HloInstruction* operand, HloInstruction* update,
    absl::Span<HloInstruction* const> start_indices) {
  return std::make_unique<HloDynamicUpdateSliceInstruction>(
      shape, operand, update, start_indices);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateConcatenate(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    int64_t dimension) {
  return std::make_unique<HloConcatenateInstruction>(shape, operands,
                                                     dimension);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateConvert(
    const Shape& shape, HloInstruction* operand) {
  auto instruction =
      absl::WrapUnique(new HloInstruction(HloOpcode::kConvert, shape));
  instruction->AppendOperand(operand);
  return instruction;
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateAddDependency(
    HloInstruction* data_operand, HloInstruction* token_operand) {
  auto instruction = absl::WrapUnique(
      new HloInstruction(HloOpcode::kAddDependency, data_operand->shape()));
  instruction->AppendOperand(data_operand);
  instruction->AppendOperand(token_operand);
  return instruction;
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateBroadcast(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64_t> broadcast_dimensions) {
  return std::make_unique<HloBroadcastInstruction>(shape, operand,
                                                   broadcast_dimensions);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateGetDimensionSize(
    const Shape& shape, HloInstruction* operand, int64_t dimension) {
  return std::make_unique<HloGetDimensionSizeInstruction>(shape, operand,
                                                          dimension);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateSetDimensionSize(
    const Shape& shape, HloInstruction* operand, HloInstruction* val,
    int64_t dimension) {
  return std::make_unique<HloSetDimensionSizeInstruction>(shape, operand, val,
                                                          dimension);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreatePad(
    const Shape& shape, HloInstruction* operand, HloInstruction* padding_value,
    const PaddingConfig& padding_config) {
  return std::make_unique<HloPadInstruction>(shape, operand, padding_value,
                                             padding_config);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateReshape(
    const Shape& shape, HloInstruction* operand, int64_t inferred_dimension) {
  CHECK(operand->shape().is_unbounded_dynamic() ||
        ShapeUtil::StaticExtentProduct(shape) ==
            ShapeUtil::StaticExtentProduct(operand->shape()))
      << "shape: " << ShapeUtil::HumanString(shape)
      << " operand: " << ShapeUtil::HumanString(operand->shape());
  return std::make_unique<HloReshapeInstruction>(shape, operand,
                                                 inferred_dimension);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateDynamicReshape(
    const Shape& shape, HloInstruction* data_operand,
    absl::Span<HloInstruction* const> dim_sizes) {
  CHECK_EQ(ShapeUtil::StaticExtentProduct(shape),
           ShapeUtil::StaticExtentProduct(data_operand[0].shape()))
      << "shape: " << ShapeUtil::HumanString(shape)
      << " operand: " << ShapeUtil::HumanString(data_operand[0].shape());
  CHECK_EQ(shape.rank(), dim_sizes.size());
  return std::make_unique<HloDynamicReshapeInstruction>(shape, data_operand,
                                                        dim_sizes);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateTranspose(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64_t> dimensions) {
  return std::make_unique<HloTransposeInstruction>(shape, operand, dimensions);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateSort(
    const Shape& shape, int64_t dimension,
    absl::Span<HloInstruction* const> operands, HloComputation* compare,
    bool is_stable) {
  return std::make_unique<HloSortInstruction>(shape, dimension, operands,
                                              compare, is_stable);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateFusion(
    const Shape& shape, FusionKind fusion_kind, HloInstruction* fused_root,
    std::string_view prefix) {
  return std::make_unique<HloFusionInstruction>(shape, fusion_kind, fused_root,
                                                prefix);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateFusion(
    const Shape& shape, FusionKind fusion_kind,
    absl::Span<HloInstruction* const> operands,
    HloComputation* fusion_computation, std::string_view prefix) {
  return std::make_unique<HloFusionInstruction>(shape, fusion_kind, operands,
                                                fusion_computation, prefix);
}

void HloInstruction::set_single_sharding(const HloSharding& sharding) {
  CHECK(!sharding.IsTuple()) << sharding;
  if (shape().IsTuple()) {
    set_sharding(HloSharding::Tuple(sharding.GetAsShapeTree(shape())));
  } else {
    set_sharding(sharding);
  }
}

void HloInstruction::SetupDerivedInstruction(
    HloInstruction* derived_instruction) const {
  if (sharding_ != nullptr &&
      ShapeUtil::CompatibleKind(shape_, derived_instruction->shape())) {
    // Only copy sharding if the tuple tree shape of the two instruction is
    // compatible because copying it between differently shaped instructions
    // can produce invalid shardings.
    derived_instruction->set_sharding(*sharding_);
  } else if (!ShapeUtil::CompatibleKind(shape_, derived_instruction->shape())) {
    derived_instruction->clear_sharding();
  }
  derived_instruction->set_metadata(*metadata_);
  if (has_rare()) {
    derived_instruction->set_frontend_attributes(frontend_attributes());
    derived_instruction->set_statistics_viz(statistics_viz());
  } else if (derived_instruction->has_rare()) {
    derived_instruction->mutable_rare()->frontend_attributes.Clear();
    derived_instruction->mutable_rare()->statistics_viz.Clear();
  }
  // If the derived instruction has the same opcode as current, then the backend
  // config is also applicable (only if derived instruction doesn't have its own
  // backend config which might be different from the original one).
  if (opcode() == derived_instruction->opcode() && has_backend_config() &&
      !derived_instruction->has_backend_config()) {
    derived_instruction->CopyBackendConfigFrom(this);
  }
}

bool HloInstruction::HasSideEffectNoRecurse() const {
  switch (opcode_) {
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kCollectiveBroadcast:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
      return true;

    case HloOpcode::kAllGather:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllToAll:
    case HloOpcode::kReduceScatter:
      if (Cast<HloCollectiveInstruction>(this)->constrain_layout()) {
        return true;
      }
      [[fallthrough]];
    case HloOpcode::kCollectivePermute:
      // Collective instructions with channel_id are side effecting only if
      // they are used in non-spmd context.
      return Cast<HloChannelInstruction>(this)->channel_id().has_value() &&
             !GetModule()->config().use_spmd_partitioning();

    case HloOpcode::kCustomCall:
      return Cast<HloCustomCallInstruction>(this)
          ->custom_call_has_side_effect();
    default:
      return false;
  }
}

bool HloInstruction::HasSideEffect() const {
  if (HasSideEffectNoRecurse()) {
    return true;
  }
  // Check if any of the called computations has a side effect.
  for (const auto& computation : called_computations()) {
    if (computation->HasSideEffect()) {
      return true;
    }
  }
  return false;
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateCall(
    const Shape& shape, HloInstruction* called_computation_root) {
  return std::make_unique<HloCallInstruction>(shape, called_computation_root);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateCall(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* computation) {
  return std::make_unique<HloCallInstruction>(shape, operands, computation);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateCompositeCall(
    const Shape& shape, HloInstruction* decomposition_root,
    const std::string& name, const std::string& attributes, int64_t version) {
  return std::make_unique<HloCallInstruction>(shape, decomposition_root, name,
                                              attributes, version);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateCompositeCall(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* decomposition, const std::string& name,
    const std::string& attributes, int64_t version) {
  return std::make_unique<HloCallInstruction>(shape, operands, decomposition,
                                              name, attributes, version);
}

std::unique_ptr<HloInstruction> HloInstruction::CreateCustomCall(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    std::string_view custom_call_target, std::string opaque,
    CustomCallApiVersion api_version) {
  return std::make_unique<HloCustomCallInstruction>(
      shape, operands, custom_call_target, std::move(opaque), api_version);
}

std::unique_ptr<HloInstruction> HloInstruction::CreateCustomCall(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* to_apply, std::string_view custom_call_target,
    std::string opaque, CustomCallApiVersion api_version) {
  return std::make_unique<HloCustomCallInstruction>(
      shape, operands, to_apply, custom_call_target, std::move(opaque),
      api_version);
}

std::unique_ptr<HloInstruction> HloInstruction::CreateCustomCall(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::Span<HloComputation* const> called_computations,
    std::string_view custom_call_target, std::string opaque,
    CustomCallApiVersion api_version) {
  return std::make_unique<HloCustomCallInstruction>(
      shape, operands, called_computations, custom_call_target,
      std::move(opaque), api_version);
}

std::unique_ptr<HloInstruction> HloInstruction::CreateCustomCall(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    std::string_view custom_call_target,
    absl::Span<const Shape> operand_shapes_with_layout, std::string opaque,
    CustomCallApiVersion api_version) {
  return std::make_unique<HloCustomCallInstruction>(
      shape, operands, custom_call_target, std::move(opaque),
      operand_shapes_with_layout, api_version);
}

// static
bool HloInstruction::IsThreadIncluded(
    std::string_view execution_thread,
    const absl::flat_hash_set<std::string_view>& execution_threads_set) {
  return execution_threads_set.empty() ||
         execution_threads_set.contains(execution_thread);
}

void HloInstruction::DetachFromOperandsAndUsers() {
  if (cleaned_up_) {
    return;
  }
  cleaned_up_ = true;
  // Detach from operands. An instruction may be repeated as an operand. To
  // avoid calling RemoveUser twice on the same operand, check before remove.
  for (int64_t operand_num = 0; operand_num < operand_count(); ++operand_num) {
    HloInstruction* operand = operands_[operand_num];
    if (operand == nullptr) {
      continue;
    }
    operand->users_.MaybeRemoveUser(this);
    operands_[operand_num] = nullptr;
  }

  // Update users. Set `nullptr` to the corresponding operand slot for users.
  for (auto& user : this->users()) {
    for (int i = 0; i < user->operand_count(); ++i) {
      if (user->operands_[i] == this) {
        user->operands_[i] = nullptr;
      }
    }
  }
}

void HloInstruction::AddSuffixToInstructionName(const std::string_view suffix) {
  // If an instruction is cloned multiple times avoid names like
  // foo.suffix.suffix.suffix. Instead of repeating the suffix add a numeric
  // suffix. Specifically, the clone of foo.suffix is named foo.suffix2, the
  // clone of foo.suffix2 is named foo.suffix3 and so on.
  const std::string dot_suffix = absl::StrCat(".", suffix);
  size_t index = name().rfind(dot_suffix);
  if (index == std::string::npos) {
    // Existing name does not include ".suffix".
    this->name_ = absl::StrCat(name(), dot_suffix);
  } else {
    // Existing name includes ".suffix". Determine if substring after
    // ".suffix" is numeric and should be replaced with an incremented number.
    auto after_suffix = name().substr(index + dot_suffix.size());
    if (after_suffix.empty()) {
      // Existing name ends in ".suffix". New name should end in ".suffix2".
      this->name_ = absl::StrCat(name(), "2");
    } else {
      // If names ends with .suffix[0-9]+ then replace with a suffix with the
      // numeric value incremented.
      int64_t numeric_suffix;
      if (absl::SimpleAtoi(after_suffix, &numeric_suffix)) {
        this->name_ = absl::StrCat(name().substr(0, index), dot_suffix,
                                   numeric_suffix + 1);
      } else {
        // Substring after ".suffix" is non-numeric.
        this->name_ = absl::StrCat(name(), dot_suffix);
      }
    }
  }
}

std::unique_ptr<HloInstruction> HloInstruction::CloneWithNewOperands(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  return CloneWithNewOperands(shape, new_operands, "", context);
}

std::unique_ptr<HloInstruction> HloInstruction::CloneWithNewOperands(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    const std::string& suffix, HloCloneContext* context) const {
  VLOG(3) << "CloneWithNewOperands:\n  " << ToString();
  VLOG(3) << "  new operands:";
  for (const HloInstruction* new_operand : new_operands) {
    VLOG(3) << "    %" << new_operand->name();
  }

  std::unique_ptr<HloInstruction> clone;
  // Explicitly call the factory for the instruction type. This is more robust
  // in the face of code changes than copying fields explicitly. This also
  // properly sets the user fields of the operands.
  switch (opcode_) {
    // Ops migrated to subclasses.
    // TODO(b/80131774): Remove this switch when migration is complete.
    case HloOpcode::kCompare:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllToAll:
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
    case HloOpcode::kBroadcast:
    case HloOpcode::kCollectiveBroadcast:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kConcatenate:
    case HloOpcode::kConstant:
    case HloOpcode::kCopyStart:
    case HloOpcode::kCustomCall:
    case HloOpcode::kDomain:
    case HloOpcode::kDot:
    case HloOpcode::kDynamicReshape:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kFft:
    case HloOpcode::kFusion:
    case HloOpcode::kGather:
    case HloOpcode::kGetDimensionSize:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kMap:
    case HloOpcode::kInfeed:
    case HloOpcode::kIota:
    case HloOpcode::kOutfeed:
    case HloOpcode::kPad:
    case HloOpcode::kParameter:
    case HloOpcode::kRaggedAllToAll:
    case HloOpcode::kRaggedDot:
    case HloOpcode::kRecv:
    case HloOpcode::kBitReverse:
    case HloOpcode::kRecvDone:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kScatter:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kSetDimensionSize:
    case HloOpcode::kSlice:
    case HloOpcode::kSort:
    case HloOpcode::kTranspose:
      clone = CloneWithNewOperandsImpl(shape, new_operands, context);
      break;
    case HloOpcode::kReduceWindow:
    // Unary ops.
    case HloOpcode::kAbs:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kBitcast:
    case HloOpcode::kClz:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kCopy:
    case HloOpcode::kCopyDone:
    case HloOpcode::kInverse:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kSign:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateUnary(shape, opcode_, new_operands[0]);
      break;
    // Binary ops.
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kDivide:
    case HloOpcode::kMultiply:
    case HloOpcode::kSubtract:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMsm:
    case HloOpcode::kOr:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kXor:
      CHECK_EQ(new_operands.size(), 2);
      clone = CreateBinary(shape, opcode_, new_operands[0], new_operands[1]);
      break;
      // Ternary ops.
    case HloOpcode::kClamp:
    case HloOpcode::kSelect:
      CHECK_EQ(new_operands.size(), 3);
      clone = CreateTernary(shape, opcode_, new_operands[0], new_operands[1],
                            new_operands[2]);
      break;
    // Other supported ops.
    case HloOpcode::kAddDependency:
      CHECK_EQ(new_operands.size(), 2);
      clone = CreateAddDependency(new_operands[0], new_operands[1]);
      break;
    case HloOpcode::kAfterAll:
      if (new_operands.empty()) {
        clone = CreateToken();
      } else {
        clone = CreateAfterAll(new_operands);
      }
      break;
    case HloOpcode::kBitcastConvert:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateBitcastConvert(shape, new_operands[0]);
      break;
    case HloOpcode::kCall:
      clone = CreateCall(shape, new_operands, to_apply());
      break;
    case HloOpcode::kConditional:
      CHECK_EQ(new_operands.size(), branch_count() + 1);
      clone = CreateConditional(shape, new_operands[0],
                                absl::MakeSpan(branch_computations()),
                                new_operands.subspan(1));
      break;
    case HloOpcode::kConvert:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateConvert(shape, new_operands[0]);
      break;
    case HloOpcode::kDynamicUpdateSlice:
      clone = CreateDynamicUpdateSlice(shape, new_operands[0], new_operands[1],
                                       new_operands.subspan(2));
      break;
    case HloOpcode::kPartitionId:
      CHECK_EQ(new_operands.size(), 0);
      clone = CreatePartitionId(shape);
      break;
    case HloOpcode::kReplicaId:
      CHECK_EQ(new_operands.size(), 0);
      clone = CreateReplicaId(shape);
      break;
    case HloOpcode::kTuple:
      clone = CreateTuple(new_operands);
      *clone->mutable_shape() = shape;
      break;
    case HloOpcode::kWhile:
      CHECK_EQ(new_operands.size(), 1);
      clone =
          CreateWhile(shape, while_condition(), while_body(), new_operands[0]);
      // Repoint the while body back at the original while instruction.
      // If a context was passed, the body will be cloned and the clone will
      // point to the copied instruction.
      while_body()->SetWhileCallInstruction(const_cast<HloInstruction*>(this));
      break;
    default:
      CHECK(0) << "Unsupported opcode: " << opcode_;
  }
  // SetupDerivedInstruction will setup the precision_config_ field.
  SetupDerivedInstruction(clone.get());
  clone->set_parent(parent_);
  clone->backend_config_ = BackendConfigWrapper(backend_config_);
  // The new instruction's name will be uniquified when it's added to a
  // computation.
  clone->SetAndSanitizeName(name());
  if (context != nullptr) {
    context->MapInstruction(this, clone.get());
    clone->ReplaceCalledComputations([&](HloComputation* callee) {
      return callee->parent() != context->module()
                 ? context->module()->DeepCloneComputation(callee, context)
                 : callee;
    });
    if (opcode() == HloOpcode::kWhile) {
      clone->while_body()->SetWhileCallInstruction(clone.get());
    }
  }

  if (!suffix.empty()) {
    clone->AddSuffixToInstructionName(suffix);
  }
  return clone;
}

std::unique_ptr<HloInstruction> HloInstruction::CloneWithNewShape(
    const Shape& shape, const std::string& suffix,
    HloCloneContext* context) const {
  std::unique_ptr<HloInstruction> clone =
      CloneWithNewOperands(shape, operands_, context);
  if (suffix.empty()) {
    clone->name_.assign(name().begin(), name().end());
  } else {
    clone->AddSuffixToInstructionName(suffix);
  }
  return clone;
}

std::unique_ptr<HloInstruction> HloInstruction::Clone(
    const std::string& suffix, HloCloneContext* context) const {
  std::unique_ptr<HloInstruction> clone =
      CloneWithNewShape(shape_, suffix, context);
  return clone;
}

std::pair<const HloInstruction*, ShapeIndex>
HloInstruction::LatestNonGteAncestorAndIndex() const {
  const HloInstruction* hlo = this;
  ShapeIndex index;
  while (hlo->opcode() == HloOpcode::kGetTupleElement) {
    index.push_back(hlo->tuple_index());
    hlo = hlo->operand(0);
  }

  // We built up index in the reverse order from what we want.
  std::reverse(index.begin(), index.end());

  return {hlo, index};
}

const HloInstruction* HloInstruction::LatestNonGteAncestor() const {
  const HloInstruction* hlo = this;
  while (hlo->opcode() == HloOpcode::kGetTupleElement) {
    hlo = hlo->operand(0);
  }
  return hlo;
}

HloInstruction::InstructionVector HloInstruction::unique_operands() const {
  InstructionVector unique;
  absl::flat_hash_set<const HloInstruction*> seen;
  for (HloInstruction* operand : operands()) {
    if (seen.insert(operand).second) {
      unique.push_back(operand);
    }
  }
  return unique;
}

absl::Status HloInstruction::AddControlDependencyTo(
    HloInstruction* instruction) {
  TF_RET_CHECK(instruction->parent() == parent());
  if (!absl::c_linear_search(control_successors(), instruction)) {
    mutable_rare()->control_successors.push_back(instruction);
    TF_RET_CHECK(!absl::c_linear_search(
        instruction->rare()->control_predecessors, this));
    instruction->mutable_rare()->control_predecessors.push_back(this);
  }
  return absl::OkStatus();
}

absl::Status HloInstruction::RemoveControlDependencyTo(
    HloInstruction* instruction) {
  TF_RET_CHECK(instruction->parent() == parent());
  if (has_rare()) {
    TF_RETURN_IF_ERROR(EraseElementFromVector(
        &mutable_rare()->control_successors, instruction));
  }
  if (instruction->has_rare()) {
    TF_RETURN_IF_ERROR(EraseElementFromVector(
        &instruction->mutable_rare()->control_predecessors, this));
  }
  return absl::OkStatus();
}

absl::Status HloInstruction::DropAllControlDeps() {
  if (has_rare()) {
    for (auto* ctrl_succ : rare()->control_successors) {
      TF_RETURN_IF_ERROR(EraseElementFromVector(
          &ctrl_succ->mutable_rare()->control_predecessors, this));
    }
    for (auto* ctrl_pred : rare()->control_predecessors) {
      TF_RETURN_IF_ERROR(EraseElementFromVector(
          &ctrl_pred->mutable_rare()->control_successors, this));
    }
    Rare* r = mutable_rare();
    r->control_successors.clear();
    r->control_predecessors.clear();
  }
  return absl::OkStatus();
}

absl::Status HloInstruction::SafelyDropAllControlDependencies() {
  // Add all pairs of transitive dependencies from predecessors to successors.
  if (has_rare()) {
    for (HloInstruction* predecessor : rare()->control_predecessors) {
      for (HloInstruction* successor : rare()->control_successors) {
        TF_RETURN_IF_ERROR(predecessor->AddControlDependencyTo(successor));
      }
    }
  }
  TF_RETURN_IF_ERROR(DropAllControlDeps());
  return absl::OkStatus();
}

bool HloInstruction::HasControlDependencies() const {
  const Rare* r = rare();
  return (!r->control_predecessors.empty() || !r->control_successors.empty());
}

absl::Status HloInstruction::CopyAllControlDepsTo(HloInstruction* start,
                                                  HloInstruction* end) const {
  for (auto* ctrl_pred : control_predecessors()) {
    TF_RETURN_IF_ERROR(ctrl_pred->AddControlDependencyTo(start));
  }
  for (auto* ctrl_succ : control_successors()) {
    TF_RETURN_IF_ERROR(end->AddControlDependencyTo(ctrl_succ));
  }
  return absl::OkStatus();
}

bool HloInstruction::IdenticalInternal(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloInstruction*, const HloInstruction*)>
        eq_operands,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations,
    bool layout_sensitive, bool sharding_sensitive,
    bool ignore_channel_id_values,
    bool ignore_commutative_operand_order) const {
  // An instruction is always identical to itself.
  if (this == &other) {
    return true;
  }

  // Identical instruction must have the same opcode, shape, shardings and
  // identical operands.
  if (opcode() != other.opcode()) {
    return false;
  }
  if (!(layout_sensitive ? ShapeUtil::Equal(shape(), other.shape())
                         : ShapeUtil::Compatible(shape(), other.shape()))) {
    return false;
  }
  if (sharding_sensitive && has_sharding() && other.has_sharding() &&
      sharding() != other.sharding()) {
    return false;
  }
  if (operands().size() != other.operands().size()) {
    return false;
  }

  // Check that operands are equal.
  //
  // Use an explicit loop rather than ContainerEquals, because copying around
  // std::functions may be too expensive in some cases.
  if (ignore_commutative_operand_order &&
      HloOpcodeIsBinaryCommutative(opcode())) {
    CHECK_EQ(operand_count(), 2);
    if (!(eq_operands(operand(0), other.operand(0)) &&
          eq_operands(operand(1), other.operand(1))) &&
        !(eq_operands(operand(0), other.operand(1)) &&
          eq_operands(operand(1), other.operand(0)))) {
      return false;
    }
  } else {
    for (size_t i = 0; i < operands().size(); ++i) {
      if (!eq_operands(operand(i), other.operand(i))) {
        return false;
      }
    }
  }

  if (backend_config_ != other.backend_config_) {
    return false;
  }

  if (ignore_channel_id_values) {
    if (auto channel_inst = DynCast<HloChannelInstruction>(this)) {
      return channel_inst->IdenticalSlowPathIgnoringChannelIdValues(
          other, eq_computations);
    }
  }
  return IdenticalSlowPath(other, eq_computations);
}

void HloInstruction::AppendOperand(HloInstruction* operand) {
  if (operand->parent() != nullptr) {
    DCHECK(!operand->parent()->IsMarkedAsDead(operand))
        << "Operand " << operand->name() << " is already marked dead";
  }
  operands_.push_back(operand);
  operand->AddUser(this);
}

void HloInstruction::AppendOperands(
    absl::Span<HloInstruction* const> operands) {
  for (HloInstruction* operand : operands) {
    HloInstruction::AppendOperand(operand);
  }
}

void HloInstruction::RemoveOperandsAtAscendingIndices(
    absl::Span<const int> ascending_indices) {
  if (ascending_indices.empty()) {
    return;
  }
  int next_index = 0;
  int removed_count = 0;
  for (int to_remove : ascending_indices) {
    while (next_index < to_remove) {
      operands_[next_index - removed_count] = operands_[next_index];
      ++next_index;
    }
    CHECK_LT(to_remove, operands_.size());
    ++removed_count;
    ++next_index;
  }
  while (next_index < operands_.size()) {
    operands_[next_index - removed_count] = operands_[next_index];
    ++next_index;
  }
  CHECK_EQ(removed_count, ascending_indices.size());
  operands_.resize(operands_.size() - removed_count);
}

bool HloInstruction::HasConstantOperand() const {
  for (const HloInstruction* operand : operands_) {
    if (operand->IsConstant()) {
      return true;
    }
  }
  return false;
}

bool HloInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  // Perform opcode specific checks.
  switch (opcode()) {
    // The result of these instructions only depend upon their opcode and
    // operands.
    case HloOpcode::kAbs:
    case HloOpcode::kAdd:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAnd:
    case HloOpcode::kBitcast:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kClamp:
    case HloOpcode::kClz:
    case HloOpcode::kConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kCopyDone:
    case HloOpcode::kCopyStart:
    case HloOpcode::kDivide:
    case HloOpcode::kDynamicReshape:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kInverse:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kOr:
    case HloOpcode::kPartitionId:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kReplicaId:
    case HloOpcode::kReshape:
    case HloOpcode::kSelect:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kSign:
    case HloOpcode::kSubtract:
    case HloOpcode::kTuple:
    case HloOpcode::kXor:
      return true;

    // This opcode has complex or special behavior so just return false.
    case HloOpcode::kAddDependency:
    case HloOpcode::kAfterAll:
      return false;

      // Remaining instructions with special values.
    case HloOpcode::kCall:
      return eq_computations(to_apply(), other.to_apply());
    case HloOpcode::kConditional:
      for (int j = 0; j < branch_count(); ++j) {
        if (!eq_computations(branch_computation(j),
                             other.branch_computation(j))) {
          return false;
        }
      }
      return true;
    case HloOpcode::kWhile:
      return (eq_computations(while_body(), other.while_body()) &&
              eq_computations(while_condition(), other.while_condition()));

    // Ops migrated to subclasses should never come to this line.
    // TODO(b/80131774): Remove this switch when migration is complete.
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllToAll:
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
    case HloOpcode::kBitReverse:
    case HloOpcode::kBroadcast:
    case HloOpcode::kCollectiveBroadcast:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kCompare:
    case HloOpcode::kConcatenate:
    case HloOpcode::kConstant:
    case HloOpcode::kCustomCall:
    case HloOpcode::kDomain:
    case HloOpcode::kDot:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kFft:
    case HloOpcode::kFusion:
    case HloOpcode::kGather:
    case HloOpcode::kGetDimensionSize:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kInfeed:
    case HloOpcode::kIota:
    case HloOpcode::kMap:
    case HloOpcode::kMsm:
    case HloOpcode::kOutfeed:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kPad:
    case HloOpcode::kParameter:
    case HloOpcode::kRaggedAllToAll:
    case HloOpcode::kRaggedDot:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kReduce:
    case HloOpcode::kReverse:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kSlice:
    case HloOpcode::kSort:
    case HloOpcode::kTranspose:
    case HloOpcode::kScatter:
    case HloOpcode::kSetDimensionSize:
      LOG(FATAL) << "Base class impl called for opcode with subclass: "
                 << opcode();
  }
  return false;
}

absl::Status HloInstruction::ReplaceUseWith(HloInstruction* user,
                                            HloInstruction* new_producer) {
  TF_RET_CHECK(ShapeUtil::Compatible(shape(), new_producer->shape()))
      << "this shape: " << ShapeUtil::HumanString(shape())
      << ", replacement shape: "
      << ShapeUtil::HumanString(new_producer->shape());
  return ReplaceUseWithDifferentShape(user, new_producer);
}

absl::Status HloInstruction::ReplaceUseWithDifferentShape(
    HloInstruction* user, HloInstruction* new_producer) {
  VLOG(3) << "Replacing uses of " << name() << " in " << user->name()
          << " with " << new_producer->name();

  RemoveUser(user);

  TF_RET_CHECK(absl::c_count(user->operands_, this) >= 0);
  std::replace(user->operands_.begin(), user->operands_.end(), this,
               new_producer);
  new_producer->AddUser(user);
  // Custom fusions may not be able to handle deduplicated operands.
  if (user->opcode() == HloOpcode::kFusion) {
    TF_RETURN_IF_ERROR(
        Cast<HloFusionInstruction>(user)->DeduplicateFusionOperands());
  }
  return absl::OkStatus();
}

absl::Status HloInstruction::ReplaceUseWith(HloInstruction* user,
                                            int operand_number,
                                            HloInstruction* new_producer) {
  TF_RET_CHECK(ShapeUtil::Compatible(shape(), new_producer->shape()))
      << "this shape: " << ShapeUtil::HumanString(shape())
      << ", replacement shape: "
      << ShapeUtil::HumanString(new_producer->shape());
  return ReplaceUseWithDifferentShape(user, operand_number, new_producer);
}

absl::Status HloInstruction::ReplaceUseWithDifferentShape(
    HloInstruction* user, int operand_number, HloInstruction* new_producer) {
  VLOG(3) << "Replacing operand " << operand_number << " of " << name()
          << " in " << user->name() << " with " << new_producer->name();

  if (absl::c_count(user->operands_, this) == 1) {
    RemoveUser(user);
  }

  TF_RET_CHECK(user->operand(operand_number) == this)
      << "Expected operand " << operand_number << " of " << user->ToString()
      << " to be equal to " << ToString();
  user->operands_[operand_number] = new_producer;
  new_producer->AddUser(user);
  return absl::OkStatus();
}

absl::Status HloInstruction::ReplaceOperandWith(int64_t operand_num,
                                                HloInstruction* new_operand) {
  auto old_operand = operand(operand_num);
  TF_RET_CHECK(
      ShapeUtil::Compatible(old_operand->shape(), new_operand->shape()))
      << old_operand->shape() << " is not compatible with "
      << new_operand->shape();
  return ReplaceOperandWithDifferentShape(operand_num, new_operand);
}

absl::Status HloInstruction::ReplaceOperandWithDifferentShape(
    int64_t operand_num, HloInstruction* new_operand) {
  TF_RET_CHECK(operand_num >= 0);
  TF_RET_CHECK(operand_num < operand_count());
  HloInstruction* old_operand = mutable_operand(operand_num);
  if (old_operand == new_operand) {
    return absl::OkStatus();
  }

  operands_[operand_num] = new_operand;

  VLOG(3) << "Replacing operand " << operand_num << " of " << name() << " with "
          << new_operand->name() << ", was " << old_operand->name();

  if (!absl::c_linear_search(operands_, old_operand)) {
    old_operand->RemoveUser(this);
  }
  new_operand->AddUser(this);
  return absl::OkStatus();
}

absl::Status HloInstruction::ReplaceUsesWith(
    absl::Span<HloInstruction* const> users, HloInstruction* new_producer) {
  TF_RET_CHECK(ShapeUtil::Compatible(shape(), new_producer->shape()))
      << shape() << " is not compatible with " << new_producer->shape();
  return ReplaceAllUsesWithDifferentShape(users, new_producer);
}

absl::Status HloInstruction::ReplaceAllUsesWithDifferentShape(
    absl::Span<HloInstruction* const> users, HloInstruction* new_producer) {
  // Make a copy since users span might get mutated during the loop
  std::vector<HloInstruction*> users_vector(users.begin(), users.end());
  for (HloInstruction* user : users_vector) {
    TF_RETURN_IF_ERROR(ReplaceUseWithDifferentShape(user, new_producer));
  }

  if (parent_ && parent_->root_instruction() == this) {
    parent_->set_root_instruction(new_producer,
                                  /*accept_different_shape=*/true);
  }
  return absl::OkStatus();
}

absl::Status HloInstruction::ReplaceAllUsesWith(HloInstruction* new_producer,
                                                std::string_view trigger) {
  auto print_options = HloPrintOptions::ShortParsable()
                           .set_print_operand_shape(true)
                           .set_print_extra_attributes(false);
  TF_RET_CHECK(ShapeUtil::Compatible(shape(), new_producer->shape()))
      << "The shape doesn't match when replacing '" << ToString(print_options)
      << "' with '" << new_producer->ToString(print_options) << "'. " << shape()
      << " is not compatible with " << new_producer->shape() << "\n '"
      << trigger << "' triggered this wrong replacement.";
  return ReplaceAllUsesWithDifferentShape(new_producer);
}

absl::Status HloInstruction::ReplaceAllUsesWithDifferentShape(
    HloInstruction* new_producer) {
  bool new_producer_is_user = false;
  // Make a copy since users span might get mutated during the loop
  std::vector<HloInstruction*> users_vector(users().begin(), users().end());
  for (HloInstruction* user : users_vector) {
    if (user == new_producer) {
      // It's possible that new_producer is a user of this instruction as might
      // be the case when replacing an instruction with a kCopy of itself. In
      // this case, don't do the replacement to avoid creating a cycle in the
      // graph. new_producer remains the only user of this instruction.
      new_producer_is_user = true;
    } else {
      std::replace(user->operands_.begin(), user->operands_.end(), this,
                   new_producer);
      new_producer->AddUser(user);
      if (user->opcode() == HloOpcode::kFusion) {
        TF_RETURN_IF_ERROR(
            Cast<HloFusionInstruction>(user)->DeduplicateFusionOperands());
      }
    }
  }
  users_.Clear();
  if (new_producer_is_user) {
    AddUser(new_producer);
  }
  if (parent_ && parent_->root_instruction() == this) {
    parent_->set_root_instruction(new_producer,
                                  /*accept_different_shape=*/true);
  }

  return absl::OkStatus();
}

bool HloInstruction::IsEffectiveBitcast() const {
  return opcode_ == HloOpcode::kBitcast ||
         (opcode_ == HloOpcode::kTranspose &&
          ShapeUtil::TransposeIsBitcast(operand(0)->shape(), shape(),
                                        dimensions()));
}

HloComputation* HloInstruction::to_apply() const {
  if (has_to_apply()) {
    CHECK_EQ(called_computations().size(), 1)
        << "Expected a to_apply computation for " << opcode();
    return called_computations()[0];
  }
  LOG(FATAL) << "Invalid opcode for to_apply(): " << opcode();
}

void HloInstruction::set_to_apply(HloComputation* computation) {
  if (has_to_apply()) {
    CHECK_EQ(called_computations().size(), 1)
        << "Expected a to_apply computation for " << opcode();
    rare_->called_computations[0] = computation;
    return;
  }
  LOG(FATAL) << "Invalid opcode for to_apply(): " << opcode();
}

bool HloInstruction::has_to_apply() const {
  switch (opcode_) {
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kCall:
    case HloOpcode::kMap:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kScatter:
    case HloOpcode::kSort:
      return true;
    case HloOpcode::kCustomCall:
      // CustomCall can have a to_apply computation, but it is not required to
      // have one.
      return called_computations().size() == 1;
    default:
      return false;
  }
}

HloComputation* HloInstruction::while_condition() const {
  CHECK_EQ(HloOpcode::kWhile, opcode_);
  return called_computations()[kConditionComputationIndex];
}

HloComputation* HloInstruction::while_body() const {
  CHECK_EQ(HloOpcode::kWhile, opcode_);
  return called_computations()[kBodyComputationIndex];
}

void HloInstruction::set_while_condition(HloComputation* computation) {
  CHECK_EQ(HloOpcode::kWhile, opcode_);
  rare_->called_computations[kConditionComputationIndex] = computation;
}

void HloInstruction::set_while_body(HloComputation* computation) {
  CHECK_EQ(HloOpcode::kWhile, opcode_);
  rare_->called_computations[kBodyComputationIndex] = computation;
}

HloInstruction* HloInstruction::while_init() const {
  CHECK_EQ(HloOpcode::kWhile, opcode_);
  return operands_[0];
}

HloComputation* HloInstruction::true_computation() const {
  CHECK_EQ(HloOpcode::kConditional, opcode_);
  CHECK_EQ(PRED, operand(0)->shape().element_type());
  return called_computations()[kTrueComputationIndex];
}

HloComputation* HloInstruction::false_computation() const {
  CHECK_EQ(HloOpcode::kConditional, opcode_);
  CHECK_EQ(PRED, operand(0)->shape().element_type());
  return called_computations()[kFalseComputationIndex];
}

const PtrVec<HloComputation*>& HloInstruction::branch_computations() const {
  CHECK(HloOpcode::kConditional == opcode_);
  return called_computations();
}

int32_t HloInstruction::branch_count() const {
  CHECK(HloOpcode::kConditional == opcode_);
  return called_computations().size();
}

HloComputation* HloInstruction::branch_computation(int32_t b) const {
  CHECK_EQ(HloOpcode::kConditional, opcode_);
  CHECK_GE(b, 0);
  CHECK_LT(b, called_computations().size());
  return called_computations()[b];
}

int32_t HloInstruction::branch_index(HloComputation* computation) const {
  CHECK_EQ(HloOpcode::kConditional, opcode_);
  CHECK_NE(computation, nullptr);
  for (int32_t idx = 0; idx < branch_count(); idx++) {
    if (branch_computation(idx) == computation) {
      return idx;
    }
  }
  LOG(FATAL) << absl::StrFormat("Conditional %s does not contain branch %s",
                                name(), computation->name());
}

void HloInstruction::set_branch_computation(int b,
                                            HloComputation* computation) {
  CHECK_EQ(HloOpcode::kConditional, opcode_);
  rare_->called_computations[b] = computation;
}

int64_t HloInstruction::operand_index(const HloInstruction* target) const {
  for (int64_t i = 0; i < operand_count(); ++i) {
    if (target == operand(i)) {
      return i;
    }
  }
  LOG(FATAL) << "target was not an operand: " << target->ToString();
}

std::vector<int64_t> HloInstruction::operand_indices(
    const HloInstruction* target) const {
  std::vector<int64_t> indices;
  for (int64_t i = 0; i < operand_count(); ++i) {
    if (target == operand(i)) {
      indices.push_back(i);
    }
  }
  if (indices.empty()) {
    LOG(FATAL) << "target was not an operand: " << target->ToString();
  }
  return indices;
}

std::string_view PrintName(std::string_view name, bool print_ids) {
  if (print_ids) {
    return name;
  } else {
    auto dot_position = name.find_first_of('.');
    return name.substr(0, dot_position);
  }
}

namespace {

using DFSStack = absl::InlinedVector<std::pair<int, HloInstruction*>, 16>;

void PrintNameInternal(Printer* printer, std::string_view name,
                       const HloPrintOptions& options) {
  if (options.print_percent()) {
    printer->Append("%");
  }
  printer->Append(PrintName(name, options.print_ids()));
}

std::string PrintCycle(const HloInstruction* child, DFSStack* dfs_stack,
                       bool ignore_control_predecessors) {
  // This set contains HloInstructions from the top of `DFSStack` that might
  // belong to the cycle, i.e. if  DFSStack :=[back,...,child,...,top], then
  // `subgraph` := {child,...,top}.
  absl::flat_hash_set<const HloInstruction*> subgraph;
  while (!dfs_stack->empty() && dfs_stack->back().second != child) {
    subgraph.insert(dfs_stack->back().second);
    dfs_stack->pop_back();
  }
  // Start dfs at `child` and find a cycle with all nodes in `subgraph`.
  absl::flat_hash_set<const HloInstruction*> visited;
  absl::InlinedVector<const HloInstruction*, 16> dfs;
  dfs.push_back(child);
  std::string result;
  while (!dfs.empty() && result.empty()) {
    bool found_next_instr = false;
    auto process_users_or_successors =
        [&](const std::vector<HloInstruction*>& users_or_successors) {
          for (const auto& user : users_or_successors) {
            if (user == child) {
              dfs.push_back(child);
              result = "\n\nDirected cycle:\n  " +
                       absl::StrJoin(
                           dfs, "\n ",
                           [](std::string* out, const HloInstruction* instr) {
                             absl::StrAppend(out, instr->name());
                           });
              return;
            }
            if (!subgraph.contains(user) || visited.contains(user)) {
              continue;
            }
            visited.insert(user);
            dfs.push_back(user);
            found_next_instr = true;
          }
        };
    const HloInstruction* back = dfs.back();
    process_users_or_successors(back->users());
    if (!ignore_control_predecessors) {
      process_users_or_successors(back->control_successors());
    }
    if (!found_next_instr) {
      dfs.pop_back();
    }
  }

  return result;
}

}  // namespace

void HloInstruction::PrintWithCanonicalNameMap(
    Printer* printer, const HloPrintOptions& options,
    CanonicalNameMap* canonical_name_map) const {
  // Logic to print the instruction name (e.g. "%foo = ").
  if (options.canonicalize_instruction_names()) {
    if (options.is_in_nested_computation()) {
      // If we are canonicalizing instruction names and this is a top-level
      // HloInstruction::ToString() call, don't print an instruction name.
      DCHECK(!options.print_percent());  // no need to call PrintNameInternal
      printer->Append(canonical_name_map->LookupOrInsert(unique_id()));
      printer->Append(" = ");
    }
  } else {
    PrintNameInternal(printer, name(), options);
    printer->Append(" = ");
  }

  if (options.print_result_shape()) {
    // Print shape.
    if (options.include_layout_in_shapes()) {
      ShapeUtil::PrintHumanStringWithLayout(printer, shape());
    } else {
      ShapeUtil::PrintHumanString(printer, shape());
    }
    printer->Append(" ");
  }

  // Print opcode, operand(s).
  if (options.syntax_sugar_async_ops() && HloOpcodeIsAsync(opcode()) &&
      (async_wrapped_computation() &&
       async_wrapped_computation()->CanExpandIntoSingleInstruction())) {
    std::string_view suffix = [&]() {
      switch (opcode()) {
        case HloOpcode::kAsyncStart:
          return "-start";
        case HloOpcode::kAsyncUpdate:
          return "-update";
        default:
          CHECK(opcode() == HloOpcode::kAsyncDone)
              << "Unexpected async opcode: " << opcode();
          return "-done";
      }
    }();
    printer->Append(HloOpcodeString(async_wrapped_opcode()));
    printer->Append(suffix);
  } else {
    printer->Append(HloOpcodeString(opcode()));
  }
  printer->Append("(");
  PrintOperandsWithCanonicalNameMap(printer, options, canonical_name_map);
  printer->Append(")");

  // Print additional attributes. If an instruction contains a subcomputation,
  // the subcomputation is also printed here.
  AttributePrinter attr_printer([printer]() {
    printer->Append(", ");
    return printer;
  });
  PrintExtraAttributes(attr_printer, options);

  if (options.print_original_value() && original_value_) {
    printer->Append(", origin={");
    printer->Append(OriginalValueToString(*original_value()));
    printer->Append("}");
  }

  if (options.print_metadata() &&
      (!metadata_->op_type().empty() || !metadata_->op_name().empty() ||
       !metadata_->source_file().empty() ||
       !metadata_->scheduling_name().empty())) {
    printer->Append(", metadata={");
    printer->Append(
        OpMetadataToString(*metadata_, options.print_metadata_only_op_name()));
    printer->Append("}");
  }
  if (options.print_backend_config() && !backend_config_.empty()) {
    std::string_view config = backend_config_.GetRawString();
    // TODO(chokobole): Uncomment this. Dependency: SortJson
    // std::string sorted_config;
    // if (options.sort_backend_config()) {
    //   // Use `value_or` below, because the backend config string isn't
    //   // guaranteed to be a JSON string.
    //   sorted_config = SortJson(config).value_or(std::string(config));
    //   config = sorted_config;
    // }
    printer->Append(", backend_config=");
    // In the common case that the backend-config is valid-ish JSON, the parser
    // doesn't need it delimited by quotes, so we can print it without
    // CEsape'ing.  This is much easier to read.
    // TODO(chokobole): Uncomment this. Dependency: LexesAsJsonDict
    // if (LexesAsJsonDict(config)) {
    //   printer->Append(config);
    // } else {
    printer->Append("\"");
    printer->Append(absl::CEscape(config));
    printer->Append("\"");
    // TODO(chokobole): Uncomment this. Dependency: LexesAsJsonDict
    // }
  }
}

void HloInstruction::PrintOperandsWithCanonicalNameMap(
    Printer* printer, const HloPrintOptions& options,
    CanonicalNameMap* canonical_name_map) const {
  if (operands_.empty()) return;
  absl::Span<HloInstruction* const> slice(operands_);
  constexpr int64_t kMaxOperandsToShowIfCompact = 4;
  if (options.compact_operands() &&
      slice.size() > kMaxOperandsToShowIfCompact) {
    slice.remove_suffix(slice.size() - kMaxOperandsToShowIfCompact);
  }
  auto print_one = [&](const HloInstruction* operand) {
    // If operand is already been deleted, put `null` to the string output.
    if (operand == nullptr) {
      printer->Append("null ");
      return;
    }
    bool add_space = false;
    if (options.print_operand_shape()) {
      if (options.include_layout_in_shapes()) {
        ShapeUtil::PrintHumanStringWithLayout(printer, operand->shape());
      } else {
        ShapeUtil::PrintHumanString(printer, operand->shape());
      }
      add_space = true;
    }
    if (options.canonicalize_instruction_names()) {
      if (options.is_in_nested_computation()) {
        // In a top-level HloInstruction::ToString() call, the operand name is
        // not part of the canonical string.
        DCHECK(!options.print_percent());  // no need to call PrintNameInternal
        if (add_space) printer->Append(" ");
        printer->Append(
            canonical_name_map->LookupOrInsert(operand->unique_id()));
      }
    } else if (options.print_operand_names()) {
      if (add_space) printer->Append(" ");
      PrintNameInternal(printer, operand->name(), options);
    }
  };
  print_one(slice[0]);
  for (int64_t i = 1; i < slice.size(); ++i) {
    if (options.print_operand_index_annotation_interval() != 0 &&
        i % options.print_operand_index_annotation_interval() == 0) {
      printer->Append(absl::StrFormat(", /*index=%lld*/", i));
    } else {
      printer->Append(", ");
    }
    print_one(slice[i]);
  }
  const int64_t remaining = operands_.size() - slice.size();
  if (remaining > 0) {
    printer->Append(", ...(+");
    printer->Append(remaining);
    printer->Append(")");
  }
}

namespace {

bool IsSequentialCall(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kWhile:
      return true;
    default:
      return false;
  }
}

}  // namespace

void HloInstruction::PrintExtraAttributes(
    AttributePrinter& printer, const HloPrintOptions& options) const {
  if (options.print_extra_attributes()) {
    PrintExtraAttributesImpl(printer, options);
  }

  const auto subcomputation_mode = options.print_subcomputation_mode();
  if (subcomputation_mode ==
      HloPrintOptions::PrintSubcomputationMode::kNameOnly) {
    if (opcode() == HloOpcode::kWhile) {
      printer.Next([this, &options](Printer* printer) {
        printer->Append("condition=");
        PrintNameInternal(printer, while_condition()->name(), options);
      });
      printer.Next([this, &options](Printer* printer) {
        printer->Append("body=");
        PrintNameInternal(printer, while_body()->name(), options);
      });
      // TODO(batzor): Uncomment this. Dependency: SelectAndScatter
      // } else if (opcode() == HloOpcode::kSelectAndScatter) {
      //   printer.Next([this, &options](Printer* printer) {
      //     printer->Append("select=");
      //     PrintNameInternal(printer, select()->name(), options);
      //   });
      //   printer.Next([this, &options](Printer* printer) {
      //     printer->Append("scatter=");
      //     PrintNameInternal(printer, scatter()->name(), options);
      //   });
    } else if (opcode() == HloOpcode::kConditional) {
      if (operand(0)->shape().element_type() == PRED) {
        printer.Next([this, &options](Printer* printer) {
          printer->Append("true_computation=");
          PrintNameInternal(printer, true_computation()->name(), options);
        });
        printer.Next([this, &options](Printer* printer) {
          printer->Append("false_computation=");
          PrintNameInternal(printer, false_computation()->name(), options);
        });
      } else {
        printer.Next([this, &options](Printer* printer) {
          printer->Append("branch_computations={");
          AppendJoin(printer, branch_computations(), ", ",
                     [&](Printer* printer, const HloComputation* computation) {
                       PrintNameInternal(printer, computation->name(), options);
                     });
          printer->Append("}");
        });
      }
    } else if (opcode() == HloOpcode::kCall || opcode() == HloOpcode::kMap ||
               opcode() == HloOpcode::kReduceWindow ||
               opcode() == HloOpcode::kReduce ||
               opcode() == HloOpcode::kAllReduce ||
               opcode() == HloOpcode::kReduceScatter ||
               opcode() == HloOpcode::kAllReduceStart ||
               opcode() == HloOpcode::kScatter ||
               opcode() == HloOpcode::kSort) {
      if (!called_computations().empty()) {
        printer.Next([this, &options](Printer* printer) {
          printer->Append("to_apply=");
          PrintNameInternal(printer, to_apply()->name(), options);
        });
      }
      if (opcode() == HloOpcode::kCall && is_composite()) {
        printer.Next(
            [](Printer* printer) { printer->Append("is_composite=true"); });
      }
    } else if (opcode() == HloOpcode::kCustomCall) {
      if (!called_computations().empty()) {
        printer.Next([this, &options](Printer* printer) {
          printer->Append("called_computations={");
          AppendJoin(printer, called_computations(), ", ",
                     [&](Printer* printer, const HloComputation* computation) {
                       PrintNameInternal(printer, computation->name(), options);
                     });
          printer->Append("}");
        });
      }
    } else if (HloOpcodeIsAsync(opcode())) {
      if (opcode() == HloOpcode::kAsyncStart &&
          (!options.syntax_sugar_async_ops() ||
           (async_wrapped_computation() &&
            !async_wrapped_computation()->CanExpandIntoSingleInstruction()))) {
        printer.Next([this, &options](Printer* printer) {
          printer->Append("calls=");
          PrintNameInternal(printer, async_wrapped_computation()->name(),
                            options);
        });
      }
    } else if (!called_computations().empty()) {
      printer.Next([this, &options](Printer* printer) {
        printer->Append("calls=");
        AppendJoin(printer, called_computations(), ", ",
                   [&](Printer* printer, const HloComputation* computation) {
                     PrintNameInternal(printer, computation->name(), options);
                   });
      });
    }
  } else if ((subcomputation_mode ==
              HloPrintOptions::PrintSubcomputationMode::kFullBodies) ||
             (subcomputation_mode == HloPrintOptions::PrintSubcomputationMode::
                                         kNonSequentialBodies &&
              !IsSequentialCall(opcode()))) {
    HloPrintOptions new_options = options;
    new_options.set_is_in_nested_computation(true);
    switch (opcode()) {
      case HloOpcode::kWhile:
        printer.Next([this, &new_options](Printer* printer) {
          printer->Append("condition=\n");
          while_condition()->Print(printer, new_options);
        });
        printer.Next([this, &new_options](Printer* printer) {
          printer->Append("body=\n");
          while_body()->Print(printer, new_options);
        });
        break;
      // TODO(batzor): Uncomment this. Dependency: SelectAndScatter
      // case HloOpcode::kSelectAndScatter:
      //   printer.Next([this, &new_options](Printer* printer) {
      //     printer->Append("select=\n");
      //     select()->Print(printer, new_options);
      //   });
      //   printer.Next([this, &new_options](Printer* printer) {
      //     printer->Append("scatter=\n");
      //     scatter()->Print(printer, new_options);
      //   });
      //   break;
      case HloOpcode::kConditional:
        if (operand(0)->shape().element_type() == PRED) {
          printer.Next([this, &new_options](Printer* printer) {
            printer->Append("true_computation=\n");
            true_computation()->Print(printer, new_options);
          });
          printer.Next([this, &new_options](Printer* printer) {
            printer->Append("false_computation=\n");
            false_computation()->Print(printer, new_options);
          });
        } else {
          printer.Next([this, &new_options](Printer* printer) {
            printer->Append("branch_computations={\n");
            AppendJoin(
                printer, branch_computations(), ",\n",
                [&](Printer* printer, const HloComputation* computation) {
                  computation->Print(printer, new_options);
                });
            printer->Append("\n}");
          });
        }
        break;
      case HloOpcode::kCall:
      case HloOpcode::kMap:
      case HloOpcode::kReduceWindow:
      case HloOpcode::kReduce:
      case HloOpcode::kAllReduce:
      case HloOpcode::kAllReduceStart:
      case HloOpcode::kScatter:
      case HloOpcode::kSort:
        if (!called_computations().empty()) {
          printer.Next([this, &new_options](Printer* printer) {
            printer->Append("to_apply=\n");
            to_apply()->Print(printer, new_options);
          });
        }
        if (opcode() == HloOpcode::kCall && is_composite()) {
          printer.Next(
              [](Printer* printer) { printer->Append("is_composite=true"); });
        }
        break;
      default:
        if (!called_computations().empty()) {
          printer.Next([this, &new_options](Printer* printer) {
            printer->Append("calls=\n");
            AppendJoin(
                printer, called_computations(), ", ",
                [&](Printer* printer, const HloComputation* computation) {
                  computation->Print(printer, new_options);
                });
          });
        }
        break;
    }
  }

  if (has_sharding()) {
    printer.Next([this, &options](Printer* printer) {
      printer->Append("sharding=");
      sharding().Print(printer, options.print_metadata());
    });
  }
  if (!frontend_attributes().map().empty()) {
    printer.Next([this](Printer* printer) {
      AppendCat(printer, "frontend_attributes=",
                FrontendAttributesToString(frontend_attributes()));
    });
  }

  if (opcode() != HloOpcode::kCall) {
    CHECK(!is_composite())
        << "Only kCall instructions should have is_composite set";
  }

  if (options.print_control_dependencies() && !control_predecessors().empty()) {
    printer.Next([this, &options](Printer* printer) {
      printer->Append("control-predecessors={");
      AppendJoin(printer, control_predecessors(), ", ",
                 [&](Printer* printer, HloInstruction* pre) {
                   PrintNameInternal(printer, pre->name(), options);
                 });
      printer->Append("}");
    });
  }

  if (!statistics_viz().statistics().empty()) {
    printer.Next([this](Printer* printer) {
      AppendCat(printer,
                "statistics=", StatisticsVizToString(statistics_viz()));
    });
  }
}

std::vector<std::string> HloInstruction::ExtraAttributesToString(
    const HloPrintOptions& options) const {
  class MultiStringPrinter : public Printer {
   public:
    void Append(const absl::AlphaNum& a) override {
      if (strings_.empty()) {
        strings_.push_back({});
      }
      absl::StrAppend(&strings_.back(), a);
    }

    void Next() { strings_.push_back({}); }

    std::vector<std::string> ConsumeStrings() && { return std::move(strings_); }

   private:
    std::vector<std::string> strings_;
  } multi_string_printer;
  AttributePrinter attr_printer(/*next_printer=*/[&multi_string_printer] {
    multi_string_printer.Next();
    return &multi_string_printer;
  });
  PrintExtraAttributes(attr_printer, options);
  return std::move(multi_string_printer).ConsumeStrings();
}

void HloInstruction::Print(Printer* printer,
                           const HloPrintOptions& options) const {
  CanonicalNameMap new_map;
  PrintWithCanonicalNameMap(printer, options, &new_map);
}

std::string HloInstruction::ToString(const HloPrintOptions& options) const {
  StringPrinter printer;
  Print(&printer, options);
  return std::move(printer).ToString();
}

std::string HloInstruction::ToString() const {
  return ToString(HloPrintOptions::Default());
}

bool HloInstruction::IsOpElementwise(HloOpcode opcode) {
  switch (opcode) {
    // Unary elementwise operations.
    case HloOpcode::kAbs:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kClz:
    case HloOpcode::kConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kInverse:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kSign:
      return true;

    // Binary elementwise operations, the same as in IsElementwiseBinary().
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kCompare:
    case HloOpcode::kDivide:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kOr:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kSubtract:
    case HloOpcode::kXor:
      return true;

    // Ternary elementwise operations.
    case HloOpcode::kClamp:
    case HloOpcode::kSelect:
      return true;

    default:
      return false;
  }
}

bool HloInstruction::IsElementwiseImpl(
    const std::optional<int64_t>& operand_idx) const {
  if (opcode_ == HloOpcode::kDynamicUpdateSlice) {
    return operand_idx.has_value() && operand_idx.value() == 0;
  }
  if (opcode_ == HloOpcode::kBitcastConvert &&
      primitive_util::BitWidth(shape_.element_type()) !=
          primitive_util::BitWidth(operands_[0]->shape().element_type())) {
    return false;
  }
  return IsOpElementwise(opcode_);
}

bool HloInstruction::IsCrossModuleAllReduce() const {
  if (opcode() == HloOpcode::kAllReduce ||
      opcode() == HloOpcode::kAllReduceStart) {
    return channel_id() != std::nullopt;
  } else if (opcode() == HloOpcode::kAllReduceDone) {
    CHECK_EQ(operand_count(), 1);
    const HloInstruction* operand = this->operand(0);
    CHECK_EQ(operand->opcode(), HloOpcode::kAllReduceStart);
    return operand->channel_id() != std::nullopt;
  }
  return false;
}

std::string FrontendAttributesToString(
    const FrontendAttributes& frontend_attributes) {
  std::vector<std::pair<std::string, std::string>> sorted_attributes(
      frontend_attributes.map().begin(), frontend_attributes.map().end());
  absl::c_sort(sorted_attributes);
  const auto formatter = [](std::string* out,
                            const std::pair<std::string, std::string>& item) {
    // TODO(chokobole): Uncomment this. Dependency: LexesAsJsonDict
    // if (LexesAsJsonDict(item.second)) {
    //   absl::StrAppend(out, item.first, "=", item.second);
    // } else {
    absl::StrAppend(out, item.first, "=\"", item.second, "\"");
    // TODO(chokobole): Uncomment this. Dependency: LexesAsJsonDict
    // }
  };
  return absl::StrFormat("{%s}",
                         absl::StrJoin(sorted_attributes, ",", formatter));
}

std::string HloInstruction::ToShortString() const {
  return absl::StrCat(
      "%", name(), " = ", HloOpcodeString(opcode()), "(",
      absl::StrJoin(operands_, ", ",
                    [](std::string* out, HloInstruction* operand) {
                      absl::StrAppend(out, "%", operand->name());
                    }),
      ")");
}

HloInstructionProto HloInstruction::ToProto() const {
  HloInstructionProto proto;
  CHECK_NE(unique_id_, -1)
      << "This instruction does not have a valid id. Please make sure the "
         "instruction is inside a module before dumping it.";
  proto.set_id(unique_id_);
  proto.set_name(name_);
  *proto.mutable_opcode() = std::string(HloOpcodeString(opcode_));
  *proto.mutable_shape() = shape_.ToProto();
  for (const HloInstruction* operand : operands_) {
    proto.add_operand_ids(operand->unique_id());
  }
  for (const HloInstruction* control : control_predecessors()) {
    proto.add_control_predecessor_ids(control->unique_id());
  }

  *proto.mutable_metadata() = *metadata_;
  proto.set_backend_config(backend_config_.GetRawString());
  if (opcode() != HloOpcode::kFusion) {
    for (const HloComputation* computation : called_computations()) {
      proto.add_called_computation_ids(computation->unique_id());
    }
  }

  if (has_sharding()) {
    *proto.mutable_sharding() = sharding().ToProto();
  }

  *proto.mutable_frontend_attributes() = frontend_attributes();
  proto.set_is_composite(is_composite());

  *proto.mutable_statistics_viz() = statistics_viz();

  if (original_value_) {
    *proto.mutable_original_value() = OriginalValueToProto(*original_value_);
  }

  return proto;
}

std::string HloInstruction::ToCategory() const {
  if (opcode() == HloOpcode::kTranspose || opcode() == HloOpcode::kCopy ||
      opcode() == HloOpcode::kReshape ||
      opcode() == HloOpcode::kDynamicReshape) {
    return "data formatting";
  }

  if (IsElementwise()) {
    return "non-fusion elementwise";
  }

  return std::string(HloOpcodeString(opcode()));
}

bool HloInstruction::IsFused() const {
  return parent_ != nullptr && parent_->IsFusionComputation();
}

bool HloInstruction::IsCustomCall(std::string_view target) const {
  return opcode() == HloOpcode::kCustomCall && custom_call_target() == target;
}

bool HloInstruction::IsCustomCall(
    absl::Span<const std::string_view> targets) const {
  return opcode() == HloOpcode::kCustomCall &&
         absl::c_linear_search(targets, custom_call_target());
}

bool HloInstruction::IsInputFusion() const {
  return opcode() == HloOpcode::kFusion && fusion_kind() == FusionKind::kInput;
}

bool HloInstruction::IsLoopFusion() const {
  return opcode() == HloOpcode::kFusion && fusion_kind() == FusionKind::kLoop;
}

bool HloInstruction::IsOutputFusion() const {
  return opcode() == HloOpcode::kFusion && fusion_kind() == FusionKind::kOutput;
}

bool HloInstruction::IsCustomFusion() const {
  return opcode() == HloOpcode::kFusion && fusion_kind() == FusionKind::kCustom;
}

bool HloInstruction::IsFusible() const {
  // Some kinds of instructions don't make sense to fuse.
  switch (opcode_) {
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kDomain:
    case HloOpcode::kParameter:
    case HloOpcode::kWhile:
      return false;
    // Fusions are always fusible.
    case HloOpcode::kFusion:
    // Side effecting reduce would be invalid HLO.
    case HloOpcode::kMap:
    case HloOpcode::kReduce:
      return true;
    // TODO(chokobole): Uncomment this. Dependency: HloOpcode::kRng
    // case HloOpcode::kRng:
    //   return user_count() <= 1;
    // Side effecting instructions cannot be fused.
    default:
      return !HasSideEffect();
  }
}

HloInstruction::HloInstruction(HloOpcode opcode, const Shape& shape)
    : unique_id_(-1),
      index_in_parent_(~0u),
      opcode_(opcode),
      is_default_config_(false),
      cleaned_up_(false),
      marked_as_dead_(false),
      is_root_(false),
      shape_(shape),
      name_(HloOpcodeString(opcode)) {
  DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(shape_));
}

template <typename HloInstructionPtr>
absl::Status HloInstruction::Visit(
    DfsHloVisitorBase<HloInstructionPtr>* visitor) {
  switch (opcode_) {
    case HloOpcode::kAbs:
      return visitor->HandleAbs(this);
    case HloOpcode::kAdd:
      return visitor->HandleAdd(this);
    case HloOpcode::kAnd:
      return visitor->HandleAnd(this);
    case HloOpcode::kAddDependency:
      return visitor->HandleAddDependency(this);
    case HloOpcode::kAfterAll:
      return visitor->HandleAfterAll(this);
    case HloOpcode::kAllGather:
      return visitor->HandleAllGather(this);
    case HloOpcode::kAllGatherDone:
      return visitor->HandleAllGatherDone(this);
    case HloOpcode::kAllGatherStart:
      return visitor->HandleAllGatherStart(this);
    case HloOpcode::kAllReduce:
      return visitor->HandleAllReduce(this);
    case HloOpcode::kAllReduceDone:
      return visitor->HandleAllReduceDone(this);
    case HloOpcode::kAllReduceStart:
      return visitor->HandleAllReduceStart(this);
    case HloOpcode::kAllToAll:
      return visitor->HandleAllToAll(this);
    case HloOpcode::kAsyncDone:
      return visitor->HandleAsyncDone(this);
    case HloOpcode::kAsyncStart:
      return visitor->HandleAsyncStart(this);
    case HloOpcode::kAsyncUpdate:
      return visitor->HandleAsyncUpdate(this);
    case HloOpcode::kBitcast:
      return visitor->HandleBitcast(this);
    case HloOpcode::kBitcastConvert:
      return visitor->HandleBitcastConvert(this);
    case HloOpcode::kBitReverse:
      return visitor->HandleBitReverse(this);
    case HloOpcode::kBroadcast:
      return visitor->HandleBroadcast(this);
    case HloOpcode::kCall:
      return visitor->HandleCall(this);
    case HloOpcode::kClamp:
      return visitor->HandleClamp(this);
    case HloOpcode::kClz:
      return visitor->HandleClz(this);
    case HloOpcode::kCollectiveBroadcast:
      return visitor->HandleCollectiveBroadcast(this);
    case HloOpcode::kCollectivePermute:
      return visitor->HandleCollectivePermute(this);
    case HloOpcode::kCollectivePermuteDone:
      return visitor->HandleCollectivePermuteDone(this);
    case HloOpcode::kCollectivePermuteStart:
      return visitor->HandleCollectivePermuteStart(this);
    case HloOpcode::kConditional:
      return visitor->HandleConditional(this);
    case HloOpcode::kCopyDone:
      return visitor->HandleCopyDone(this);
    case HloOpcode::kCopyStart:
      return visitor->HandleCopyStart(this);
    case HloOpcode::kCustomCall:
      return visitor->HandleCustomCall(this);
    case HloOpcode::kCompare:
      return visitor->HandleCompare(this);
    case HloOpcode::kConcatenate:
      return visitor->HandleConcatenate(this);
    case HloOpcode::kConstant:
      return visitor->HandleConstant(this);
    case HloOpcode::kConvert:
      return visitor->HandleConvert(this);
    case HloOpcode::kCopy:
      return visitor->HandleCopy(this);
    case HloOpcode::kDivide:
      return visitor->HandleDivide(this);
    case HloOpcode::kDomain:
      return visitor->HandleDomain(this);
    case HloOpcode::kDot:
      return visitor->HandleDot(this);
    case HloOpcode::kDynamicReshape:
      return visitor->HandleDynamicReshape(this);
    case HloOpcode::kDynamicSlice:
      return visitor->HandleDynamicSlice(this);
    case HloOpcode::kDynamicUpdateSlice:
      return visitor->HandleDynamicUpdateSlice(this);
    case HloOpcode::kFft:
      return visitor->HandleFft(this);
    case HloOpcode::kFusion:
      return visitor->HandleFusion(this);
    case HloOpcode::kGather:
      return visitor->HandleGather(this);
    case HloOpcode::kGetDimensionSize:
      return visitor->HandleGetDimensionSize(this);
    case HloOpcode::kGetTupleElement:
      return visitor->HandleGetTupleElement(this);
    case HloOpcode::kInfeed:
      return visitor->HandleInfeed(this);
    case HloOpcode::kIota:
      return visitor->HandleIota(this);
    case HloOpcode::kInverse:
      return visitor->HandleInverse(this);
    case HloOpcode::kMap:
      return visitor->HandleMap(this);
    case HloOpcode::kMaximum:
      return visitor->HandleMaximum(this);
    case HloOpcode::kMinimum:
      return visitor->HandleMinimum(this);
    case HloOpcode::kMsm:
      return visitor->HandleMsm(this);
    case HloOpcode::kMultiply:
      return visitor->HandleMultiply(this);
    case HloOpcode::kNegate:
      return visitor->HandleNegate(this);
    case HloOpcode::kNot:
      return visitor->HandleNot(this);
    case HloOpcode::kOptimizationBarrier:
      return visitor->HandleOptimizationBarrier(this);
    case HloOpcode::kOr:
      return visitor->HandleOr(this);
    case HloOpcode::kOutfeed:
      return visitor->HandleOutfeed(this);
    case HloOpcode::kPad:
      return visitor->HandlePad(this);
    case HloOpcode::kParameter:
      return visitor->HandleParameter(this);
    case HloOpcode::kPartitionId:
      return visitor->HandlePartitionId(this);
    case HloOpcode::kPower:
      return visitor->HandlePower(this);
    case HloOpcode::kPopulationCount:
      return visitor->HandlePopulationCount(this);
    case HloOpcode::kRaggedAllToAll:
      return visitor->HandleRaggedAllToAll(this);
    case HloOpcode::kRaggedDot:
      return visitor->HandleRaggedDot(this);
    case HloOpcode::kRecv:
      return visitor->HandleRecv(this);
    case HloOpcode::kRecvDone:
      return visitor->HandleRecvDone(this);
    case HloOpcode::kReduce:
      return visitor->HandleReduce(this);
    case HloOpcode::kReduceWindow:
      return visitor->HandleReduceWindow(this);
    case HloOpcode::kReduceScatter:
      return visitor->HandleReduceScatter(this);
    case HloOpcode::kRemainder:
      return visitor->HandleRemainder(this);
    case HloOpcode::kReplicaId:
      return visitor->HandleReplicaId(this);
    case HloOpcode::kReshape:
      return visitor->HandleReshape(this);
    case HloOpcode::kReverse:
      return visitor->HandleReverse(this);
    case HloOpcode::kScatter:
      return visitor->HandleScatter(this);
    case HloOpcode::kSelect:
      return visitor->HandleSelect(this);
    case HloOpcode::kSend:
      return visitor->HandleSend(this);
    case HloOpcode::kSendDone:
      return visitor->HandleSendDone(this);
    case HloOpcode::kSetDimensionSize:
      return visitor->HandleSetDimensionSize(this);
    case HloOpcode::kShiftLeft:
      return visitor->HandleShiftLeft(this);
    case HloOpcode::kShiftRightArithmetic:
      return visitor->HandleShiftRightArithmetic(this);
    case HloOpcode::kShiftRightLogical:
      return visitor->HandleShiftRightLogical(this);
    case HloOpcode::kSign:
      return visitor->HandleSign(this);
    case HloOpcode::kSlice:
      return visitor->HandleSlice(this);
    case HloOpcode::kSort:
      return visitor->HandleSort(this);
    case HloOpcode::kSubtract:
      return visitor->HandleSubtract(this);
    case HloOpcode::kTranspose:
      return visitor->HandleTranspose(this);
    case HloOpcode::kTuple:
      return visitor->HandleTuple(this);
    case HloOpcode::kWhile:
      return visitor->HandleWhile(this);
    case HloOpcode::kXor:
      return visitor->HandleXor(this);
    default:
      return absl::InternalError(absl::StrFormat(
          "Unhandled HloOpcode for DfsHloVisitor: %s. This should not happen - "
          "please file a bug for ZKX.",
          HloOpcodeString(opcode_)));
  }
}

// Explicit instantiations.
template absl::Status HloInstruction::Visit(DfsHloVisitor* visitor);
template absl::Status HloInstruction::Visit(ConstDfsHloVisitor* visitor);

// Push "child" onto the dfs_stack if not already visited.  Returns false if a
// cycle was detected, and true otherwise.
template <typename Visitor>
inline bool PushDFSChild(Visitor* visitor, DFSStack* dfs_stack,
                         HloInstruction* child) {
  CHECK(child != nullptr);
  const int id = child->unique_id();
  CHECK_GE(id, 0) << "instruction may not have a parent computation";
  switch (visitor->GetVisitState(id)) {
    case Visitor::kVisiting:
      return false;

    case Visitor::kVisited:
      // Nothing to do
      return true;

    case Visitor::kNotVisited:
      dfs_stack->push_back(std::make_pair(id, child));
      return true;
  }
}

using InternalCompareFunction =
    absl::FunctionRef<bool(std::pair<int, const HloInstruction*>,
                           std::pair<int, const HloInstruction*>)>;
template <typename Visitor>
static absl::Status PostOrderDFS(
    HloInstruction* root, Visitor* visitor,
    std::optional<InternalCompareFunction> operand_order,
    bool ignore_control_predecessors, bool cross_computation) {
  visitor->ReserveVisitStates(root->parent()->instruction_count());

  // dfs_stack holds pairs of <HloInstruction*->unique_id(), HloInstruction*>.
  //
  // We need to keep track of both the id and the instruction because
  // instructions can get deleted while they are on the stack, so we
  // can't always use the (potentially dead) instruction object to grab
  // its id.
  DFSStack dfs_stack;
  dfs_stack.emplace_back(root->unique_id(), root);

  do {
    DCHECK(!dfs_stack.empty());

    int current_id = dfs_stack.back().first;
    HloInstruction* current_node = dfs_stack.back().second;
    CHECK_GE(current_id, 0) << current_id << ": " << current_node
                            << ": instruction may not have parent computation";
    typename Visitor::VisitState visit_state =
        visitor->GetVisitState(current_id);
    if (visit_state == Visitor::kVisited) {
      dfs_stack.pop_back();
      VLOG(3) << "Not visiting HLO (id = " << current_id
              << ") as it was already visited.";
      continue;
    }

    if (visit_state == Visitor::kVisiting) {
      dfs_stack.pop_back();

      TF_RETURN_IF_ERROR(visitor->Preprocess(current_node));
      VLOG(2) << "Visiting HLO %" << current_node->name();
      TF_RETURN_IF_ERROR(current_node->Visit(visitor));
      visitor->SetVisitState(current_id, Visitor::kVisited);
      TF_RETURN_IF_ERROR(visitor->Postprocess(current_node));
      continue;
    }

    visitor->SetVisitState(current_id, Visitor::kVisiting);

    const size_t old_dfs_stack_size = dfs_stack.size();
    for (HloInstruction* child : current_node->operands()) {
      if (!ABSL_PREDICT_TRUE(PushDFSChild(visitor, &dfs_stack, child))) {
        return absl::FailedPreconditionError(absl::StrFormat(
            "A cycle is detected while visiting instruction %s %s",
            current_node->ToString(),
            PrintCycle(child, &dfs_stack, ignore_control_predecessors)));
      }
    }

    if (!ignore_control_predecessors) {
      for (HloInstruction* child : current_node->control_predecessors()) {
        if (!ABSL_PREDICT_TRUE(PushDFSChild(visitor, &dfs_stack, child))) {
          return absl::FailedPreconditionError(absl::StrFormat(
              "A cycle is detected while visiting instruction %s %s",
              current_node->ToString(),
              PrintCycle(child, &dfs_stack, ignore_control_predecessors)));
        }
      }
    }

    // If `cross_computation` is enabled, and the current visiting instruction
    // is a caller of other computations, we try to push the root instruction of
    // those called computations onto the stack .
    if (cross_computation) {
      for (const HloComputation* called_computation :
           current_node->called_computations()) {
        HloInstruction* root_instruction =
            called_computation->root_instruction();
        if (!ABSL_PREDICT_TRUE(
                PushDFSChild(visitor, &dfs_stack, root_instruction))) {
          return absl::FailedPreconditionError(absl::StrFormat(
              "A cycle is detected while visiting instruction %s %s",
              current_node->ToString(),
              PrintCycle(root_instruction, &dfs_stack,
                         ignore_control_predecessors)));
        }
      }
    }

    if (operand_order != std::nullopt) {
      std::sort(dfs_stack.begin() + old_dfs_stack_size, dfs_stack.end(),
                *operand_order);
    }

    // This makes the traversal order the same as what you'd expect
    // out of a recursive algorithm.
    std::reverse(dfs_stack.begin() + old_dfs_stack_size, dfs_stack.end());
  } while (!dfs_stack.empty());

  return absl::OkStatus();
}

template <typename HloInstructionPtr>
absl::Status HloInstruction::Accept(
    DfsHloVisitorBase<HloInstructionPtr>* visitor, bool call_finish_visit,
    bool ignore_control_predecessors, bool cross_computation) {
  VLOG(3) << "HloInstruction::Accept(%" << name() << ")";
  TF_RETURN_IF_ERROR(PostOrderDFS(this, visitor, std::nullopt,
                                  ignore_control_predecessors,
                                  cross_computation));
  if (call_finish_visit) {
    TF_RETURN_IF_ERROR(visitor->FinishVisit(this));
  }
  return absl::OkStatus();
}

// Explicit instantiations.
template absl::Status HloInstruction::Accept(DfsHloVisitor*, bool, bool, bool);
template absl::Status HloInstruction::Accept(ConstDfsHloVisitor*, bool, bool,
                                             bool);

absl::Status HloInstruction::AcceptWithOperandOrder(
    DfsHloVisitor* visitor, CompareFunction operand_order,
    bool call_finish_visit) {
  VLOG(2) << "HloInstruction::AcceptWithOperandOrder(%" << name() << ")";
  auto func = [operand_order](std::pair<int, const HloInstruction*> a,
                              std::pair<int, const HloInstruction*> b) {
    // Call the client's comparison function on the actual HloInstruction*
    // objects (ignoring the internal ids we also have in our stack entries)
    return operand_order(a.second, b.second);
  };
  TF_RETURN_IF_ERROR(PostOrderDFS(this, visitor, func,
                                  /*ignore_control_predecessors=*/false,
                                  /*cross_computation=*/false));
  if (call_finish_visit) {
    VLOG(3) << "HloInstruction::AcceptWithOperandOrder BEFORE FINISH VISIT";
    TF_RETURN_IF_ERROR(visitor->FinishVisit(this));
    VLOG(3) << "HloInstruction::AcceptWithOperandOrder AFTER FINISH VISIT";
  }
  VLOG(2) << "HloInstruction::AcceptWithOperandOrder EXIT";
  return absl::OkStatus();
}

absl::InlinedVector<int64_t, 4> HloInstruction::OperandIndices(
    const HloInstruction* operand) const {
  absl::InlinedVector<int64_t, 4> result;
  for (int64_t i = 0; i < operand_count(); ++i) {
    if (this->operand(i) == operand) {
      result.push_back(i);
    }
  }
  return result;
}

bool HloInstruction::IsElementwise() const {
  return IsElementwiseImpl(std::nullopt);
}

bool HloInstruction::IsElementwiseOnOperand(int64_t operand_idx) const {
  return IsElementwiseImpl(operand_idx);
}

namespace {

// Indicates how an instruction uses a value (such as an operand).
//
// Does it (a) use it multiple times, (b) use it, or (c) not use it?
enum class UseKind { kReuse = 0, kUse = 1, kNoUse = 2 };

// A helper class for memoized, recursive computation of HloOpcode::kFusion
// in HloInstruction::OperandElementUse below.
class FusionReusesParamElements {
 public:
  static UseKind Compute(int64_t i, const HloInstruction& hlo) {
    absl::flat_hash_map<const HloInstruction*, UseKind> memoization_cache;
    return ComputeInternal(i, hlo, &memoization_cache);
  }

 private:
  static UseKind ComputeInternal(
      int64_t outer_param_num, const HloInstruction& hlo,
      absl::flat_hash_map<const HloInstruction*, UseKind>* cache);
};

}  // namespace

// Returns how this instruction uses elements of its operand at operand_num.
static UseKind OperandElementUse(const HloInstruction& instr,
                                 int64_t operand_num) {
  switch (instr.opcode()) {
    case HloOpcode::kBitcast:
    case HloOpcode::kBitReverse:
    case HloOpcode::kConcatenate:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
    case HloOpcode::kGather:
      return UseKind::kUse;
    case HloOpcode::kPad:
      // Pad reuses the padding value but not the padded array elements.
      return operand_num > 0 ? UseKind::kReuse : UseKind::kUse;
    case HloOpcode::kReduce:
      // Reduce reuses the init values but not the operand array elements.
      return operand_num >= Cast<HloReduceInstruction>(&instr)->input_count()
                 ? UseKind::kReuse
                 : UseKind::kUse;
    case HloOpcode::kFusion:
      // Uses the memoizing, recursive computation defined above.
      return FusionReusesParamElements::Compute(operand_num,
                                                *instr.fused_expression_root());
    case HloOpcode::kDot:
      // Matrix-vector dots do not reuse the matrix operand.
      if (instr.shape().dimensions_size() <= 1) {
        if ((operand_num == 0 && instr.operand(1)->shape().rank() <= 1) ||
            (operand_num == 1 && instr.operand(0)->shape().rank() <= 1)) {
          return UseKind::kUse;
        }
      }
      return UseKind::kReuse;
    case HloOpcode::kDynamicUpdateSlice:
      // Dynamic-update-slice reuses only start_indices.
      if (operand_num == 0 || operand_num == 1) {
        return UseKind::kUse;
      }
      return UseKind::kReuse;
    default:
      return instr.IsElementwise() ? UseKind::kUse : UseKind::kReuse;
  }
}

UseKind FusionReusesParamElements::ComputeInternal(
    int64_t outer_param_num, const HloInstruction& hlo,
    absl::flat_hash_map<const HloInstruction*, UseKind>* cache) {
  if (auto hlo_param = DynCast<HloParameterInstruction>(&hlo)) {
    if (hlo_param->parameter_number() == outer_param_num) {
      return UseKind::kUse;
    }
  }

  auto p = cache->emplace(&hlo, UseKind::kNoUse);
  auto value_it = p.first;
  const bool key_is_new = p.second;

  if (!key_is_new) {
    return value_it->second;
  }

  // Our dataflow graph has no loops, so we don't need the fixed point
  // computation.
  for (int64_t operand_num = 0; operand_num < hlo.operands().size();
       ++operand_num) {
    UseKind old_val = value_it->second;

    // Compute updated value.
    UseKind new_val = [&] {
      // How does the HLO use this operand.
      UseKind hlo_use = OperandElementUse(hlo, operand_num);

      // If the HLO does not use the outer operand, return previous value.
      if (hlo_use == UseKind::kNoUse) {
        return old_val;
      }

      UseKind operand_use =
          ComputeInternal(outer_param_num, *hlo.operand(operand_num), cache);

      // If the operand does not use the outer operand, return the previous
      // value.
      if (operand_use == UseKind::kNoUse) {
        return old_val;
      }

      // Meet operator on a lattice:
      //
      //   kReuse < kUse < kNoUse.
      return std::min({old_val, hlo_use, operand_use});
    }();

    value_it = cache->find(&hlo);
    value_it->second = new_val;
    // Fold() minimizes the UseKind value. If it is already minimum, we do not
    // have to check all the remaining operands.
    if (new_val == UseKind::kReuse) {
      break;
    }
  }
  return value_it->second;
}

bool HloInstruction::ReusesOperandElements(int64_t i) const {
  return OperandElementUse(*this, i) == UseKind::kReuse;
}

std::optional<ShapeUtil::ShapeEqualityDescriptor>
HloInstruction::ReshapeMerelyInsertsOrDeletes1SizedDimensions() const {
  if (HloOpcode::kReshape != opcode_) {
    return std::nullopt;
  }
  return ShapeUtil::InsertedOrDeleted1SizedDimensions(operand(0)->shape_,
                                                      shape_);
}

std::string StatisticsVizToString(const StatisticsViz& statistics_viz) {
  // Statistics is either empty, or always starts with the index of the
  // statistic that is rendered on the graph, followed by the statistics that
  // are being tracked. The index is 0 based, starting from the first statistic
  // being tracked. The index and statistics are within a comma-separated list
  // of attribute=value pairs,
  // e.g., statistics={visualizing_index=0, count_nan=100, count_inf=200}.

  if (statistics_viz.statistics().empty()) return "{}";

  std::vector<Statistic> all_statistics(statistics_viz.statistics().begin(),
                                        statistics_viz.statistics().end());

  const auto formatter = [](std::string* out, const Statistic& item) {
    absl::StrAppend(out, item.stat_name(), "=", item.stat_val());
  };
  return absl::StrFormat("{%s,%s}",
                         absl::StrCat("visualizing_index=",
                                      statistics_viz.stat_index_to_visualize()),
                         absl::StrJoin(all_statistics, ",", formatter));
}

std::string PaddingConfigToString(const PaddingConfig& padding) {
  return absl::StrJoin(
      padding.dimensions(), "x",
      [&](std::string* out, const PaddingConfig::PaddingConfigDimension& dim) {
        absl::StrAppend(out, dim.edge_padding_low(), "_",
                        dim.edge_padding_high());
      });
}

std::string DotDimensionNumbersToString(const DotDimensionNumbers& dnums) {
  std::vector<std::string> result;
  if (!dnums.lhs_batch_dimensions().empty()) {
    result.push_back(
        absl::StrCat("lhs_batch_dims={",
                     absl::StrJoin(dnums.lhs_batch_dimensions(), ","), "}"));
  }
  result.push_back(absl::StrCat(
      "lhs_contracting_dims={",
      absl::StrJoin(dnums.lhs_contracting_dimensions(), ","), "}"));

  if (!dnums.rhs_batch_dimensions().empty()) {
    result.push_back(
        absl::StrCat("rhs_batch_dims={",
                     absl::StrJoin(dnums.rhs_batch_dimensions(), ","), "}"));
  }
  result.push_back(absl::StrCat(
      "rhs_contracting_dims={",
      absl::StrJoin(dnums.rhs_contracting_dimensions(), ","), "}"));

  return absl::StrJoin(result, ", ");
}

HloModule* HloInstruction::GetModule() const {
  if (parent_) {
    return parent_->parent();
  }
  return nullptr;
}

bool HloPtrComparator::operator()(const HloInstruction* const& lhs,
                                  const HloInstruction* const& rhs) const {
  if (rhs == nullptr) {
    // Nothing compares less than nullptr.
    return false;
  }
  if (lhs == nullptr) {
    return true;
  }
  auto lhs_module = lhs->GetModule();
  auto rhs_module = rhs->GetModule();
  CHECK((lhs_module == nullptr && rhs_module == nullptr) ||
        (lhs_module != nullptr && rhs_module != nullptr));
  if (lhs_module != nullptr &&
      lhs_module->unique_id() != rhs_module->unique_id()) {
    return lhs_module->unique_id() < rhs_module->unique_id();
  }
  return lhs->unique_id() < rhs->unique_id();
}

void HloInstruction::UniquifyName(HloModule* module) {
  UniquifyName(&module->instruction_name_uniquer());
}

void HloInstruction::SortInstructionUsersAndControlLists(
    const MappedPtrContainerSorter<HloInstruction>::MapPtrFn& map_fn,
    const HloInstruction& sorted_instruction) {
  using Sorter = MappedPtrContainerSorter<HloInstruction>;
  users_.SortInstructionUsers(map_fn, sorted_instruction.users_);

  absl::Status status;
  if (has_rare()) {
    status = Sorter::Sort(map_fn, Sorter::IndexAfterMappedElementsFn(),
                          sorted_instruction.control_predecessors(),
                          mutable_rare()->control_predecessors);
  }
  if (!status.ok()) {
    LOG(ERROR) << "Failed to sort instruction control predecessors for "
               << name() << "; " << status;
  }
  if (has_rare()) {
    status = Sorter::Sort(map_fn, Sorter::IndexAfterMappedElementsFn(),
                          sorted_instruction.control_successors(),
                          mutable_rare()->control_successors);
  }
  if (!status.ok()) {
    LOG(ERROR) << "Failed to sort instruction control successors for " << name()
               << "; " << status;
  }
}

FftType HloInstruction::fft_type() const {
  return Cast<HloFftInstruction>(this)->fft_type();
}

int64_t HloInstruction::fft_length() const {
  return Cast<HloFftInstruction>(this)->fft_length();
}

bool HloInstruction::fft_do_bit_reverse() const {
  return Cast<HloFftInstruction>(this)->fft_do_bit_reverse();
}

int32_t HloInstruction::window_bits() const {
  return Cast<HloMsmInstruction>(this)->window_bits();
}

int64_t HloInstruction::concatenate_dimension() const {
  return Cast<HloConcatenateInstruction>(this)->concatenate_dimension();
}

int64_t HloInstruction::dimension() const {
  if (auto set_size = DynCast<HloSetDimensionSizeInstruction>(this)) {
    return set_size->dimension();
  }
  return Cast<HloGetDimensionSizeInstruction>(this)->dimension();
}

int64_t HloInstruction::inferred_dimension() const {
  return Cast<HloReshapeInstruction>(this)->inferred_dimension();
}

bool HloInstruction::IsRank2Transpose() const {
  auto transpose = DynCast<HloTransposeInstruction>(this);
  return transpose != nullptr && transpose->IsRank2Transpose();
}

int64_t HloInstruction::slice_starts(int64_t dimension) const {
  return Cast<HloSliceInstruction>(this)->slice_starts(dimension);
}

const std::vector<int64_t>& HloInstruction::slice_starts() const {
  return Cast<HloSliceInstruction>(this)->slice_starts();
}

std::vector<int64_t>* HloInstruction::mutable_slice_starts() {
  return Cast<HloSliceInstruction>(this)->mutable_slice_starts();
}

int64_t HloInstruction::slice_limits(int64_t dimension) const {
  return Cast<HloSliceInstruction>(this)->slice_limits(dimension);
}

const std::vector<int64_t>& HloInstruction::slice_limits() const {
  return Cast<HloSliceInstruction>(this)->slice_limits();
}

std::vector<int64_t>* HloInstruction::mutable_slice_limits() {
  return Cast<HloSliceInstruction>(this)->mutable_slice_limits();
}

int64_t HloInstruction::slice_strides(int64_t dimension) const {
  return Cast<HloSliceInstruction>(this)->slice_strides(dimension);
}

const std::vector<int64_t>& HloInstruction::slice_strides() const {
  return Cast<HloSliceInstruction>(this)->slice_strides();
}

std::vector<int64_t>* HloInstruction::mutable_slice_strides() {
  return Cast<HloSliceInstruction>(this)->mutable_slice_strides();
}

const Literal& HloInstruction::literal() const {
  return Cast<HloConstantInstruction>(this)->literal();
}

bool HloInstruction::IsConstant() const {
  return DynCast<HloConstantInstruction>(this) != nullptr;
}

void HloInstruction::RelayoutConstant(const Layout& new_layout,
                                      const ShapeIndex& shape_index) {
  Cast<HloConstantInstruction>(this)->RelayoutConstant(new_layout, shape_index);
}

// Delegates to HloCallableInstruction::AppendInstructionIntoCalledComputation.
HloInstruction* HloInstruction::AppendInstructionIntoCalledComputation(
    HloInstruction* instruction_to_append, bool add_output) {
  return Cast<HloCallableInstruction>(this)
      ->AppendInstructionIntoCalledComputation(instruction_to_append,
                                               add_output);
}

HloInstruction* HloInstruction::AddFusionOperand(HloInstruction* new_operand) {
  return Cast<HloFusionInstruction>(this)->AddFusionOperand(new_operand);
}

// Delegates to HloFusionInstruction::MergeFusionInstruction.
void HloInstruction::MergeFusionInstruction(
    HloInstruction* instruction_to_merge) {
  return Cast<HloFusionInstruction>(this)->MergeFusionInstruction(
      Cast<HloFusionInstruction>(instruction_to_merge));
}

// Delegates to HloFusionInstruction::MergeFusionInstructionIntoMultiOutput.
void HloInstruction::MergeFusionInstructionIntoMultiOutput(
    HloInstruction* instruction_to_merge) {
  return Cast<HloFusionInstruction>(this)
      ->MergeFusionInstructionIntoMultiOutput(
          Cast<HloFusionInstruction>(instruction_to_merge));
}

HloInstruction* HloInstruction::FuseInstruction(
    HloInstruction* instruction_to_fuse) {
  return Cast<HloFusionInstruction>(this)->FuseInstruction(instruction_to_fuse);
}

HloInstruction* HloInstruction::FuseInstructionIntoMultiOutput(
    HloInstruction* instruction_to_fuse) {
  return Cast<HloFusionInstruction>(this)->FuseInstructionIntoMultiOutput(
      instruction_to_fuse);
}

HloComputation* HloInstruction::fused_instructions_computation() const {
  return Cast<HloFusionInstruction>(this)->fused_instructions_computation();
}

HloInstruction* HloInstruction::fused_expression_root() const {
  return Cast<HloFusionInstruction>(this)->fused_expression_root();
}

tsl::gtl::iterator_range<HloInstructionUnwrappingConstIterator>
HloInstruction::fused_instructions() const {
  return Cast<HloFusionInstruction>(this)->fused_instructions();
}

tsl::gtl::iterator_range<HloInstructionUnwrappingIterator>
HloInstruction::fused_instructions() {
  return Cast<HloFusionInstruction>(this)->fused_instructions();
}

int64_t HloInstruction::fused_instruction_count() const {
  return Cast<HloFusionInstruction>(this)->fused_instruction_count();
}

HloInstruction* HloInstruction::fused_parameter(
    int64_t parameter_number) const {
  return Cast<HloFusionInstruction>(this)->fused_parameter(parameter_number);
}

const HloInstruction::InstructionVector& HloInstruction::fused_parameters()
    const {
  return Cast<HloFusionInstruction>(this)->fused_parameters();
}

bool HloInstruction::IsMultiOutputFusion() const {
  const HloFusionInstruction* fusion = DynCast<HloFusionInstruction>(this);
  return fusion != nullptr && fusion->IsMultiOutputFusion();
}

HloInstruction::FusionKind HloInstruction::fusion_kind() const {
  return Cast<HloFusionInstruction>(this)->fusion_kind();
}

void HloInstruction::set_fusion_kind(FusionKind kind) {
  return Cast<HloFusionInstruction>(this)->set_fusion_kind(kind);
}

int64_t HloInstruction::parameter_number() const {
  return Cast<HloParameterInstruction>(this)->parameter_number();
}

void HloInstruction::set_parameter_replicated_at_leaf_buffers(
    absl::Span<const bool> parameter_replicated_at_leaf_buffers) {
  return Cast<HloParameterInstruction>(this)
      ->set_parameter_replicated_at_leaf_buffers(
          parameter_replicated_at_leaf_buffers);
}

void HloInstruction::set_parameter_replicated_at_leaf_buffers(
    const std::vector<bool>& parameter_replicated_at_leaf_buffers) {
  return Cast<HloParameterInstruction>(this)
      ->set_parameter_replicated_at_leaf_buffers(
          parameter_replicated_at_leaf_buffers);
}

const std::optional<std::vector<bool>>&
HloInstruction::parameter_replicated_at_leaf_buffers() const {
  return Cast<HloParameterInstruction>(this)
      ->parameter_replicated_at_leaf_buffers();
}

int64_t HloInstruction::tuple_index() const {
  return Cast<HloGetTupleElementInstruction>(this)->tuple_index();
}

void HloInstruction::set_tuple_index(int64_t new_tuple_index) {
  return Cast<HloGetTupleElementInstruction>(this)->set_tuple_index(
      new_tuple_index);
}

std::string HloInstruction::infeed_config() const {
  return Cast<HloInfeedInstruction>(this)->infeed_config();
}

void HloInstruction::set_infeed_config(const std::string& config) {
  return Cast<HloInfeedInstruction>(this)->set_infeed_config(config);
}

const Shape& HloInstruction::outfeed_shape() const {
  return Cast<HloOutfeedInstruction>(this)->outfeed_shape();
}

Shape* HloInstruction::mutable_outfeed_shape() {
  return Cast<HloOutfeedInstruction>(this)->mutable_outfeed_shape();
}

const std::string& HloInstruction::outfeed_config() const {
  return Cast<HloOutfeedInstruction>(this)->outfeed_config();
}

void HloInstruction::set_outfeed_config(const std::string& config) {
  return Cast<HloOutfeedInstruction>(this)->set_outfeed_config(config);
}

const std::vector<ReplicaGroup>& HloInstruction::replica_groups() const {
  return Cast<HloCollectiveInstruction>(this)->replica_groups();
}

const CollectiveDeviceList& HloInstruction::device_list() const {
  return Cast<HloCollectiveInstruction>(this)->device_list();
}

const std::vector<std::pair<int64_t, int64_t>>&
HloInstruction::source_target_pairs() const {
  return Cast<HloCollectivePermuteInstruction>(this)->source_target_pairs();
}

std::optional<int64_t> HloInstruction::channel_id() const {
  return Cast<HloChannelInstruction>(this)->channel_id();
}

void HloInstruction::set_channel_id(const std::optional<int64_t>& channel_id) {
  return Cast<HloChannelInstruction>(this)->set_channel_id(channel_id);
}

int64_t HloInstruction::iota_dimension() const {
  return Cast<HloIotaInstruction>(this)->iota_dimension();
}

const PaddingConfig& HloInstruction::padding_config() const {
  return Cast<HloPadInstruction>(this)->padding_config();
}

PaddingConfig* HloInstruction::mutable_padding_config() {
  return Cast<HloPadInstruction>(this)->mutable_padding_config();
}

int64_t HloInstruction::slice_sizes(int64_t dimension) const {
  return Cast<HloDynamicSliceInstruction>(this)->slice_sizes(dimension);
}

const std::vector<int64_t>& HloInstruction::dynamic_slice_sizes() const {
  return Cast<HloDynamicSliceInstruction>(this)->dynamic_slice_sizes();
}

const GatherDimensionNumbers& HloInstruction::gather_dimension_numbers() const {
  return Cast<HloGatherInstruction>(this)->gather_dimension_numbers();
}

absl::Span<const int64_t> HloInstruction::gather_slice_sizes() const {
  return Cast<HloGatherInstruction>(this)->gather_slice_sizes();
}

const ScatterDimensionNumbers& HloInstruction::scatter_dimension_numbers()
    const {
  return Cast<HloScatterInstruction>(this)->scatter_dimension_numbers();
}

const DotDimensionNumbers& HloInstruction::dot_dimension_numbers() const {
  return Cast<HloDotInstruction>(this)->dot_dimension_numbers();
}

absl::Span<const SparsityDescriptor> HloInstruction::sparsity() const {
  return Cast<HloDotInstruction>(this)->sparsity();
}

bool HloInstruction::IsAsynchronous() const {
  return HloOpcodeIsAsync(opcode());
}

HloInstruction* HloInstruction::async_chain_start() const {
  return Cast<HloAsyncInstruction>(this)->async_chain_start();
}

HloInstruction* HloInstruction::async_chain_done() const {
  return Cast<HloAsyncInstruction>(this)->async_chain_done();
}

HloComputation* HloInstruction::async_wrapped_computation() const {
  return Cast<HloAsyncInstruction>(this)->async_wrapped_computation();
}

HloInstruction* HloInstruction::async_wrapped_instruction() const {
  return Cast<HloAsyncInstruction>(this)->async_wrapped_instruction();
}

HloOpcode HloInstruction::async_wrapped_opcode() const {
  return Cast<HloAsyncInstruction>(this)->async_wrapped_opcode();
}

std::string_view HloInstruction::async_execution_thread() const {
  return Cast<HloAsyncInstruction>(this)->async_execution_thread();
}

void HloInstruction::set_async_execution_thread(
    std::string_view async_execution_thread) {
  Cast<HloAsyncInstruction>(this)->set_async_execution_thread(
      async_execution_thread);
}

void HloInstruction::set_called_computations_execution_thread(
    std::string_view async_execution_thread,
    bool skip_async_execution_thread_overwrite) {
  Cast<HloCallableInstruction>(this)->RecursivelySetComputationsThreadName(
      async_execution_thread, skip_async_execution_thread_overwrite);
}

std::optional<int> HloInstruction::cross_program_prefetch_index() const {
  return Cast<HloCopyStartInstruction>(this)->cross_program_prefetch_index();
}

ComparisonDirection HloInstruction::comparison_direction() const {
  return Cast<HloCompareInstruction>(this)->direction();
}

ComparisonOrder HloInstruction::comparison_order() const {
  return Cast<HloCompareInstruction>(this)->order();
}

const std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>&
HloInstruction::output_operand_aliasing() const {
  return Cast<HloCallableInstruction>(this)->output_to_operand_aliasing();
}

void HloInstruction::set_output_to_operand_aliasing(
    std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
        aliasing) {
  Cast<HloCallableInstruction>(this)->set_output_to_operand_aliasing(
      std::move(aliasing));
}

const std::string& HloInstruction::custom_call_target() const {
  return Cast<HloCustomCallInstruction>(this)->custom_call_target();
}

void HloInstruction::set_custom_call_target(std::string_view target) {
  Cast<HloCustomCallInstruction>(this)->set_custom_call_target(target);
}

std::string_view ToString(HloInstruction::FusionKind kind) {
  switch (kind) {
    case HloInstruction::FusionKind::kLoop:
      return "kLoop";
    case HloInstruction::FusionKind::kInput:
      return "kInput";
    case HloInstruction::FusionKind::kOutput:
      return "kOutput";
    case HloInstruction::FusionKind::kCustom:
      return "kCustom";
  }
}

absl::StatusOr<CustomCallSchedule> StringToCustomCallSchedule(
    std::string_view name) {
  static const absl::flat_hash_map<std::string, CustomCallSchedule>* map = [] {
    static auto* map = absl::IgnoreLeak(
        new absl::flat_hash_map<std::string, CustomCallSchedule>);
    for (int i = 0; i < CustomCallSchedule_ARRAYSIZE; i++) {
      if (CustomCallSchedule_IsValid(i)) {
        auto value = static_cast<CustomCallSchedule>(i);
        (*map)[CustomCallScheduleToString(value)] = value;
      }
    }
    return map;
  }();
  auto found = map->find(absl::AsciiStrToLower(name));
  if (found == map->end()) {
    return absl::InvalidArgumentError("Unknown schedule");
  }
  return found->second;
}

absl::StatusOr<CustomCallApiVersion> StringToCustomCallApiVersion(
    std::string_view name) {
  static const absl::flat_hash_map<std::string, CustomCallApiVersion>* map =
      [] {
        static auto* map = absl::IgnoreLeak(
            new absl::flat_hash_map<std::string, CustomCallApiVersion>);
        for (int i = 0; i < CustomCallApiVersion_ARRAYSIZE; i++) {
          if (CustomCallApiVersion_IsValid(i)) {
            auto value = static_cast<CustomCallApiVersion>(i);
            (*map)[CustomCallApiVersionToString(value)] = value;
          }
        }
        return map;
      }();
  auto found = map->find(absl::AsciiStrToLower(name));
  if (found == map->end()) {
    return absl::InvalidArgumentError("Unknown API version");
  }
  return found->second;
}
absl::StatusOr<HloInstruction::FusionKind> StringToFusionKind(
    std::string_view kind_name) {
  if (kind_name == "kLoop") {
    return HloInstruction::FusionKind::kLoop;
  }
  if (kind_name == "kInput") {
    return HloInstruction::FusionKind::kInput;
  }
  if (kind_name == "kOutput") {
    return HloInstruction::FusionKind::kOutput;
  }
  if (kind_name == "kCustom") {
    return HloInstruction::FusionKind::kCustom;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Unknown fusion kind: %s", kind_name));
}

}  // namespace zkx
