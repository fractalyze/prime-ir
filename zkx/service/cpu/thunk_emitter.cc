/* Copyright 2024 The OpenXLA Authors.
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

#include "zkx/service/cpu/thunk_emitter.h"

#include "absl/strings/str_format.h"

#include "xla/tsl/platform/casts.h"
#include "xla/tsl/profiler/lib/traceme.h"
#include "zkx/backends/cpu/codegen/cpu_kernel_emitter.h"
#include "zkx/backends/cpu/runtime/all_gather_thunk.h"
#include "zkx/backends/cpu/runtime/all_reduce_thunk.h"
#include "zkx/backends/cpu/runtime/all_to_all_thunk.h"
#include "zkx/backends/cpu/runtime/call_thunk.h"
#include "zkx/backends/cpu/runtime/collective_permute_thunk.h"
#include "zkx/backends/cpu/runtime/conditional_thunk.h"
#include "zkx/backends/cpu/runtime/copy_thunk.h"
#include "zkx/backends/cpu/runtime/infeed_thunk.h"
#include "zkx/backends/cpu/runtime/kernel_thunk.h"
#include "zkx/backends/cpu/runtime/outfeed_thunk.h"
#include "zkx/backends/cpu/runtime/reduce_scatter_thunk.h"
#include "zkx/backends/cpu/runtime/while_thunk.h"
#include "zkx/codegen/llvm_ir_kernel_source.h"
#include "zkx/cpu_function_runtime.h"
#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/service/collective_ops_utils.h"
#include "zkx/service/pattern_matcher.h"

namespace zkx::cpu {
namespace {

Thunk::Info ThunkInfo(const HloInstruction* instruction) {
  const HloModule* module = instruction->GetModule();
  return Thunk::Info{std::string(instruction->name()),
                     std::string(module->name()), module->unique_id()};
}

absl::StatusOr<ReductionKind> MatchReductionKind(
    const HloComputation* computation) {
  if (auto reduction_kind = MatchReductionComputation(computation)) {
    return reduction_kind.value();
  }
  return absl::UnimplementedError(absl::StrFormat(
      "Unsupported reduction computation: %s", computation->ToString()));
}

template <typename CollectiveInstruction>
absl::StatusOr<CollectiveThunk::OpParams> GetCollectiveOpParams(
    const CollectiveInstruction* instruction) {
  return CollectiveThunk::OpParams{
      /*op_id=*/instruction->channel_id().has_value()
          ? instruction->channel_id().value()
          : instruction->GetModule()->unique_id(),
      /*has_channel_id=*/instruction->channel_id().has_value(),
      /*use_global_device_ids=*/instruction->use_global_device_ids(),
      /*replica_groups=*/instruction->replica_groups(),
  };
}

// TODO(ezhulenev): Figure out why AllToAll instruction does not have
// `use_global_device_ids` field and how to unify it with every other collective
// operation.
absl::StatusOr<CollectiveThunk::OpParams> GetCollectiveOpParams(
    const HloAllToAllInstruction* instruction) {
  return CollectiveThunk::OpParams{
      /*op_id=*/instruction->channel_id().has_value()
          ? instruction->channel_id().value()
          : instruction->GetModule()->unique_id(),
      /*has_channel_id=*/instruction->channel_id().has_value(),
      /*use_global_device_ids=*/std::nullopt,
      /*replica_groups=*/instruction->replica_groups(),
  };
}

// TODO(ezhulenev): Figure out why CollectivePermute instruction does not have
// `use_global_device_ids` field and how to unify it with every other collective
// operation.
absl::StatusOr<CollectiveThunk::OpParams> GetCollectiveOpParams(
    const HloCollectivePermuteInstruction* instruction) {
  return CollectiveThunk::OpParams{
      /*op_id=*/instruction->channel_id().has_value()
          ? instruction->channel_id().value()
          : instruction->GetModule()->unique_id(),
      /*has_channel_id=*/instruction->channel_id().has_value(),
      /*use_global_device_ids=*/std::nullopt,
      /*replica_groups=*/{},  // CollectivePermute does not have replica groups
  };
}

absl::StatusOr<CollectiveThunk::OpBuffers> GetCollectiveOpBuffers(
    const HloInstruction* instruction,
    const BufferAssignment* buffer_assignment) {
  // Collect buffer slices for all operands.
  std::vector<BufferAllocation::Slice> source_buffers;
  std::vector<Shape> source_shapes;

  for (const HloInstruction* operand : instruction->operands()) {
    TF_ASSIGN_OR_RETURN(source_buffers.emplace_back(),
                        buffer_assignment->GetUniqueSlice(operand, {}));
    source_shapes.push_back(operand->shape());
  }

  // Collect buffer slices for all results.
  std::vector<BufferAllocation::Slice> destination_buffers;
  std::vector<Shape> destination_shapes;

  for (auto& indexed : ShapeUtil::GetLeafShapes(instruction->shape())) {
    TF_ASSIGN_OR_RETURN(
        destination_buffers.emplace_back(),
        buffer_assignment->GetUniqueSlice(instruction, indexed.index));
    destination_shapes.push_back(indexed.shape);
  }

  return CollectiveThunk::OpBuffers{
      /*source_buffers=*/std::move(source_buffers),
      /*source_shapes=*/std::move(source_shapes),
      /*destination_buffers=*/std::move(destination_buffers),
      /*destination_shapes=*/std::move(destination_shapes),
  };
}

}  // namespace

ThunkEmitter::ThunkEmitter(const BufferAssignment* buffer_assignment,
                           mlir::MLIRContext* mlir_context)
    : buffer_assignment_(buffer_assignment),
      mlir_context_(mlir_context),
      communicator_resource_(
          Resource::Create(Resource::kCollectiveCommunicator)) {}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitHloInstruction(
    const HloInstruction* instr) {
  switch (instr->opcode()) {
    // Instructions that do not have a thunk implementation and are instead
    // fully defined by the corresponding buffer assignment.
    case HloOpcode::kBitcast:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kParameter:
    case HloOpcode::kTuple:
      return ThunkSequence::Empty();

    // No-op operations that are used to provide more metadata about the HLO
    // dataflow graph.
    case HloOpcode::kAfterAll:             // Defines an execution order.
    case HloOpcode::kAddDependency:        // Defines an execution order.
    case HloOpcode::kDomain:               // Defines an HLO domain.
    case HloOpcode::kOptimizationBarrier:  // Prevents moving ops past barrier.
      return ThunkSequence::Empty();

    // Allocations for constants owned by the executable, and resolved at run
    // time according to the buffer assignment (using allocation index). We do
    // not need to emit any thunks for constant instructions.
    case HloOpcode::kConstant:
      return ThunkSequence::Empty();

    // Call operations are simply converted to a ThunkSequence emitted from the
    // called computation and embedded into the "main" one.
    case HloOpcode::kCall:
      return EmitCallThunk(instr);

    // Control flow thunks check predicates on the host and launch nested thunk
    // sequences for branches and loops.
    case HloOpcode::kConditional:
      return EmitConditionThunk(instr);
    case HloOpcode::kWhile:
      return EmitWhileThunk(instr);

    case HloOpcode::kAbs:
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kBitReverse:
    case HloOpcode::kBroadcast:
    case HloOpcode::kClamp:
    case HloOpcode::kClz:
    case HloOpcode::kCompare:
    case HloOpcode::kConcatenate:
    case HloOpcode::kConvert:
    case HloOpcode::kDivide:
    case HloOpcode::kDot:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kFft:
    case HloOpcode::kFusion:
    case HloOpcode::kInverse:
    case HloOpcode::kIota:
    case HloOpcode::kMap:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMsm:
    case HloOpcode::kMultiply:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kOr:
    case HloOpcode::kPad:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kPower:
    case HloOpcode::kReduce:
    case HloOpcode::kRemainder:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kSelect:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kSign:
    case HloOpcode::kSlice:
    case HloOpcode::kSubtract:
    case HloOpcode::kTranspose:
    case HloOpcode::kXor:
      return EmitKernelThunk(instr);
    case HloOpcode::kAllGather:
      return EmitAllGatherThunk(instr);
    case HloOpcode::kAllReduce:
      return EmitAllReduceThunk(instr);
    case HloOpcode::kAllToAll:
      return EmitAllToAllThunk(instr);
    case HloOpcode::kCollectivePermute:
      return EmitCollectivePermuteThunk(instr);
    case HloOpcode::kReduceScatter:
      return EmitReduceScatterThunk(instr);
    case HloOpcode::kInfeed:
      return EmitInfeedThunk(instr);
    case HloOpcode::kOutfeed:
      return EmitOutfeedThunk(instr);
    case HloOpcode::kCopy:
      return EmitCopyThunk(instr);
    case HloOpcode::kSort:
      return EmitSortThunk(instr);

    default:
      return absl::InternalError(
          absl::StrFormat("Unsupported instruction opcode: %s",
                          HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<BufferAllocation::Slice> ThunkEmitter::GetAllocationSlice(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  return buffer_assignment_->GetUniqueSlice(instruction, index);
}

absl::StatusOr<std::shared_ptr<Resource>> ThunkEmitter::GetTokenResource(
    const HloInstruction* instruction, const ShapeIndex& index) {
  DCHECK(ShapeUtil::GetSubshape(instruction->shape(), index).IsToken());
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                      GetAllocationSlice(instruction, index));
  if (auto it = token_resources_.find(slice); it != token_resources_.end()) {
    return it->second;
  }
  return token_resources_[slice] = Resource::Create(Resource::kToken);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitHloComputation(
    const HloComputation* computation) {
  ThunkSequence thunks;

  const HloSchedule& schedule = computation->parent()->schedule();
  if (!schedule.is_computation_scheduled(computation))
    return absl::InternalError(absl::StrFormat(
        "Sequence not found for computation: %s", computation->name()));

  const HloInstructionSequence& sequence = schedule.sequence(computation);
  for (HloInstruction* instr : sequence.instructions()) {
    TF_ASSIGN_OR_RETURN(ThunkSequence instr_thunks, EmitHloInstruction(instr));
    thunks.Append(std::move(instr_thunks));
  }

  return thunks;
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitEntryComputation(
    const HloModule& module) {
  if (!module.has_schedule()) {
    return absl::InternalError("HLO module must be scheduled to emit thunks");
  }
  tsl::profiler::TraceMe trace("ThunkEmitter::EmitEntryComputation");
  return EmitHloComputation(module.entry_computation());
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitKernelThunk(
    const HloInstruction* instruction) {
  CpuKernelEmitter emitter(mlir_context_, instruction, buffer_assignment_);
  TF_ASSIGN_OR_RETURN(KernelDefinition kernel_definition,
                      emitter.EmitKernelDefinition());

  auto [kernel_spec, kernel_source] = std::move(kernel_definition).release();
  auto llvm_ir_kernel_source = absl::WrapUnique<LlvmIrKernelSource>(
      tsl::down_cast<LlvmIrKernelSource*>(kernel_source.release()));

  kernels_.push_back({kernel_spec.name(),
                      std::move(*llvm_ir_kernel_source).thread_safe_module()});

  return MakeKernelThunkSequence(
      instruction, std::move(kernel_spec),
      /*min_alignment=*/cpu_function_runtime::MinAlign());
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitAllGatherThunk(
    const HloInstruction* instruction) {
  auto* all_gather = Cast<HloAllGatherInstruction>(instruction);

  TF_ASSIGN_OR_RETURN(AllGatherThunk::OpParams op_params,
                      GetCollectiveOpParams(all_gather));
  TF_ASSIGN_OR_RETURN(AllGatherThunk::OpBuffers op_buffers,
                      GetCollectiveOpBuffers(all_gather, buffer_assignment_));
  AllGatherThunk::OpResources op_resources = {communicator_resource_};

  return ThunkSequence::Of<AllGatherThunk>(
      ThunkInfo(all_gather), std::move(op_params), std::move(op_buffers),
      std::move(op_resources));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitAllReduceThunk(
    const HloInstruction* instruction) {
  auto* all_reduce = Cast<HloAllReduceInstruction>(instruction);

  TF_ASSIGN_OR_RETURN(ReductionKind reduction_kind,
                      MatchReductionKind(all_reduce->to_apply()));
  TF_ASSIGN_OR_RETURN(AllReduceThunk::OpParams op_params,
                      GetCollectiveOpParams(all_reduce));
  TF_ASSIGN_OR_RETURN(AllReduceThunk::OpBuffers op_buffers,
                      GetCollectiveOpBuffers(all_reduce, buffer_assignment_));
  AllReduceThunk::OpResources op_resources = {communicator_resource_};

  const HloModuleConfig& config = instruction->GetModule()->config();
  bool single_replica =
      config.replica_count() == 1 && config.num_partitions() == 1;

  return ThunkSequence::Of<AllReduceThunk>(
      ThunkInfo(all_reduce), reduction_kind, std::move(op_params),
      std::move(op_buffers), std::move(op_resources), single_replica);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitAllToAllThunk(
    const HloInstruction* instruction) {
  auto* all_to_all = Cast<HloAllToAllInstruction>(instruction);

  TF_ASSIGN_OR_RETURN(AllToAllThunk::OpParams op_params,
                      GetCollectiveOpParams(all_to_all));
  TF_ASSIGN_OR_RETURN(AllToAllThunk::OpBuffers op_buffers,
                      GetCollectiveOpBuffers(all_to_all, buffer_assignment_));
  AllToAllThunk::OpResources op_resources = {communicator_resource_};

  return ThunkSequence::Of<AllToAllThunk>(
      ThunkInfo(all_to_all), std::move(op_params), std::move(op_buffers),
      std::move(op_resources));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCollectivePermuteThunk(
    const HloInstruction* instruction) {
  auto* collective_permute = Cast<HloCollectivePermuteInstruction>(instruction);

  TF_ASSIGN_OR_RETURN(CollectivePermuteThunk::OpParams op_params,
                      GetCollectiveOpParams(collective_permute));
  TF_ASSIGN_OR_RETURN(
      CollectivePermuteThunk::OpBuffers op_buffers,
      GetCollectiveOpBuffers(collective_permute, buffer_assignment_));
  CollectivePermuteThunk::OpResources op_resources = {communicator_resource_};

  return ThunkSequence::Of<CollectivePermuteThunk>(
      ThunkInfo(collective_permute), std::move(op_params),
      std::move(op_buffers), std::move(op_resources),
      collective_permute->source_target_pairs());
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitReduceScatterThunk(
    const HloInstruction* instruction) {
  auto* reduce_scatter = Cast<HloReduceScatterInstruction>(instruction);

  TF_ASSIGN_OR_RETURN(ReductionKind reduction_kind,
                      MatchReductionKind(reduce_scatter->to_apply()));
  TF_ASSIGN_OR_RETURN(ReduceScatterThunk::OpParams op_params,
                      GetCollectiveOpParams(reduce_scatter));
  TF_ASSIGN_OR_RETURN(
      ReduceScatterThunk::OpBuffers op_buffers,
      GetCollectiveOpBuffers(reduce_scatter, buffer_assignment_));
  ReduceScatterThunk::OpResources op_resources = {communicator_resource_};

  return ThunkSequence::Of<ReduceScatterThunk>(
      ThunkInfo(reduce_scatter), reduction_kind, std::move(op_params),
      std::move(op_buffers), std::move(op_resources));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitInfeedThunk(
    const HloInstruction* instruction) {
  auto* infeed = Cast<HloInfeedInstruction>(instruction);
  const Shape& infeed_shape = infeed->infeed_shape();

  // Collect buffer allocation slices corresponding to data buffers produced by
  // the infeed instruction;
  std::vector<InfeedThunk::InfeedBuffer> infeed_buffers;
  for (auto& infeed_leaf : ShapeUtil::GetLeafShapes(infeed_shape)) {
    infeed_leaf.index.push_front(0);  // prepend infeed tuple index

    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice infeed_slice,
                        GetAllocationSlice(infeed, infeed_leaf.index));

    infeed_buffers.push_back(InfeedThunk::InfeedBuffer{
        infeed_slice,
        infeed_leaf.shape,
    });
  }

  // Collect resources for consumed and produced tokens.
  InfeedThunk::InfeedResources infeed_resources;
  TF_ASSIGN_OR_RETURN(infeed_resources.consume_token,
                      GetTokenResource(infeed->operand(0)));
  TF_ASSIGN_OR_RETURN(infeed_resources.produce_token,
                      GetTokenResource(infeed, {1}));

  return ThunkSequence::Of<InfeedThunk>(ThunkInfo(instruction), infeed_buffers,
                                        std::move(infeed_resources));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitOutfeedThunk(
    const HloInstruction* instruction) {
  auto* outfeed = Cast<HloOutfeedInstruction>(instruction);
  const Shape& outfeed_shape = outfeed->outfeed_shape();

  // Collect buffer allocation slices corresponding to data buffers fed into the
  // outfeed instruction as first operand.
  std::vector<OutfeedThunk::OutfeedBuffer> outfeed_buffers;
  for (auto& outfeed_leaf : ShapeUtil::GetLeafShapes(outfeed_shape)) {
    TF_ASSIGN_OR_RETURN(
        BufferAllocation::Slice outfeed_slice,
        GetAllocationSlice(outfeed->operand(0), outfeed_leaf.index));

    outfeed_buffers.push_back(OutfeedThunk::OutfeedBuffer{
        outfeed_slice,
        outfeed_leaf.shape,
    });
  }

  // Collect resources for consumed and produced tokens.
  OutfeedThunk::OutfeedResources outfeed_resources;
  TF_ASSIGN_OR_RETURN(outfeed_resources.consume_token,
                      GetTokenResource(outfeed->operand(1)));
  TF_ASSIGN_OR_RETURN(outfeed_resources.produce_token,
                      GetTokenResource(outfeed));

  return ThunkSequence::Of<OutfeedThunk>(
      ThunkInfo(instruction), outfeed_buffers, std::move(outfeed_resources));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCallThunk(
    const HloInstruction* instruction) {
  TF_ASSIGN_OR_RETURN(
      ThunkSequence called_sequence,
      EmitHloComputation(instruction->called_computations().front()));
  return ThunkSequence::Of<CallThunk>(ThunkInfo(instruction),
                                      std::move(called_sequence));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitConditionThunk(
    const HloInstruction* instruction) {
  std::vector<ThunkSequence> branches;
  TF_ASSIGN_OR_RETURN(auto branch_index_buffer,
                      GetAllocationSlice(instruction->operand(0)));

  for (HloComputation* branch : instruction->branch_computations()) {
    TF_ASSIGN_OR_RETURN(branches.emplace_back(), EmitHloComputation(branch));
  }

  return ThunkSequence::Of<ConditionalThunk>(
      ThunkInfo(instruction), branch_index_buffer, std::move(branches));
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitWhileThunk(
    const HloInstruction* instruction) {
  HloInstruction* cond = instruction->while_condition()->root_instruction();
  TF_ASSIGN_OR_RETURN(auto cond_buffer, GetAllocationSlice(cond));

  TF_ASSIGN_OR_RETURN(ThunkSequence cond_thunk,
                      EmitHloComputation(instruction->while_condition()));
  TF_ASSIGN_OR_RETURN(ThunkSequence body_thunk,
                      EmitHloComputation(instruction->while_body()));

  // Check if while loop has a statically known trip count.
  TF_ASSIGN_OR_RETURN(auto loop_config,
                      instruction->backend_config<WhileLoopBackendConfig>());

  std::optional<int64_t> trip_count;
  if (loop_config.has_known_trip_count()) {
    trip_count = loop_config.known_trip_count().n();
  }

  return ThunkSequence::Of<WhileThunk>(ThunkInfo(instruction), cond_buffer,
                                       std::move(cond_thunk),
                                       std::move(body_thunk), trip_count);
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitCopyThunk(
    const HloInstruction* instruction) {
  const HloInstruction* source = instruction->operand(0);
  TF_ASSIGN_OR_RETURN(auto source_buffer, GetAllocationSlice(source));
  TF_ASSIGN_OR_RETURN(auto destination_buffer, GetAllocationSlice(instruction));
  return ThunkSequence::Of<CopyThunk>(ThunkInfo(instruction), source_buffer,
                                      source->shape(), destination_buffer,
                                      instruction->shape());
}

// Parse the sort comparator to determine the sort direction. Comparator is
// expected to be an HloOpcode::kCompare with two parameters.
std::optional<SortThunk::SortDirection> ThunkEmitter::MatchSortDirection(
    const HloComputation* hlo_comparator) const {
  namespace m = match;
  std::optional<SortThunk::SortDirection> direction = std::nullopt;

  // TODO(tsilytskyi): Handle more than two input parameters.
  if (hlo_comparator->root_instruction()->opcode() == HloOpcode::kCompare &&
      hlo_comparator->root_instruction()->operand(0)->opcode() ==
          HloOpcode::kParameter &&
      hlo_comparator->root_instruction()->operand(1)->opcode() ==
          HloOpcode::kParameter &&
      hlo_comparator->num_parameters() == 2) {
    auto* compare =
        Cast<HloCompareInstruction>(hlo_comparator->root_instruction());

    // Take into account the order of the parameters. If they are swapped,
    // the sort direction will be reversed.
    const bool expected_param_order =
        (Match(compare, m::Op()
                            .WithOperand(0, m::Parameter(0))
                            .WithOperand(1, m::Parameter(1))));
    switch (compare->comparison_direction()) {
      case ComparisonDirection::kGe:
        direction = (expected_param_order)
                        ? SortThunk::SortDirection::kDescending
                        : SortThunk::SortDirection::kAscending;
        break;
      case ComparisonDirection::kLt:
        direction = (expected_param_order)
                        ? SortThunk::SortDirection::kAscending
                        : SortThunk::SortDirection::kDescending;
        break;
      default:
        break;
    }
  }

  return direction;
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitSortThunk(
    const HloInstruction* instruction) {
  auto* sort = Cast<HloSortInstruction>(instruction);

  HloComputation* hlo_comparator = sort->to_apply();

  const std::optional<SortThunk::SortDirection> direction =
      MatchSortDirection(hlo_comparator);

  CpuKernelEmitter emitter(mlir_context_, instruction, buffer_assignment_);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<KernelSource> comparator_source,
                      emitter.EmitComparator(hlo_comparator));
  auto llvm_ir_comparator_source = absl::WrapUnique<LlvmIrKernelSource>(
      tsl::down_cast<LlvmIrKernelSource*>(comparator_source.release()));

  comparators_.push_back(
      {std::string(hlo_comparator->name()),
       std::move(*llvm_ir_comparator_source).thread_safe_module()});

  TF_ASSIGN_OR_RETURN(ThunkSequence comparator_thunks,
                      EmitHloComputation(hlo_comparator));

  TF_ASSIGN_OR_RETURN(auto buffers, GetHostKernelAllocationSlices(sort));

  if (buffers.arguments.size() != buffers.results.size()) {
    return absl::InternalError(
        "Sort operation expects the same number of operands and results");
  }

  ThunkSequence thunks;

  std::vector<SortThunk::Input> inputs;
  inputs.reserve(sort->operand_count());

  for (size_t i = 0; i < sort->operand_count(); ++i) {
    const Shape& shape = sort->operand(i)->shape();

    BufferAllocation::Slice arg = buffers.arguments[i];
    BufferAllocation::Slice result = buffers.results[i];

    // Copy argument to result if they are not the same buffer.
    if (arg != result) {
      TF_ASSIGN_OR_RETURN(
          thunks.emplace_back(),
          CopyThunk::Create(ThunkInfo(instruction), arg, shape, result, shape));
    }

    // Add sort thunk input to sort result buffer inplace.
    inputs.push_back(SortThunk::Input{result, shape});
  }

  TF_ASSIGN_OR_RETURN(
      thunks.emplace_back(),
      SortThunk::Create(ThunkInfo(instruction), inputs, sort->sort_dimension(),
                        sort->is_stable(), comparators_.back().comparator_name,
                        direction));

  return thunks;
}

absl::StatusOr<ThunkEmitter::HostKernelAllocationSlices>
ThunkEmitter::GetHostKernelAllocationSlices(
    const HloInstruction* instruction) const {
  HostKernelAllocationSlices slices;

  auto add_buffers = [&](std::vector<BufferAllocation::Slice>& buffers,
                         const HloInstruction* instr) -> absl::Status {
    for (const auto& indexed : ShapeUtil::GetLeafShapes(instr->shape())) {
      TF_ASSIGN_OR_RETURN(buffers.emplace_back(),
                          GetAllocationSlice(instr, indexed.index));
    }
    return absl::OkStatus();
  };

  for (HloInstruction* operand : instruction->operands()) {
    TF_RETURN_IF_ERROR(add_buffers(slices.arguments, operand));
  }

  TF_RETURN_IF_ERROR(add_buffers(slices.results, instruction));

  return slices;
}

// static
absl::StatusOr<ThunkSequence> ThunkEmitter::MakeKernelThunkSequence(
    const HloInstruction* instruction, const KernelSpec& kernel_spec,
    std::optional<uint64_t> min_alignment) {
  return ThunkSequence::Of<KernelThunk>(ThunkInfo(instruction), kernel_spec,
                                        min_alignment);
}

}  // namespace zkx::cpu
