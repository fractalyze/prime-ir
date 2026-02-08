/* Copyright 2019 The OpenXLA Authors.
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

#include "zkx/service/collective_ops_utils.h"

#include "absl/strings/str_join.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/service/pattern_matcher.h"
#include "zkx/status_macros.h"
#include "zkx/util.h"

namespace zkx {

std::string_view ReductionKindToString(ReductionKind reduction_kind) {
  switch (reduction_kind) {
    case ReductionKind::kSum:
      return "sum";
    case ReductionKind::kProduct:
      return "prod";
    case ReductionKind::kMin:
      return "min";
    case ReductionKind::kMax:
      return "max";
  }
}

absl::StatusOr<ReductionKind> StringToReductionKind(
    std::string_view reduction_kind) {
  if (reduction_kind == "sum") {
    return ReductionKind::kSum;
  } else if (reduction_kind == "prod") {
    return ReductionKind::kProduct;
  } else if (reduction_kind == "min") {
    return ReductionKind::kMin;
  } else if (reduction_kind == "max") {
    return ReductionKind::kMax;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Invalid reduction kind: %s", reduction_kind));
}

// Match the instruction to a reduction kind. We can represent and/or of pred as
// min/max. This works because pred is stored as an 8-bit int of value 0 or 1.
std::optional<ReductionKind> MatchReductionInstruction(
    const HloInstruction* hlo) {
  PrimitiveType type = hlo->shape().element_type();
  switch (hlo->opcode()) {
    case HloOpcode::kAdd:
      return ReductionKind::kSum;
    case HloOpcode::kAnd:
      return type == PRED ? std::optional<ReductionKind>(ReductionKind::kMin)
                          : std::nullopt;
    case HloOpcode::kMultiply:
      return ReductionKind::kProduct;
    case HloOpcode::kMinimum:
      return ReductionKind::kMin;
    case HloOpcode::kMaximum:
      return ReductionKind::kMax;
    case HloOpcode::kOr:
      return type == PRED ? std::optional<ReductionKind>(ReductionKind::kMax)
                          : std::nullopt;
    default:
      return std::nullopt;
  }
}

std::optional<ReductionKind> MatchReductionComputation(
    const HloComputation* computation) {
  namespace m = match;
  const HloInstruction* root = computation->root_instruction();
  std::optional<ReductionKind> kind = MatchReductionInstruction(root);
  if (kind && !Match(root, m::Op()
                               .WithBinaryOperandsAnyOrder(m::Parameter(0),
                                                           m::Parameter(1))
                               .WithShape(m::Shape().IsEffectiveScalar()))) {
    kind = std::nullopt;
  }
  return kind;
}

std::string_view CollectiveOpGroupModeToString(
    CollectiveOpGroupMode group_mode) {
  switch (group_mode) {
    case CollectiveOpGroupMode::kCrossReplica:
      return "kCrossReplica";
    case CollectiveOpGroupMode::kCrossPartition:
      return "kCrossPartition";
    case CollectiveOpGroupMode::kCrossReplicaAndPartition:
      return "kCrossReplicaAndPartition";
    case CollectiveOpGroupMode::kFlattenedID:
      return "kFlattenedID";
  }
}

absl::StatusOr<std::vector<int>> GetParticipatingIDs(
    CollectiveOpGroupMode group_mode, int current_id,
    std::optional<int> total_participant_count,
    absl::Span<const ReplicaGroup> groups) {
  // Empty replica_groups() means that all replicas participate.
  if (groups.empty()) {
    TF_RET_CHECK(total_participant_count.has_value());
    std::vector<int> all_participants(*total_participant_count);
    absl::c_iota(all_participants, 0);
    return all_participants;
  }

  // Formatter for printing replica groups in StrJoin.
  auto group_formatter = [](std::string* out, const ReplicaGroup& group) {
    out->append("[");
    out->append(absl::StrJoin(group.replica_ids(), ", "));
    out->append("]");
  };

  // Figure out the other replicas that go together with this one.
  std::optional<ReplicaGroup> group;
  for (const ReplicaGroup& g : groups) {
    if (absl::c_linear_search(g.replica_ids(), current_id)) {
      TF_RET_CHECK(!group.has_value())
          << "Replica ID " << current_id << " appears twice in replica groups"
          << "; group_mode=" << CollectiveOpGroupModeToString(group_mode)
          << "; groups_size=" << groups.size()
          << "; groups= " << absl::StrJoin(groups, ", ", group_formatter);
      group = g;
    }
  }
  TF_RET_CHECK(group.has_value())
      << "Replica ID " << current_id << " doesn't appear in replica groups"
      << "; group_mode=" << CollectiveOpGroupModeToString(group_mode)
      << "; groups_size=" << groups.size()
      << "; groups= " << absl::StrJoin(groups, ", ", group_formatter);
  return std::vector<int>(group->replica_ids().begin(),
                          group->replica_ids().end());
}

// Returns the group formation mode implied by (a) whether the operation has
// channel_id and (b) if it has use_global_device_ids and if yes, its value.
absl::StatusOr<CollectiveOpGroupMode> GetCollectiveOpGroupMode(
    bool has_channel_id, std::optional<bool> use_global_device_ids) {
  if (!has_channel_id) {
    if (!use_global_device_ids.has_value() || !*use_global_device_ids) {
      return CollectiveOpGroupMode::kCrossReplica;
    } else {
      return absl::InvalidArgumentError(
          "Invalid combination of has_channel_id and use_global_device_ids");
    }
  } else {
    if (!use_global_device_ids.has_value()) {
      return CollectiveOpGroupMode::kCrossPartition;
    } else if (!*use_global_device_ids) {
      return CollectiveOpGroupMode::kCrossReplicaAndPartition;
    } else {
      return CollectiveOpGroupMode::kFlattenedID;
    }
  }
}

absl::StatusOr<std::vector<GlobalDeviceId>> GetParticipatingDevices(
    GlobalDeviceId device_id, const DeviceAssignment& device_assignment,
    absl::Span<const ReplicaGroup> replica_groups,
    CollectiveOpGroupMode group_mode) {
  int replica_count = device_assignment.replica_count();
  int partition_count = device_assignment.computation_count();

  TF_ASSIGN_OR_RETURN(const DeviceAssignment::LogicalID logical_id,
                      device_assignment.LogicalIdForDevice(device_id));
  int current_replica_id = logical_id.replica_id;
  int current_partition_id = logical_id.computation_id;
  TF_RET_CHECK(0 <= current_replica_id && current_replica_id < replica_count)
      << current_replica_id << " " << replica_count;
  TF_RET_CHECK(0 <= current_partition_id &&
               current_partition_id < partition_count)
      << current_partition_id << " " << partition_count;

  std::vector<GlobalDeviceId> participants;
  switch (group_mode) {
    case CollectiveOpGroupMode::kCrossReplica: {
      // This is a cross replica operation. replica group contains replica id.
      // use current replica id to find the set of participating replicas. If
      // replica groups are empty, assume a group with all replicas.
      TF_ASSIGN_OR_RETURN(std::vector<int> participating_replicas,
                          GetParticipatingIDs(group_mode, current_replica_id,
                                              replica_count, replica_groups));

      // The set of participating devices is the replicas from the current
      // partition.
      participants.reserve(participating_replicas.size());
      for (int replica_id : participating_replicas) {
        TF_RET_CHECK(0 <= replica_id && replica_id < replica_count)
            << replica_id << " " << replica_count;
        participants.emplace_back(
            device_assignment(replica_id, current_partition_id));
      }
      return participants;
    }

    case CollectiveOpGroupMode::kCrossPartition: {
      // replica_groups contain partition_id, group contains all partitions
      // for the current replica.
      TF_ASSIGN_OR_RETURN(std::vector<int> participating_partitions,
                          GetParticipatingIDs(group_mode, current_partition_id,
                                              partition_count, replica_groups));
      participants.reserve(participating_partitions.size());
      for (int partition_id : participating_partitions) {
        TF_RET_CHECK(0 <= partition_id && partition_id < partition_count)
            << partition_id << " " << partition_count;
        participants.emplace_back(
            device_assignment(current_replica_id, partition_id));
      }
      return participants;
    }

    case CollectiveOpGroupMode::kCrossReplicaAndPartition: {
      // replica_groups contain replica_ids. Group contains replicas for all
      // partitions.
      TF_ASSIGN_OR_RETURN(std::vector<int> participating_replicas,
                          GetParticipatingIDs(group_mode, current_replica_id,
                                              replica_count, replica_groups));
      participants.reserve(participating_replicas.size() * partition_count);
      for (int replica_id : participating_replicas) {
        TF_RET_CHECK(0 <= replica_id && replica_id < replica_count)
            << replica_id << " " << replica_count;
        for (int partition_id = 0; partition_id < partition_count;
             ++partition_id) {
          participants.emplace_back(
              device_assignment(replica_id, partition_id));
        }
      }
      return participants;
    }

    case CollectiveOpGroupMode::kFlattenedID: {
      // replica groups contain flattened-ids and cannot be empty.
      TF_RET_CHECK(!replica_groups.empty())
          << "replica groups cannot be empty for kFlattenedID mode";

      int current_flattened_id =
          current_replica_id * partition_count + current_partition_id;

      // Find participants based on flattened id. replica_groups cannot be
      // empty so no need to pass in total_participant_count.
      TF_ASSIGN_OR_RETURN(
          std::vector<int> participating_flattened_ids,
          GetParticipatingIDs(group_mode, current_flattened_id,
                              /*total_participant_count=*/std::nullopt,
                              replica_groups));

      participants.reserve(participating_flattened_ids.size());
      for (int flattened_id : participating_flattened_ids) {
        // Map from flattened id back to replica_id, partition_id.
        int replica_id = flattened_id / partition_count;
        TF_RET_CHECK(0 <= replica_id && replica_id < replica_count)
            << replica_id << " " << replica_count;
        int partition_id = flattened_id % partition_count;
        participants.emplace_back(device_assignment(replica_id, partition_id));
      }
      return participants;
    }
  }
}

absl::StatusOr<std::vector<int64_t>> GetParticipantCountsForReplicaGroups(
    int64_t num_replicas, int64_t num_partitions,
    absl::Span<const ReplicaGroup> replica_groups,
    CollectiveOpGroupMode group_mode) {
  std::vector<int64_t> participant_counts;
  std::vector<ReplicaGroup> participating_replica_groups =
      SpanToVector(replica_groups);

  // If replica groups are empty, assume a group with all replicas.
  if (replica_groups.empty()) {
    if (group_mode == CollectiveOpGroupMode::kFlattenedID) {
      // replica groups contain flattened-ids and cannot be empty.
      TF_RET_CHECK(!replica_groups.empty())
          << "replica groups cannot be empty for kFlattenedID mode";
    }

    int total_participant_count;
    if (group_mode == CollectiveOpGroupMode::kCrossPartition) {
      // replica group are partition ids.
      total_participant_count = num_partitions;
    } else {
      // replica group are replica ids.
      total_participant_count = num_replicas;
    }

    ReplicaGroup replica_group = ReplicaGroup();
    for (int id = 0; id < total_participant_count; id++) {
      replica_group.add_replica_ids(id);
    }
    participating_replica_groups.push_back(replica_group);
  }

  switch (group_mode) {
    case CollectiveOpGroupMode::kCrossReplica: {
      for (const auto& replica_group : participating_replica_groups) {
        for (int partition_id = 0; partition_id < num_partitions;
             ++partition_id) {
          participant_counts.push_back(replica_group.replica_ids().size());
        }
      }
      return participant_counts;
    }
    case CollectiveOpGroupMode::kCrossPartition: {
      for (const auto& replica_group : participating_replica_groups) {
        participant_counts.push_back(replica_group.replica_ids().size());
      }
      return participant_counts;
    }
    case CollectiveOpGroupMode::kCrossReplicaAndPartition: {
      for (const auto& replica_group : participating_replica_groups) {
        participant_counts.push_back(replica_group.replica_ids().size() *
                                     num_partitions);
      }
      return participant_counts;
    }
    case CollectiveOpGroupMode::kFlattenedID: {
      for (const auto& replica_group : participating_replica_groups) {
        participant_counts.push_back(replica_group.replica_ids().size());
      }
      return participant_counts;
    }
  }
}

bool IsNonFusionCollective(const HloInstruction* instruction) {
  switch (instruction->opcode()) {
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectiveBroadcast:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kRaggedAllToAll:
    case HloOpcode::kReduceScatter:
      return true;
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
      return IsNonFusionCollective(instruction->async_wrapped_instruction());
    case HloOpcode::kSend:
    case HloOpcode::kRecv:
      return !Cast<HloSendRecvInstruction>(instruction)->is_host_transfer();
    default:
      return false;
  }
}

bool IsCollective(const HloInstruction* instruction) {
  if (IsNonFusionCollective(instruction)) {
    return true;
  }
  if (instruction->opcode() == HloOpcode::kFusion &&
      instruction->IsCustomFusion()) {
    for (const auto* inner_inst : instruction->fused_instructions()) {
      if (IsCollective(inner_inst)) {
        return true;
      }
    }
  }
  return false;
}

HloInstruction* IsOrHasCollectiveWithChannelId(HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kFusion) {
    for (auto* inner_inst : instruction->fused_instructions()) {
      if (IsOrHasCollectiveWithChannelId(inner_inst) != nullptr) {
        return inner_inst;
      }
    }
    return nullptr;
  }
  if (DynCast<HloChannelInstruction>(instruction) == nullptr) {
    return nullptr;
  }
  if (IsCollective(instruction) && instruction->channel_id().has_value()) {
    return instruction;
  }
  return nullptr;
}

}  // namespace zkx
