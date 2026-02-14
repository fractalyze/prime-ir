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

#ifndef ZKX_SERVICE_COLLECTIVE_OPS_UTILS_H_
#define ZKX_SERVICE_COLLECTIVE_OPS_UTILS_H_

#include <stdint.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"

#include "zkx/executable_run_options.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/service/computation_placer.h"
#include "zkx/service/global_device_id.h"
#include "zkx/zkx_data.pb.h"

namespace zkx {

enum class ReductionKind { kSum, kProduct, kMin, kMax };

std::string_view ReductionKindToString(ReductionKind reduction_kind);

absl::StatusOr<ReductionKind> StringToReductionKind(
    std::string_view reduction_kind);

// Attempts to match instruction to one of the possible cases for ReductionKind.
std::optional<ReductionKind> MatchReductionInstruction(
    const HloInstruction* hlo);

// Attempts to match computation to one of the possible cases in ReductionKind.
std::optional<ReductionKind> MatchReductionComputation(
    const HloComputation* computation);

// There are broadly 4 modes that collective communication ops use to describe
// which sets of devices are participating with a given device in the operation.
// These modes are determined by the values of channel_id (optional) and
// use_global_device_ids (optional). The modes are as follows:
//
// kCrossReplica:
//    implied by: no channel id, use_global_device_ids = false, or
//                no channel_id, no use_global_device_ids:
//    replica_groups contain replica_id, group contains all replicas for the
//    current partition
//
// kCrossPartition:
//    implied by: channel_id is set, no use_global_device_ids:
//    replica_groups contain partition_id, group contains all partitions for the
//    current replica.
//
// kCrossReplicaAndPartition:
//    implied by: channel_id is set, use_global_device_ids = false:
//    replica_groups contain replica_id, group contains all replicas for all
//    partitions (as opposed to just current partition).
//
// kFlattenedID:
//    implied by: channel_id is set, use_global_device_ids = true:
//    replica_groups contain flattened-ids, group contains devices that are
//    listed in the flattened-id list.
//
// Rest of the combinations are invalid.
//
// Since the actual value of channel_id does not matter, we use a bool argument
// `has_channel_id`, and optional<bool> for use_global_device_ids.
// Note that use_global_device_ids true requires channel_id to be set as well.
// Additionally, if use_global_device_ids = true, replica groups cannot be
// empty (verified in the HLO verifier).
enum class CollectiveOpGroupMode {
  kCrossReplica,
  kCrossPartition,
  kCrossReplicaAndPartition,
  kFlattenedID,
};

std::string_view CollectiveOpGroupModeToString(
    CollectiveOpGroupMode group_mode);

// Figures out which IDs are participating in the collective subgroup.
// An empty `groups` indicates that all [0, total_participant_count) IDs
// are participating. Note that for CollectiveOpGroupMode::kFlattenedID,
// groups cannot be empty, so `total_participant_count` is an optional.
absl::StatusOr<std::vector<int>> GetParticipatingIDs(
    CollectiveOpGroupMode group_mode, int current_id,
    std::optional<int> total_participant_count,
    absl::Span<const ReplicaGroup> groups);

// Returns the group formation mode implied by (a) whether the operation has
// channel_id and (b) if it has use_global_device_ids and if yes, its value.
absl::StatusOr<CollectiveOpGroupMode> GetCollectiveOpGroupMode(
    bool has_channel_id, std::optional<bool> use_global_device_ids);

// Figures out which devices are participating in the collective subgroup.
absl::StatusOr<std::vector<GlobalDeviceId>> GetParticipatingDevices(
    GlobalDeviceId device_id, const DeviceAssignment& device_assignment,
    absl::Span<const ReplicaGroup> replica_groups,
    CollectiveOpGroupMode group_mode);

// Figures out how many ranks are participating in each collective subgroup.
absl::StatusOr<std::vector<int64_t>> GetParticipantCountsForReplicaGroups(
    int64_t num_replicas, int64_t num_partitions,
    absl::Span<const ReplicaGroup> replica_groups,
    CollectiveOpGroupMode group_mode);

// Key that identifies a particular Rendezvous object in our global hashtable.
// This determines which calls to ExecuteOnStream communicate with each other.
// The rules are as follows.
//
// * Only ops with the same RunId can communicate with each other. (This is the
//   whole purpose of RunId).
//
// * Only ops with the same set of participating replicas can communicate with
//   each other.  This is how we separate out different replica groups (e.g. a
//   single AllReduce HLO might do two reductions, between say GPUs {0,2} and
//   {1,3}).
//
// * Only ops with the same opcode can communicate with each other. At the
//   moment we only support kAllReduce, so we don't check for this explicitly.
//
// * For cross-module all-reduces (i.e. instr->channel_id().has_value()),
//   only ops with the same value for channel_id() can communicate with each
//   other.
//
// * For cross-replica (i.e. same-module) all-reduces (i.e.
//   !channel_id().has_value()), only ops from the same module (as
//   identified by its unique_id()) can communicate with each other.
//
struct RendezvousKey {
  enum CollectiveOpKind {
    kCrossModule,
    kCrossReplica,
  };

  explicit RendezvousKey(const RunId& run_id,
                         std::vector<GlobalDeviceId> global_devices,
                         int num_local_participants,
                         CollectiveOpKind collective_op_kind, int64_t op_id)
      : run_id(run_id),
        global_devices(std::move(global_devices)),
        num_local_participants(num_local_participants),
        collective_op_kind(collective_op_kind),
        op_id(op_id) {}

  template <typename H>
  friend H AbslHashValue(H h, const RendezvousKey& k) {
    return H::combine(std::move(h), k.run_id, k.global_devices,
                      k.num_local_participants, k.collective_op_kind, k.op_id);
  }
  friend bool operator==(const RendezvousKey& a, const RendezvousKey& b) {
    return a.run_id == b.run_id && a.global_devices == b.global_devices &&
           a.num_local_participants == b.num_local_participants &&
           a.collective_op_kind == b.collective_op_kind &&  //
           a.op_id == b.op_id;
  }
  friend bool operator!=(const RendezvousKey& a, const RendezvousKey& b) {
    return !(a == b);
  }

  std::string_view CollectiveOpKindString() const {
    switch (collective_op_kind) {
      case kCrossModule:
        return "cross_module";
      case kCrossReplica:
        return "cross_replica";
    }
  }

  std::string ToString() const {
    return absl::StrFormat(
        "RendezvousKey{run_id=%s, global_devices=[%s], "
        "num_local_participants=%d, collective_op_kind=%s, op_id=%d}",
        run_id.ToString(), GlobalDeviceIdsToString(global_devices),
        num_local_participants, CollectiveOpKindString(), op_id);
  }

  RunId run_id;
  std::vector<GlobalDeviceId> global_devices;
  int num_local_participants;
  CollectiveOpKind collective_op_kind;
  int64_t op_id;
};

// Returns true if instruction is a collective op that is not a collective
// fusion.
bool IsNonFusionCollective(const HloInstruction* instruction);

// Returns true if instruction is a collective op or a collective fusion.
bool IsCollective(const HloInstruction* instruction);

// Returns the collective instruction if argument is a collective op (or a
// collective fusion) with channel_id.
HloInstruction* IsOrHasCollectiveWithChannelId(HloInstruction* instruction);

// When a Send or Recv is annotated with frontend attribute
// _zkx_send_recv_pipeline="1", asynchronous stream kP2P1 is used to execute the
// Send or Recv. For all other cases, asynchronous stream kP2P0 is used.
constexpr char kSendRecvPipelineAttr[] = "_zkx_send_recv_pipeline";

}  // namespace zkx

#endif  // ZKX_SERVICE_COLLECTIVE_OPS_UTILS_H_
