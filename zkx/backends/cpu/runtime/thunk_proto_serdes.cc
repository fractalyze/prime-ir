/* Copyright 2025 The OpenXLA Authors.
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

#include "zkx/backends/cpu/runtime/thunk_proto_serdes.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_format.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
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
#include "zkx/backends/cpu/runtime/resource_use.h"
#include "zkx/backends/cpu/runtime/sort_thunk.h"
#include "zkx/backends/cpu/runtime/while_thunk.h"
#include "zkx/service/collective_ops_utils.h"

namespace zkx::cpu {
namespace {

Thunk::Kind ProtoThunkToThunkKind(const ThunkProto& proto) {
  auto collective_proto_kind_to_kind =
      [](const CollectiveThunkProto::ImplCase& proto_kind) {
        switch (proto_kind) {
          case CollectiveThunkProto::ImplCase::kAllGatherThunk:
            return Thunk::Kind::kAllGather;
          case CollectiveThunkProto::ImplCase::kAllReduceThunk:
            return Thunk::Kind::kAllReduce;
          case CollectiveThunkProto::ImplCase::kAllToAllThunk:
            return Thunk::Kind::kAllToAll;
          case CollectiveThunkProto::ImplCase::kCollectivePermuteThunk:
            return Thunk::Kind::kCollectivePermute;
          case CollectiveThunkProto::ImplCase::kReduceScatterThunk:
            return Thunk::Kind::kReduceScatter;
          default:
            return Thunk::Kind::kUnknown;
        }
      };

  switch (proto.impl_case()) {
    case ThunkProto::ImplCase::kCollectiveThunk:
      return collective_proto_kind_to_kind(
          proto.collective_thunk().impl_case());
    case ThunkProto::ImplCase::kCallThunk:
      return Thunk::Kind::kCall;
    case ThunkProto::ImplCase::kConditionalThunk:
      return Thunk::Kind::kConditional;
    case ThunkProto::ImplCase::kCopyThunk:
      return Thunk::Kind::kCopy;
    case ThunkProto::ImplCase::kInfeedThunk:
      return Thunk::Kind::kInfeed;
    case ThunkProto::ImplCase::kKernelThunk:
      return Thunk::Kind::kKernel;
    case ThunkProto::ImplCase::kOutfeedThunk:
      return Thunk::Kind::kOutfeed;
    case ThunkProto::ImplCase::kSortThunk:
      return Thunk::Kind::kSort;
    case ThunkProto::ImplCase::kWhileThunk:
      return Thunk::Kind::kWhile;
    default:
      return Thunk::Kind::kUnknown;
  }
}

absl::StatusOr<std::shared_ptr<Resource>> CreateResourceFromProto(
    const ResourceProto& proto) {
  switch (proto.kind()) {
    case ResourceProto::TOKEN:
      return Resource::Create(Resource::kToken);
    case ResourceProto::COLLECTIVE_COMMUNICATOR:
      return Resource::Create(Resource::kCollectiveCommunicator);
    default:
      return absl::UnimplementedError("Resource kind not supported.");
  }
}

absl::StatusOr<ResourceProto> ToProto(const Resource& resource) {
  ResourceProto proto;
  switch (resource.kind()) {
    case Resource::kToken:
      proto.set_kind(ResourceProto::TOKEN);
      break;
    case Resource::kCollectiveCommunicator:
      proto.set_kind(ResourceProto::COLLECTIVE_COMMUNICATOR);
      break;
    default:
      return absl::UnimplementedError("Resource kind not supported.");
  }
  return proto;
}

InfoProto ThunkInfoToProto(const Thunk::Info& info) {
  InfoProto proto;
  proto.set_op_name(info.op_name);
  proto.set_module_name(info.module_name);
  proto.set_module_id(info.module_id);
  return proto;
}

absl::StatusOr<Thunk::Info> ThunkInfoFromProto(const InfoProto& proto) {
  Thunk::Info info;
  info.op_name = proto.op_name();
  info.module_name = proto.module_name();
  info.module_id = proto.module_id();
  return info;
}

absl::StatusOr<CollectiveThunk::OpParams> OpParamsFromProto(
    const OpParamsProto& proto) {
  CollectiveThunk::OpParams op_params;
  op_params.has_channel_id = proto.has_channel_id();
  if (proto.use_global_device_ids().contains_value()) {
    op_params.use_global_device_ids = proto.use_global_device_ids().value();
  } else {
    op_params.use_global_device_ids = std::nullopt;
  }
  op_params.op_id = proto.op_id();
  for (const auto& replica_group : proto.replica_group()) {
    ReplicaGroup group;
    for (const auto& replica_id : replica_group.replica_ids()) {
      group.add_replica_ids(replica_id);
    }
    op_params.group.push_back(group);
  }
  return op_params;
}

absl::StatusOr<BufferAllocationSliceProto> SerializeSliceIntoProto(
    const BufferAllocation::Slice& slice) {
  BufferAllocationSliceProto proto;
  proto.set_offset(slice.offset());
  proto.set_size(slice.size());
  proto.set_buffer_allocation_index(
      slice.allocation() == nullptr ? -1 : slice.index());
  return proto;
}

absl::StatusOr<BufferAllocation::Slice> DeserializeSliceFromProto(
    const BufferAllocationSliceProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  const BufferAllocation& allocation =
      buffer_allocations[proto.buffer_allocation_index()];
  return BufferAllocation::Slice(&allocation, proto.offset(), proto.size());
}

absl::Status SerializeSliceShapeIntoProto(
    const BufferAllocation::Slice& slice, const Shape& shape,
    ShapeBufferAllocationSliceProto* proto) {
  *proto->mutable_shape() = shape.ToProto();
  TF_ASSIGN_OR_RETURN(*proto->mutable_slice(), SerializeSliceIntoProto(slice));
  return absl::OkStatus();
}

absl::StatusOr<std::pair<BufferAllocation::Slice, Shape>>
DeserializeSliceShapeFromProto(
    const ShapeBufferAllocationSliceProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice slice,
      DeserializeSliceFromProto(proto.slice(), buffer_allocations));
  Shape shape(proto.shape());
  return std::make_pair(slice, shape);
}

absl::StatusOr<std::tuple<CollectiveThunk::OpParams, CollectiveThunk::OpBuffers,
                          CollectiveThunk::OpResources>>
GetCollectiveThunkParamsFromProto(
    const CollectiveThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  TF_ASSIGN_OR_RETURN(CollectiveThunk::OpParams op_params,
                      OpParamsFromProto(proto.op_params()));

  CollectiveThunk::OpBuffers op_buffers;
  for (const auto& shape_buffer_slice_proto :
       proto.op_buffers().source_shapes_buffer_slices()) {
    TF_ASSIGN_OR_RETURN(auto slice_shape,
                        DeserializeSliceShapeFromProto(shape_buffer_slice_proto,
                                                       buffer_allocations));
    const auto& [slice, shape] = slice_shape;
    op_buffers.source_buffers.push_back(slice);
    op_buffers.source_shapes.push_back(shape);
  }

  for (const auto& shape_buffer_slice_proto :
       proto.op_buffers().destination_shapes_buffer_slices()) {
    TF_ASSIGN_OR_RETURN(auto slice_shape,
                        DeserializeSliceShapeFromProto(shape_buffer_slice_proto,
                                                       buffer_allocations));

    const auto& [slice, shape] = slice_shape;
    op_buffers.destination_buffers.push_back(slice);
    op_buffers.destination_shapes.push_back(shape);
  }

  CollectiveThunk::OpResources op_resources;
  if (proto.op_resources().communicator_resource().has_value()) {
    TF_ASSIGN_OR_RETURN(
        op_resources.communicator_resource,
        CreateResourceFromProto(
            proto.op_resources().communicator_resource().value()));
  } else {
    op_resources.communicator_resource = nullptr;
  }

  return std::make_tuple(op_params, op_buffers, op_resources);
}

absl::StatusOr<OpParamsProto> ToProto(
    const CollectiveThunk::OpParams& op_params) {
  OpParamsProto proto;
  proto.set_has_channel_id(op_params.has_channel_id);

  proto.mutable_use_global_device_ids()->set_contains_value(
      op_params.use_global_device_ids.has_value());
  if (op_params.use_global_device_ids) {
    proto.mutable_use_global_device_ids()->set_value(
        *op_params.use_global_device_ids);
  }

  proto.set_op_id(op_params.op_id);
  for (const auto& group : op_params.group) {
    ReplicaGroup* replica_group = proto.add_replica_group();
    for (const auto& device : group.replica_ids()) {
      replica_group->add_replica_ids(device);
    }
  }
  return proto;
}

class ThunkSerDesProtobuf : public SerDesBase<Thunk> {
  friend class zkx::cpu::ThunkSequenceSerDesProtobuf;

 public:
  explicit ThunkSerDesProtobuf(
      const std::vector<BufferAllocation>* buffer_allocations =
          nullptr);  // NOTE buffer assignment isn't
                     // needed for serialization.
  absl::StatusOr<std::string> Serialize(const Thunk& thunk) override;
  absl::StatusOr<std::unique_ptr<Thunk>> Deserialize(
      const std::string& serialized) override;

 protected:
  absl::StatusOr<ThunkProto> ToProto(const Thunk& thunk) const;
  absl::StatusOr<std::unique_ptr<Thunk>> FromProto(
      const ThunkProto& proto) const;

 private:
  const std::vector<BufferAllocation>* buffer_allocations_;
};

ThunkSerDesProtobuf::ThunkSerDesProtobuf(
    const std::vector<BufferAllocation>* buffer_allocations)
    : buffer_allocations_(buffer_allocations) {}

absl::StatusOr<std::string> ThunkSerDesProtobuf::Serialize(const Thunk& thunk) {
  TF_ASSIGN_OR_RETURN(ThunkProto proto, ToProto(thunk));
  return proto.SerializeAsString();
}

absl::StatusOr<std::unique_ptr<Thunk>> ThunkSerDesProtobuf::Deserialize(
    const std::string& serialized) {
  ThunkProto proto;
  if (!proto.ParseFromString(serialized)) {
    return absl::InternalError(
        absl::StrFormat("Failed to parse thunk proto:\n %s", serialized));
  }
  return FromProto(proto);
}

absl::Status ToProto(const AllGatherThunk& thunk, AllGatherThunkProto& proto) {
  // NOTE(basioli) AllGatherThunkProto has no extra fields to serialize.
  return absl::OkStatus();
}

absl::Status ToProto(const AllReduceThunk& thunk, AllReduceThunkProto& proto) {
  std::string_view reduction_kind_as_string_view =
      ReductionKindToString(thunk.reduction_kind());
  std::string reduction_kind_as_string(reduction_kind_as_string_view.begin(),
                                       reduction_kind_as_string_view.end());
  proto.set_reduction_kind(reduction_kind_as_string);
  proto.set_single_replica(thunk.single_replica());
  return absl::OkStatus();
}

absl::Status ToProto(const AllToAllThunk& thunk, AllToAllThunkProto& proto) {
  // NOTE(basioli) AllToAllThunkProto has no extra fields to serialize.
  return absl::OkStatus();
}

absl::Status ToProto(const ReduceScatterThunk& thunk,
                     ReduceScatterThunkProto& proto) {
  std::string_view reduction_kind_as_string_view =
      ReductionKindToString(thunk.reduction_kind());
  std::string reduction_kind_as_string(reduction_kind_as_string_view.begin(),
                                       reduction_kind_as_string_view.end());
  proto.set_reduction_kind(reduction_kind_as_string);
  return absl::OkStatus();
}

absl::Status ToProto(const CollectivePermuteThunk& thunk,
                     CollectivePermuteThunkProto& proto) {
  for (const auto& source_target_pair : thunk.source_target_pairs()) {
    CollectivePermuteThunkProto::SourceTargetPairProto*
        source_target_pair_proto = proto.add_source_target_pairs();
    source_target_pair_proto->set_source(source_target_pair.first);
    source_target_pair_proto->set_target(source_target_pair.second);
  }
  return absl::OkStatus();
}

absl::Status ToProto(const CollectiveThunk& thunk, ThunkProto& proto) {
  CollectiveThunkProto* collective_thunk_proto =
      proto.mutable_collective_thunk();

  TF_ASSIGN_OR_RETURN(*collective_thunk_proto->mutable_op_params(),
                      ToProto(thunk.op_params()));

  collective_thunk_proto->mutable_op_resources()
      ->mutable_communicator_resource()
      ->set_contains_value(thunk.op_resources().communicator_resource !=
                           nullptr);
  if (thunk.op_resources().communicator_resource != nullptr) {
    TF_ASSIGN_OR_RETURN(*collective_thunk_proto->mutable_op_resources()
                             ->mutable_communicator_resource()
                             ->mutable_value(),
                        ToProto(*thunk.op_resources().communicator_resource));
  }

  for (size_t i = 0; i < thunk.op_buffers().source_buffers.size(); ++i) {
    TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
        thunk.op_buffers().source_buffers[i],
        thunk.op_buffers().source_shapes[i],
        collective_thunk_proto->mutable_op_buffers()
            ->add_source_shapes_buffer_slices()));
  }

  for (size_t i = 0; i < thunk.op_buffers().destination_buffers.size(); ++i) {
    TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
        thunk.op_buffers().destination_buffers[i],
        thunk.op_buffers().destination_shapes[i],
        collective_thunk_proto->mutable_op_buffers()
            ->add_destination_shapes_buffer_slices()));
  }

  if (proto.kind() == Thunk::KindToString(Thunk::Kind::kAllGather)) {
    TF_RETURN_IF_ERROR(
        ToProto(static_cast<const AllGatherThunk&>(thunk),
                *collective_thunk_proto->mutable_all_gather_thunk()));
  } else if (proto.kind() == Thunk::KindToString(Thunk::Kind::kAllReduce)) {
    TF_RETURN_IF_ERROR(
        ToProto(static_cast<const AllReduceThunk&>(thunk),
                *collective_thunk_proto->mutable_all_reduce_thunk()));
  } else if (proto.kind() == Thunk::KindToString(Thunk::Kind::kAllToAll)) {
    TF_RETURN_IF_ERROR(
        ToProto(static_cast<const AllToAllThunk&>(thunk),
                *collective_thunk_proto->mutable_all_to_all_thunk()));
  } else if (proto.kind() == Thunk::KindToString(Thunk::Kind::kReduceScatter)) {
    TF_RETURN_IF_ERROR(
        ToProto(static_cast<const ReduceScatterThunk&>(thunk),
                *collective_thunk_proto->mutable_reduce_scatter_thunk()));
  } else if (proto.kind() ==
             Thunk::KindToString(Thunk::Kind::kCollectivePermute)) {
    TF_RETURN_IF_ERROR(
        ToProto(static_cast<const CollectivePermuteThunk&>(thunk),
                *collective_thunk_proto->mutable_collective_permute_thunk()));
  } else {
    return absl::UnimplementedError(
        "SerializeAsStringCollectiveImpl not implemented");
  }

  return absl::OkStatus();
}

absl::Status ToProto(const CallThunk& thunk, ThunkProto& proto) {
  ThunkSequenceSerDesProtobuf thunk_sequence_serdes;
  CallThunkProto* call_thunk_proto = proto.mutable_call_thunk();

  TF_ASSIGN_OR_RETURN(
      *call_thunk_proto->mutable_called_sequence(),
      thunk_sequence_serdes.ToProto(thunk.called_executor().thunk_sequence()));
  return absl::OkStatus();
}

absl::Status ToProto(const CopyThunk& thunk, ThunkProto& proto) {
  CopyThunkProto* copy_thunk_proto = proto.mutable_copy_thunk();

  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      thunk.src_buffer(), thunk.src_shape(),
      copy_thunk_proto->mutable_src_buffer_shape()));
  TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      thunk.dst_buffer(), thunk.dst_shape(),
      copy_thunk_proto->mutable_dst_buffer_shape()));
  return absl::OkStatus();
}

absl::Status ToProto(const InfeedThunk& thunk, ThunkProto& proto) {
  InfeedThunkProto* infeed_thunk_proto = proto.mutable_infeed_thunk();

  infeed_thunk_proto->mutable_infeed_resources()
      ->mutable_consume_token()
      ->set_contains_value(thunk.infeed_resources().consume_token != nullptr);
  if (thunk.infeed_resources().consume_token != nullptr) {
    TF_ASSIGN_OR_RETURN(*infeed_thunk_proto->mutable_infeed_resources()
                             ->mutable_consume_token()
                             ->mutable_value(),
                        ToProto(*thunk.infeed_resources().consume_token));
  }

  infeed_thunk_proto->mutable_infeed_resources()
      ->mutable_produce_token()
      ->set_contains_value(thunk.infeed_resources().produce_token != nullptr);
  if (thunk.infeed_resources().produce_token != nullptr) {
    TF_ASSIGN_OR_RETURN(*infeed_thunk_proto->mutable_infeed_resources()
                             ->mutable_produce_token()
                             ->mutable_value(),
                        ToProto(*thunk.infeed_resources().produce_token));
  }

  for (const InfeedThunk::InfeedBuffer& infeed_buffer :
       thunk.infeed_buffers()) {
    TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
        infeed_buffer.slice, infeed_buffer.shape,
        infeed_thunk_proto->add_infeed_buffers_shapes()));
  }
  return absl::OkStatus();
}

absl::Status ToProto(const OutfeedThunk& thunk, ThunkProto& proto) {
  OutfeedThunkProto* outfeed_thunk_proto = proto.mutable_outfeed_thunk();
  outfeed_thunk_proto->mutable_outfeed_resources()
      ->mutable_consume_token()
      ->set_contains_value(thunk.outfeed_resources().consume_token != nullptr);
  if (thunk.outfeed_resources().consume_token != nullptr) {
    TF_ASSIGN_OR_RETURN(*outfeed_thunk_proto->mutable_outfeed_resources()
                             ->mutable_consume_token()
                             ->mutable_value(),
                        ToProto(*thunk.outfeed_resources().consume_token));
  }

  outfeed_thunk_proto->mutable_outfeed_resources()
      ->mutable_produce_token()
      ->set_contains_value(thunk.outfeed_resources().produce_token != nullptr);
  if (thunk.outfeed_resources().produce_token != nullptr) {
    TF_ASSIGN_OR_RETURN(*outfeed_thunk_proto->mutable_outfeed_resources()
                             ->mutable_produce_token()
                             ->mutable_value(),
                        ToProto(*thunk.outfeed_resources().produce_token));
  }

  for (const OutfeedThunk::OutfeedBuffer& outfeed_buffer :
       thunk.outfeed_buffers()) {
    TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
        outfeed_buffer.slice, outfeed_buffer.shape,
        outfeed_thunk_proto->add_outfeed_buffers_shapes()));
  }
  return absl::OkStatus();
}

absl::Status ToProto(const SortThunk& thunk, ThunkProto& proto) {
  SortThunkProto* sort_thunk_proto = proto.mutable_sort_thunk();

  sort_thunk_proto->set_dimension(thunk.dimension());
  sort_thunk_proto->set_is_stable(thunk.is_stable());
  sort_thunk_proto->set_comparator_name(thunk.comparator_name());
  sort_thunk_proto->mutable_direction()->set_contains_value(
      thunk.direction().has_value());
  if (thunk.direction().has_value()) {
    switch (*thunk.direction()) {
      case SortThunk::SortDirection::kAscending:
        sort_thunk_proto->mutable_direction()->set_value(
            SortDirectionProto::ASCENDING);
        break;
      case SortThunk::SortDirection::kDescending:
        sort_thunk_proto->mutable_direction()->set_value(
            SortDirectionProto::DESCENDING);
        break;
    }
  }

  if (thunk.has_less_than()) {
    return absl::UnimplementedError("`LessThan` is not serializable.");
  }

  for (const SortThunk::Input& input : thunk.inputs()) {
    TF_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
        input.slice, input.shape, sort_thunk_proto->add_inputs_shapes()));
  }

  return absl::OkStatus();
}

absl::Status ToProto(const WhileThunk& thunk, ThunkProto& proto) {
  ThunkSequenceSerDesProtobuf thunk_sequence_serdes;
  WhileThunkProto* while_thunk_proto = proto.mutable_while_thunk();
  while_thunk_proto->mutable_trip_count()->set_contains_value(
      thunk.trip_count().has_value());
  if (thunk.trip_count().has_value()) {
    while_thunk_proto->mutable_trip_count()->set_value(*thunk.trip_count());
  }

  TF_ASSIGN_OR_RETURN(
      *while_thunk_proto->mutable_cond_sequence(),
      thunk_sequence_serdes.ToProto(thunk.cond_executor().thunk_sequence()));

  TF_ASSIGN_OR_RETURN(
      *while_thunk_proto->mutable_body_sequence(),
      thunk_sequence_serdes.ToProto(thunk.body_executor().thunk_sequence()));

  TF_ASSIGN_OR_RETURN(*while_thunk_proto->mutable_cond_buffer(),
                      SerializeSliceIntoProto(thunk.cond_buffer()));
  return absl::OkStatus();
}

absl::Status ToProto(const KernelThunkBase& thunk, ThunkProto& proto) {
  KernelThunkProto* kernel_thunk_proto = proto.mutable_kernel_thunk();

  // NOTE OSS doesn't accept string_view as a parameter to set_kernel_name
  const std::string_view kernel_name = thunk.kernel_name();
  const std::string kernel_name_str(kernel_name.begin(), kernel_name.end());
  kernel_thunk_proto->set_kernel_name(kernel_name_str);
  kernel_thunk_proto->mutable_thread_dim()->set_x(thunk.thread_dim().x);
  kernel_thunk_proto->mutable_thread_dim()->set_y(thunk.thread_dim().y);
  kernel_thunk_proto->mutable_thread_dim()->set_z(thunk.thread_dim().z);
  kernel_thunk_proto->mutable_min_alignment()->set_contains_value(
      thunk.min_alignment().has_value());
  if (thunk.min_alignment().has_value()) {
    kernel_thunk_proto->mutable_min_alignment()->set_value(
        *thunk.min_alignment());
  }

  for (const BufferAllocation::Slice& buffer : thunk.arguments_buffers()) {
    TF_ASSIGN_OR_RETURN(*kernel_thunk_proto->add_arguments_buffers(),
                        SerializeSliceIntoProto(buffer));
  }

  for (const BufferAllocation::Slice& buffer : thunk.results_buffers()) {
    TF_ASSIGN_OR_RETURN(*kernel_thunk_proto->add_results_buffers(),
                        SerializeSliceIntoProto(buffer));
  }
  return absl::OkStatus();
}

absl::Status ToProto(const ConditionalThunk& thunk, ThunkProto& proto) {
  ConditionalThunkProto* conditional_thunk_proto =
      proto.mutable_conditional_thunk();
  ThunkSequenceSerDesProtobuf thunk_sequence_serdes;

  conditional_thunk_proto->mutable_branch_sequences()->Reserve(
      thunk.branch_executors().size());
  for (const auto& branch_executor : thunk.branch_executors()) {
    TF_ASSIGN_OR_RETURN(
        *conditional_thunk_proto->add_branch_sequences(),
        thunk_sequence_serdes.ToProto(branch_executor.thunk_sequence()));
  }

  TF_ASSIGN_OR_RETURN(*conditional_thunk_proto->mutable_branch_index_buffer(),
                      SerializeSliceIntoProto(thunk.branch_index_buffer()));
  return absl::OkStatus();
}

absl::StatusOr<ThunkProto> ThunkSerDesProtobuf::ToProto(
    const Thunk& thunk) const {
  ThunkProto proto;
  // NOTE OSS doesn't accept string_view as a parameter to set_kind
  const auto kind_as_str_view = Thunk::KindToString(thunk.kind());
  const std::string kind_as_str(kind_as_str_view);
  proto.set_kind(kind_as_str);
  *proto.mutable_info() = ThunkInfoToProto(thunk.info());
  switch (thunk.kind()) {
    // NOTE collective thunks
    case Thunk::Kind::kAllGather:
    case Thunk::Kind::kAllReduce:
    case Thunk::Kind::kAllToAll:
    case Thunk::Kind::kCollectivePermute:
    case Thunk::Kind::kReduceScatter:
      TF_RETURN_IF_ERROR(::zkx::cpu::ToProto(
          static_cast<const CollectiveThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kConditional:
      TF_RETURN_IF_ERROR(::zkx::cpu::ToProto(
          static_cast<const ConditionalThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kKernel:
      TF_RETURN_IF_ERROR(::zkx::cpu::ToProto(
          static_cast<const KernelThunkBase&>(thunk), proto));
      break;
    case Thunk::Kind::kCall:
      TF_RETURN_IF_ERROR(
          ::zkx::cpu::ToProto(static_cast<const CallThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kCopy:
      TF_RETURN_IF_ERROR(
          ::zkx::cpu::ToProto(static_cast<const CopyThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kInfeed:
      TF_RETURN_IF_ERROR(
          ::zkx::cpu::ToProto(static_cast<const InfeedThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kOutfeed:
      TF_RETURN_IF_ERROR(
          ::zkx::cpu::ToProto(static_cast<const OutfeedThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kSort:
      TF_RETURN_IF_ERROR(
          ::zkx::cpu::ToProto(static_cast<const SortThunk&>(thunk), proto));
      break;
    case Thunk::Kind::kWhile:
      TF_RETURN_IF_ERROR(
          ::zkx::cpu::ToProto(static_cast<const WhileThunk&>(thunk), proto));
      break;
    default:
      return absl::UnimplementedError(
          absl::StrFormat("ToProto is not implemented for thunk kind: %s",
                          Thunk::KindToString(thunk.kind())));
  }
  return proto;
}

absl::StatusOr<std::unique_ptr<AllGatherThunk>> AllGatherThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  TF_ASSIGN_OR_RETURN(auto collective_thunk_params,
                      GetCollectiveThunkParamsFromProto(
                          proto.collective_thunk(), buffer_allocations));

  const auto& [op_params, op_buffers, op_resources] = collective_thunk_params;
  return AllGatherThunk::Create(info, op_params, op_buffers, op_resources);
}

absl::StatusOr<std::unique_ptr<AllReduceThunk>> AllReduceThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  TF_ASSIGN_OR_RETURN(auto collective_thunk_params,
                      GetCollectiveThunkParamsFromProto(
                          proto.collective_thunk(), buffer_allocations));

  const auto& [op_params, op_buffers, op_resources] = collective_thunk_params;
  TF_ASSIGN_OR_RETURN(
      ReductionKind reduction_kind,
      StringToReductionKind(
          proto.collective_thunk().all_reduce_thunk().reduction_kind()));

  return AllReduceThunk::Create(
      info, reduction_kind, op_params, op_buffers, op_resources,
      proto.collective_thunk().all_reduce_thunk().single_replica());
}

absl::StatusOr<std::unique_ptr<AllToAllThunk>> AllToAllThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));
  TF_ASSIGN_OR_RETURN(auto collective_thunk_params,
                      GetCollectiveThunkParamsFromProto(
                          proto.collective_thunk(), buffer_allocations));

  const auto& [op_params, op_buffers, op_resources] = collective_thunk_params;
  return AllToAllThunk::Create(info, op_params, op_buffers, op_resources);
}

absl::StatusOr<std::unique_ptr<CollectivePermuteThunk>>
CollectivePermuteThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  TF_ASSIGN_OR_RETURN(auto collective_thunk_params,
                      GetCollectiveThunkParamsFromProto(
                          proto.collective_thunk(), buffer_allocations));

  const auto& [op_params, op_buffers, op_resources] = collective_thunk_params;
  std::vector<CollectivePermuteThunk::SourceTargetPair> source_target_pairs;
  for (const auto& source_target_pair_proto : proto.collective_thunk()
                                                  .collective_permute_thunk()
                                                  .source_target_pairs()) {
    source_target_pairs.push_back(
        {source_target_pair_proto.source(), source_target_pair_proto.target()});
  }
  return CollectivePermuteThunk::Create(info, op_params, op_buffers,
                                        op_resources, source_target_pairs);
}

absl::StatusOr<std::unique_ptr<ReduceScatterThunk>> ReduceScatterThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  TF_ASSIGN_OR_RETURN(auto collective_thunk_params,
                      GetCollectiveThunkParamsFromProto(
                          proto.collective_thunk(), buffer_allocations));

  const auto& [op_params, op_buffers, op_resources] = collective_thunk_params;

  TF_ASSIGN_OR_RETURN(
      ReductionKind reduction_kind,
      StringToReductionKind(
          proto.collective_thunk().reduce_scatter_thunk().reduction_kind()));
  return ReduceScatterThunk::Create(info, reduction_kind, op_params, op_buffers,
                                    op_resources);
}

absl::StatusOr<std::unique_ptr<CallThunk>> CallThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  ThunkSequenceSerDesProtobuf thunk_sequence_serdes(&buffer_allocations);

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<ThunkSequence> call_sequence,
      thunk_sequence_serdes.FromProto(proto.call_thunk().called_sequence()));
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  return CallThunk::Create(std::move(info), std::move(*call_sequence));
}

absl::StatusOr<std::unique_ptr<ConditionalThunk>> ConditionalThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  ThunkSequenceSerDesProtobuf thunk_sequence_serdes(&buffer_allocations);

  std::vector<ThunkSequence> branch_sequences;
  for (const ThunkSequenceProto& branch_sequence_proto :
       proto.conditional_thunk().branch_sequences()) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<ThunkSequence> branch_sequence,
                        thunk_sequence_serdes.FromProto(branch_sequence_proto));
    branch_sequences.push_back(std::move(*branch_sequence));
  }
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice branch_index_buffer,
      DeserializeSliceFromProto(proto.conditional_thunk().branch_index_buffer(),
                                buffer_allocations));

  return ConditionalThunk::Create(std::move(info),
                                  std::move(branch_index_buffer),
                                  std::move(branch_sequences));
}

absl::StatusOr<std::unique_ptr<CopyThunk>> CopyThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  TF_ASSIGN_OR_RETURN(
      auto src_slice_shape,
      DeserializeSliceShapeFromProto(proto.copy_thunk().src_buffer_shape(),
                                     buffer_allocations));
  TF_ASSIGN_OR_RETURN(
      auto dst_slice_shape,
      DeserializeSliceShapeFromProto(proto.copy_thunk().dst_buffer_shape(),
                                     buffer_allocations));

  const auto& [src_buffer, src_shape] = src_slice_shape;
  const auto& [dst_buffer, dst_shape] = dst_slice_shape;

  return CopyThunk::Create(std::move(info), std::move(src_buffer), src_shape,
                           std::move(dst_buffer), dst_shape);
}

absl::StatusOr<std::unique_ptr<InfeedThunk>> InfeedThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  std::vector<InfeedThunk::InfeedBuffer> infeed_buffers;
  for (const ShapeBufferAllocationSliceProto& infeed_buffer_shape :
       proto.infeed_thunk().infeed_buffers_shapes()) {
    TF_ASSIGN_OR_RETURN(auto infeed_buffer_slice_shape,
                        DeserializeSliceShapeFromProto(infeed_buffer_shape,
                                                       buffer_allocations));

    const auto& [infeed_buffer, infeed_shape] = infeed_buffer_slice_shape;
    infeed_buffers.push_back(
        {std::move(infeed_buffer), std::move(infeed_shape)});
  }

  InfeedThunk::InfeedResources infeed_resources;
  if (proto.infeed_thunk()
          .infeed_resources()
          .consume_token()
          .contains_value()) {
    TF_ASSIGN_OR_RETURN(
        infeed_resources.consume_token,
        CreateResourceFromProto(
            proto.infeed_thunk().infeed_resources().consume_token().value()));
  } else {
    infeed_resources.consume_token = nullptr;
  }

  if (proto.infeed_thunk()
          .infeed_resources()
          .produce_token()
          .contains_value()) {
    TF_ASSIGN_OR_RETURN(
        infeed_resources.produce_token,
        CreateResourceFromProto(
            proto.infeed_thunk().infeed_resources().produce_token().value()));
  } else {
    infeed_resources.produce_token = nullptr;
  }

  return InfeedThunk::Create(std::move(info), std::move(infeed_buffers),
                             std::move(infeed_resources));
}

absl::StatusOr<std::unique_ptr<Thunk>> KernelThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  std::vector<BufferAllocation::Slice> arguments_buffers;
  std::vector<BufferAllocation::Slice> results_buffers;

  for (const BufferAllocationSliceProto& buffer_proto :
       proto.kernel_thunk().arguments_buffers()) {
    TF_ASSIGN_OR_RETURN(auto buffer, DeserializeSliceFromProto(
                                         buffer_proto, buffer_allocations));
    arguments_buffers.push_back(std::move(buffer));
  }

  for (const BufferAllocationSliceProto& buffer_proto :
       proto.kernel_thunk().results_buffers()) {
    TF_ASSIGN_OR_RETURN(auto buffer, DeserializeSliceFromProto(
                                         buffer_proto, buffer_allocations));
    results_buffers.push_back(std::move(buffer));
  }

  se::ThreadDim thread_dim(proto.kernel_thunk().thread_dim().x(),
                           proto.kernel_thunk().thread_dim().y(),
                           proto.kernel_thunk().thread_dim().z());

  absl::flat_hash_set<int64_t> invariant_arguments;
  for (int64_t invariant_argument :
       proto.kernel_thunk().invariant_arguments()) {
    invariant_arguments.insert(invariant_argument);
  }

  std::optional<uint64_t> min_alignment = std::nullopt;
  if (proto.kernel_thunk().min_alignment().contains_value()) {
    min_alignment = proto.kernel_thunk().min_alignment().value();
  }

  return KernelThunk::Create(
      std::move(info), std::move(arguments_buffers), std::move(results_buffers),
      proto.kernel_thunk().kernel_name(), thread_dim,
      invariant_arguments.empty() ? std::nullopt
                                  : std::make_optional(invariant_arguments),
      min_alignment);
}

absl::StatusOr<std::unique_ptr<OutfeedThunk>> OutfeedThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  std::vector<OutfeedThunk::OutfeedBuffer> outfeed_buffers;
  for (const ShapeBufferAllocationSliceProto& buffer_proto :
       proto.outfeed_thunk().outfeed_buffers_shapes()) {
    TF_ASSIGN_OR_RETURN(
        auto buffer_slice_shape,
        DeserializeSliceShapeFromProto(buffer_proto, buffer_allocations));

    const auto& [buffer_slice, buffer_shape] = buffer_slice_shape;
    outfeed_buffers.push_back(
        {std::move(buffer_slice), std::move(buffer_shape)});
  }

  OutfeedThunk::OutfeedResources outfeed_resources;
  if (proto.outfeed_thunk()
          .outfeed_resources()
          .consume_token()
          .contains_value()) {
    TF_ASSIGN_OR_RETURN(
        outfeed_resources.consume_token,
        CreateResourceFromProto(
            proto.outfeed_thunk().outfeed_resources().consume_token().value()));
  } else {
    outfeed_resources.consume_token = nullptr;
  }

  if (proto.outfeed_thunk()
          .outfeed_resources()
          .produce_token()
          .contains_value()) {
    TF_ASSIGN_OR_RETURN(
        outfeed_resources.produce_token,
        CreateResourceFromProto(
            proto.outfeed_thunk().outfeed_resources().produce_token().value()));
  } else {
    outfeed_resources.produce_token = nullptr;
  }

  return OutfeedThunk::Create(std::move(info), outfeed_buffers,
                              outfeed_resources);
}

absl::StatusOr<std::unique_ptr<SortThunk>> SortThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));
  std::vector<SortThunk::Input> inputs;
  for (const ShapeBufferAllocationSliceProto& buffer_proto :
       proto.sort_thunk().inputs_shapes()) {
    TF_ASSIGN_OR_RETURN(
        auto buffer_slice_shape,
        DeserializeSliceShapeFromProto(buffer_proto, buffer_allocations));

    const auto& [buffer_slice, buffer_shape] = buffer_slice_shape;
    inputs.push_back({std::move(buffer_slice), std::move(buffer_shape)});
  }

  std::optional<SortThunk::SortDirection> sort_direction = std::nullopt;
  if (proto.sort_thunk().direction().contains_value()) {
    sort_direction =
        proto.sort_thunk().direction().value() == SortDirectionProto::ASCENDING
            ? SortThunk::SortDirection::kAscending
            : SortThunk::SortDirection::kDescending;
  }

  return SortThunk::Create(
      std::move(info), inputs, proto.sort_thunk().dimension(),
      proto.sort_thunk().is_stable(), proto.sort_thunk().comparator_name(),
      sort_direction);
}

absl::StatusOr<std::unique_ptr<WhileThunk>> WhileThunkFromProto(
    const ThunkProto& proto,
    const std::vector<BufferAllocation>& buffer_allocations) {
  ThunkSequenceSerDesProtobuf thunk_sequence_serdes(&buffer_allocations);

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<ThunkSequence> cond_sequence,
      thunk_sequence_serdes.FromProto(proto.while_thunk().cond_sequence()));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<ThunkSequence> body_sequence,
      thunk_sequence_serdes.FromProto(proto.while_thunk().body_sequence()));

  TF_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));

  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice cond_buffer,
      DeserializeSliceFromProto(proto.while_thunk().cond_buffer(),
                                buffer_allocations));

  std::optional<int64_t> trip_count = std::nullopt;
  if (proto.while_thunk().has_trip_count()) {
    trip_count = proto.while_thunk().trip_count().value();
  }
  return WhileThunk::Create(std::move(info), cond_buffer,
                            std::move(*cond_sequence),
                            std::move(*body_sequence), trip_count);
}

absl::StatusOr<std::unique_ptr<Thunk>> ThunkSerDesProtobuf::FromProto(
    const ThunkProto& proto) const {
  Thunk::Kind kind = ProtoThunkToThunkKind(proto);
  if (Thunk::KindToString(kind) != proto.kind()) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat(
            "Kind mismatch between proto kind `%s` and thunk kind `%s`.",
            proto.kind(), Thunk::KindToString(kind)));
  }

  switch (kind) {
    case Thunk::Kind::kAllGather:
      return AllGatherThunkFromProto(proto, *buffer_allocations_);
    case Thunk::Kind::kAllReduce:
      return AllReduceThunkFromProto(proto, *buffer_allocations_);
    case Thunk::Kind::kAllToAll:
      return AllToAllThunkFromProto(proto, *buffer_allocations_);
    case Thunk::Kind::kCollectivePermute:
      return CollectivePermuteThunkFromProto(proto, *buffer_allocations_);
    case Thunk::Kind::kReduceScatter:
      return ReduceScatterThunkFromProto(proto, *buffer_allocations_);
    case Thunk::Kind::kCall:
      return CallThunkFromProto(proto, *buffer_allocations_);
    case Thunk::Kind::kConditional:
      return ConditionalThunkFromProto(proto, *buffer_allocations_);
    case Thunk::Kind::kCopy:
      return CopyThunkFromProto(proto, *buffer_allocations_);
    case Thunk::Kind::kInfeed:
      return InfeedThunkFromProto(proto, *buffer_allocations_);
    case Thunk::Kind::kKernel:
      return KernelThunkFromProto(proto, *buffer_allocations_);
    case Thunk::Kind::kOutfeed:
      return OutfeedThunkFromProto(proto, *buffer_allocations_);
    case Thunk::Kind::kSort:
      return SortThunkFromProto(proto, *buffer_allocations_);
    case Thunk::Kind::kWhile:
      return WhileThunkFromProto(proto, *buffer_allocations_);
    default:
      return absl::Status(absl::StatusCode::kInvalidArgument,
                          absl::StrFormat("Unsupported thunk kind: %s",
                                          Thunk::KindToString(kind)));
  }
  return absl::UnimplementedError("FromProto is not implemented");
}

}  // namespace

ThunkSequenceSerDesProtobuf::ThunkSequenceSerDesProtobuf(
    const std::vector<BufferAllocation>* buffer_allocations)
    : buffer_allocations_(buffer_allocations) {}

absl::StatusOr<std::string> ThunkSequenceSerDesProtobuf::Serialize(
    const ThunkSequence& thunk_sequence) {
  TF_ASSIGN_OR_RETURN(ThunkSequenceProto proto, ToProto(thunk_sequence));
  return proto.SerializeAsString();
}

absl::StatusOr<std::unique_ptr<ThunkSequence>>
ThunkSequenceSerDesProtobuf::Deserialize(const std::string& serialized) {
  ThunkSequenceProto proto;
  if (!proto.ParseFromString(serialized)) {
    return absl::InternalError(absl::StrFormat(
        "Failed to parse thunk sequence proto:\n %s", serialized));
  }
  return FromProto(proto);
}

absl::StatusOr<ThunkSequenceProto> ThunkSequenceSerDesProtobuf::ToProto(
    const ThunkSequence& thunk_sequence) const {
  ThunkSerDesProtobuf thunk_serdes(buffer_allocations_);
  ThunkSequenceProto proto;
  proto.mutable_thunks()->Reserve(thunk_sequence.size());
  for (auto& thunk : thunk_sequence) {
    TF_ASSIGN_OR_RETURN(*proto.add_thunks(), thunk_serdes.ToProto(*thunk));
  }
  return proto;
}

absl::StatusOr<std::unique_ptr<ThunkSequence>>
ThunkSequenceSerDesProtobuf::FromProto(const ThunkSequenceProto& proto) const {
  ThunkSerDesProtobuf thunk_serdes(buffer_allocations_);
  auto thunk_sequence = std::make_unique<ThunkSequence>();
  for (const ThunkProto& thunk_proto : proto.thunks()) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<Thunk> thunk,
                        thunk_serdes.FromProto(thunk_proto));
    thunk_sequence->push_back(std::move(thunk));
  }
  return thunk_sequence;
}

}  // namespace zkx::cpu
