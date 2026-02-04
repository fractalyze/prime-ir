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

#include "zkx/backends/cpu/runtime/all_reduce_thunk.h"

#include <cstring>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"

#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/lib/traceme.h"
#include "zkx/backends/cpu/collectives/cpu_collectives.h"
#include "zkx/core/collectives/communicator.h"
#include "zkx/primitive_util.h"
#include "zkx/shape.h"
#include "zkx/shape_util.h"

namespace zkx::cpu {

absl::StatusOr<std::unique_ptr<AllReduceThunk>> AllReduceThunk::Create(
    Info info, ReductionKind reduction_kind, OpParams op_params,
    OpBuffers op_buffers, OpResources op_resources, bool single_replica) {
  auto datatype = op_buffers.source_shapes[0].element_type();
  if (!IsDataTypeSupportedByCollectiveReduce(datatype)) {
    return absl::UnimplementedError(
        absl::StrFormat("AllReduce for datatype '%s' is not supported",
                        primitive_util::LowercasePrimitiveTypeName(datatype)));
  }

  return absl::WrapUnique(new AllReduceThunk(
      std::move(info), reduction_kind, std::move(op_params),
      std::move(op_buffers), std::move(op_resources), single_replica));
}

tsl::AsyncValueRef<AllReduceThunk::ExecuteEvent> AllReduceThunk::Execute(
    const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  TF_ASSIGN_OR_RETURN(OpDeviceMemory data, GetOpDeviceMemory(params));

  VLOG(3) << absl::StreamFormat(
      "AllReduce: #source_buffers=%d, #destination_buffers=%d, "
      "reduction_kind=%s, single_replica=%v",
      data.source.size(), data.destination.size(),
      ReductionKindToString(reduction_kind_), single_replica_);

  for (int i = 0; i < data.source.size(); ++i) {
    VLOG(3) << absl::StreamFormat(
        "  src: %s in slice %s (%p)", source_shape(i).ToString(true),
        source_buffer(i).ToString(), data.source[i].opaque());
  }

  for (int i = 0; i < data.destination.size(); ++i) {
    VLOG(3) << absl::StreamFormat(
        "  dst: %s in slice %s (%p)", destination_shape(i).ToString(true),
        destination_buffer(i).ToString(), data.destination[i].opaque());
  }

  // Handle single-replica case by copying the source to the destination.
  if (single_replica_) {
    DCHECK_EQ(data.source.size(), data.destination.size());
    for (int i = 0; i < data.source.size(); ++i) {
      std::memcpy(data.destination[i].opaque(), data.source[i].opaque(),
                  data.destination[i].size());
    }
    return OkExecuteEvent();
  }

  return ExecuteWithCommunicator(
      params.collective_params,
      [&](const RendezvousKey& key, Communicator& comm) {
        CpuCollectives::Executor executor(key, DefaultCollectiveTimeout());
        for (int32_t i = 0; i < data.source.size(); ++i) {
          const Shape& shape = destination_shape(i);
          TF_RETURN_IF_ERROR(comm.AllReduce(
              data.source[i], data.destination[i], shape.element_type(),
              ShapeUtil::ElementsIn(shape), reduction_kind_, executor));
        }
        return absl::OkStatus();
      });

  return OkExecuteEvent();
}

}  // namespace zkx::cpu
