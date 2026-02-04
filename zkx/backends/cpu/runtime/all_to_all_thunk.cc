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

#include "zkx/backends/cpu/runtime/all_to_all_thunk.h"

#include "absl/log/log.h"
#include "absl/strings/str_format.h"

#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/lib/traceme.h"
#include "zkx/backends/cpu/collectives/cpu_collectives.h"
#include "zkx/core/collectives/communicator.h"
#include "zkx/service/collective_ops_utils.h"
#include "zkx/shape.h"
#include "zkx/shape_util.h"

namespace zkx::cpu {

tsl::AsyncValueRef<AllToAllThunk::ExecuteEvent> AllToAllThunk::Execute(
    const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  TF_ASSIGN_OR_RETURN(OpDeviceMemory data, GetOpDeviceMemory(params));

  VLOG(3) << absl::StreamFormat(
      "AllToAll: #source_buffers=%d, #destination_buffers=%d",
      data.source.size(), data.destination.size());

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

  return ExecuteWithCommunicator(
      params.collective_params,
      [&](const RendezvousKey& key, Communicator& comm) {
        CpuCollectives::Executor executor(key, DefaultCollectiveTimeout());
        const Shape& shape = destination_shape(0);

        TF_RETURN_IF_ERROR(
            comm.AllToAll(data.source, data.destination, shape.element_type(),
                          ShapeUtil::ElementsIn(shape), executor));

        return absl::OkStatus();
      });
}

}  // namespace zkx::cpu
