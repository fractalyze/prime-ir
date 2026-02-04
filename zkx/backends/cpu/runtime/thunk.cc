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

#include "zkx/backends/cpu/runtime/thunk.h"

#include "absl/debugging/leak_check.h"

#include "xla/tsl/profiler/lib/traceme_encode.h"
#include "zkx/backends/cpu/collectives/in_process_collectives.h"
#include "zkx/service/cpu/cpu_executable_run_options.h"

namespace zkx::cpu {

std::string_view Thunk::KindToString(Kind kind) {
  switch (kind) {
    case Kind::kAllGather:
      return "all-gather";
    case Kind::kAllReduce:
      return "all-reduce";
    case Kind::kAllToAll:
      return "all-to-all";
    case Kind::kCall:
      return "call";
    case Kind::kCollectivePermute:
      return "collective-permute";
    case Kind::kConditional:
      return "conditional";
    case Kind::kCopy:
      return "copy";
    case Kind::kInfeed:
      return "infeed";
    case Kind::kKernel:
      return "kernel";
    case Kind::kOutfeed:
      return "outfeed";
    case Kind::kReduceScatter:
      return "reduce-scatter";
    case Kind::kSort:
      return "sort";
    case Kind::kWhile:
      return "while";
    case Kind::kUnknown:
      return "unknown";
  }
}

Thunk::Thunk(Kind kind, Info info)
    : kind_(kind),
      info_(std::move(info)),
      ok_event_(OkExecuteEventSingleton()) {}

absl::StatusOr<Thunk::CollectiveExecuteParams>
Thunk::CollectiveExecuteParams::Create(
    const ExecutableRunOptions* run_options) {
  // Device ordinal must be set by caller and passed in run options, if not,
  // we use the device ordinal from the parent StreamExecutor.
  int32_t device_ordinal = -1;
  // TODO(chokobole): Uncomment this. Dependency: Stream
  // run_options->device_ordinal() >= 0
  //     ? run_options->device_ordinal()
  //     : run_options->stream()->parent()->device_ordinal();

  // Default implementation of a collectives interface that can execute
  // collective operations within the same process.
  static CpuCollectives* in_process_collectives =
      absl::IgnoreLeak(new InProcessCollectives());

  // If CPU executable run options are set, use the collectives interface
  // provided by the executable run options if it is set. Otherwise, use the
  // in-process collectives interface.
  const CpuExecutableRunOptions* cpu_run_options =
      run_options->cpu_executable_run_options();
  CpuCollectives* collectives =
      cpu_run_options && cpu_run_options->collectives()
          ? cpu_run_options->collectives()
          : in_process_collectives;

  return CollectiveExecuteParams{run_options->run_id(), device_ordinal,
                                 GlobalDeviceId(run_options->device_ordinal()),
                                 run_options->device_assignment(), collectives};
}

tsl::AsyncValueRef<Thunk::ExecuteEvent> Thunk::OkExecuteEventSingleton() {
  static tsl::AsyncValueOwningRef<ExecuteEvent>* singleton = [] {
    auto* storage =
        absl::IgnoreLeak(new tsl::internal::AsyncValueStorage<ExecuteEvent>());
    return absl::IgnoreLeak(new tsl::AsyncValueOwningRef<ExecuteEvent>(
        tsl::MakeAvailableAsyncValueRef<ExecuteEvent>(*storage)));
  }();
  return singleton->AsRef();
}

Thunk::ExecuteSession::ExecuteSession(int64_t max_workers,
                                      int64_t split_threshold)
    : lock_(std::make_shared<std::nullopt_t>(std::nullopt)),
      max_workers_(max_workers),
      split_threshold_(split_threshold) {}

// Encodes thunk info into the TraceMe compatible format.
std::string Thunk::TraceMeEncode() const {
  return tsl::profiler::TraceMeEncode(info_.op_name,
                                      {{"hlo_op", info_.op_name},
                                       {"hlo_module", info_.module_name},
                                       {"program_id", info_.module_id}});
}

ThunkSequence::ThunkSequence(std::unique_ptr<Thunk> thunk) {
  push_back(std::move(thunk));
}

void ThunkSequence::Append(ThunkSequence other) {
  reserve(size() + other.size());
  for (auto& thunk : other) {
    push_back(std::move(thunk));
  }
}

ThunkSequence::BufferUses ThunkSequence::buffer_uses() const {
  BufferUses buffer_uses;
  for (auto& thunk : *this) {
    BufferUses uses = thunk->buffer_uses();
    buffer_uses.insert(buffer_uses.end(), uses.begin(), uses.end());
  }
  return buffer_uses;
}

ThunkSequence::ResourceUses ThunkSequence::resource_uses() const {
  ResourceUses resource_uses;
  for (auto& thunk : *this) {
    ResourceUses uses = thunk->resource_uses();
    resource_uses.insert(resource_uses.end(), uses.begin(), uses.end());
  }
  return resource_uses;
}

}  // namespace zkx::cpu
