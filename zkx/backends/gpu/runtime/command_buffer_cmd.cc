/* Copyright 2023 The OpenXLA Authors.
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

#include "zkx/backends/gpu/runtime/command_buffer_cmd.h"

#include <cassert>
#include <iterator>
#include <variant>

#include "absl/base/optimization.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/backends/gpu/runtime/annotation.h"
#include "zkx/debug_options_flags.h"
#include "zkx/executable_run_options.h"
#include "zkx/service/computation_placer.h"
#include "zkx/service/global_device_id.h"
#include "zkx/service/gpu/stream_executor_util.h"
#include "zkx/stream_executor/device_description.h"
#include "zkx/stream_executor/launch_dim.h"
#include "zkx/stream_executor/stream.h"
#include "zkx/stream_executor/trace_command_buffer_factory.h"

namespace zkx::gpu {

using ExecutionScopeId = se::CommandBuffer::ExecutionScopeId;
using MemoryAccess = BufferUse::MemoryAccess;

std::string CommandBufferCmdString(CommandBufferCmdType type) {
  switch (type) {
#define CASE_CMD_STRING(enum_name, cmd_name, ...) \
  case CommandBufferCmdType::enum_name:           \
    return cmd_name;
    COMMAND_BUFFER_CMD_LIST(CASE_CMD_STRING)
#undef CASE_CMD_STRING
    default:
      return "UnknownCmd";
  }
}

namespace {
// Creates command buffer builder from a cmd sequence.
se::CommandBuffer::Builder CreateBuilder(
    CommandBufferCmdSequence* commands,
    const Thunk::ExecuteParams* execute_params,
    const CommandBufferCmd::RecordParams* record_params) {
  return [=](se::CommandBuffer* command_buffer) {
    return commands->Record(*execute_params, *record_params, command_buffer,
                            CommandBufferCmdSequence::RecordMode::kConditional);
  };
}

// Creates command buffer builders from a span of cmd sequences.
std::vector<se::CommandBuffer::Builder> CreateBuilders(
    absl::Span<CommandBufferCmdSequence> commands,
    const Thunk::ExecuteParams* execute_params,
    const CommandBufferCmd::RecordParams* record_params) {
  std::vector<se::CommandBuffer::Builder> builders;
  for (CommandBufferCmdSequence& cmd : commands) {
    builders.push_back(CreateBuilder(&cmd, execute_params, record_params));
  }
  return builders;
}

// Creates command buffer execution scope builder from a cmd sequence.
se::CommandBuffer::ExecutionScopeBuilder CreateExecutionScopeBuilder(
    CommandBufferCmdSequence* commands,
    const Thunk::ExecuteParams* execute_params,
    const CommandBufferCmd::RecordParams* record_params) {
  return [=](ExecutionScopeId id, se::CommandBuffer* command_buffer) {
    CommandBufferCmd::RecordParams params = *record_params;
    params.execution_scope_id = id;
    return commands->Record(*execute_params, params, command_buffer,
                            CommandBufferCmdSequence::RecordMode::kConditional);
  };
}
}  // namespace

//===----------------------------------------------------------------------===//
// CommandBufferCmd
//===----------------------------------------------------------------------===//

CommandBufferCmd::State* CommandBufferCmd::StateManager::GetOrNull(
    const CommandBufferCmd* cmd) {
  if (auto it = state_.find(cmd); it != state_.end()) {
    return it->second.get();
  }
  return nullptr;
}

CommandBufferCmd::State* CommandBufferCmd::StateManager::GetOrCreate(
    const CommandBufferCmd* cmd,
    absl::FunctionRef<std::unique_ptr<State>()> create) {
  if (auto it = state_.find(cmd); it != state_.end()) {
    return it->second.get();
  }
  return state_.try_emplace(cmd, create()).first->second.get();
}

se::CommandBuffer::ExecutionScopeId CommandBufferCmd::GetExecutionScope(
    const RecordParams& record_params,
    ExecutionStreamId execution_stream_id) const {
  uint64_t base = record_params.execution_scope_id.value();
  uint64_t offset = execution_stream_id.value();
  return se::CommandBuffer::ExecutionScopeId(base + offset);
}

se::CommandBuffer::ExecutionScopeId CommandBufferCmd::GetExecutionScope(
    const RecordParams& record_params) const {
  return GetExecutionScope(record_params, execution_stream_id_);
}

//===----------------------------------------------------------------------===//
// CommandBufferCmdSequence
//===----------------------------------------------------------------------===//

CommandBufferCmdSequence::CommandBufferCmdSequence(
    SynchronizationMode synchronization_mode)
    : synchronization_mode_(synchronization_mode) {}

void CommandBufferCmdSequence::Append(std::unique_ptr<CommandBufferCmd> cmd) {
  for (const BufferUse& buffer : cmd->buffers()) {
    buffers_.insert(buffer);
    allocs_indices_.insert(buffer.slice().index());
  }

  ExecutionStreamId execution_stream_id = cmd->execution_stream_id();
  CommandBufferCmd::BufferUseVector buffers = cmd->buffers();
  bool requires_barrier = HasConflicts(execution_stream_id, buffers);

  // Always add barriers between commands if we want to serialize execution.
  if (synchronization_mode_ == SynchronizationMode::kSerialize &&
      !commands_.empty()) {
    requires_barrier = true;
  }

  // If the first recorded command is implemented as a nested command buffer we
  // force a barrier before recording the next command as a workaround for CUDA
  // graph bug, where child CUDA graph must be a single CUDA graph root node.
  if (commands_.size() == 1 && commands_.front().cmd->IsNestedCommandBuffer()) {
    requires_barrier = true;
  }

  if (requires_barrier) ClearTrackedBuffers(execution_stream_id);

  commands_.push_back({std::move(cmd), requires_barrier});
  TrackBuffers(execution_stream_id, buffers);
}

absl::Status CommandBufferCmdSequence::Prepare(
    const Thunk::PrepareParams& params,
    Thunk::ResourceRequestsInterface& resource_requests) {
  for (auto& command : commands_) {
    TF_RETURN_IF_ERROR(command.cmd->Prepare(params, resource_requests));
  }
  return absl::OkStatus();
}

absl::Status CommandBufferCmdSequence::Initialize(
    const Thunk::InitializeParams& params,
    CommandBufferCmd::StateManager& state) {
  int64_t index = 0;
  for (auto& command : commands_) {
    LOG(INFO) << "CommandBufferCmdSequence::Initialize cmd[" << index++
              << "]: " << command.cmd->ToString()
              << " stream=" << command.cmd->execution_stream_id().value();
    TF_RETURN_IF_ERROR(command.cmd->Initialize(params, state));
  }
  return absl::OkStatus();
}

namespace {
// Returns true if slice overlaps with any of the slices in read set.
bool Overlaps(const BufferAllocation::Slice& slice,
              const absl::flat_hash_set<BufferAllocation::Slice>& slices) {
  if (slices.contains(slice)) return true;
  for (auto& read : slices)
    if (read.OverlapsWith(slice)) return true;
  return false;
}
}  // namespace

bool CommandBufferCmdSequence::HasConflicts(
    ExecutionStreamId execution_stream_id,
    const CommandBufferCmd::BufferUseVector& buffers) {
  auto& rwset = read_write_sets_[execution_stream_id];

  return absl::c_any_of(buffers, [&](const auto& buffer) {
    return buffer.access() == MemoryAccess::kWrite
               ? Overlaps(buffer.slice(), rwset.write) ||
                     Overlaps(buffer.slice(), rwset.read)
               : Overlaps(buffer.slice(), rwset.write);
  });
}

void CommandBufferCmdSequence::TrackBuffers(
    ExecutionStreamId execution_stream_id,
    const CommandBufferCmd::BufferUseVector& buffers) {
  auto& rwset = read_write_sets_[execution_stream_id];
  for (const BufferUse& buffer : buffers) {
    if (buffer.access() == MemoryAccess::kWrite)
      rwset.write.insert(buffer.slice());
    if (buffer.access() == MemoryAccess::kRead)
      rwset.read.insert(buffer.slice());
  }
}

void CommandBufferCmdSequence::ClearTrackedBuffers(
    ExecutionStreamId execution_stream_id) {
  read_write_sets_[execution_stream_id] = ReadWriteSet();
}

namespace {

std::string_view RecordModeString(CommandBufferCmdSequence::RecordMode mode) {
  switch (mode) {
    case CommandBufferCmdSequence::RecordMode::kExclusive:
      return "exclusive";
    case CommandBufferCmdSequence::RecordMode::kConditional:
      return "conditional";
  }
}

}  // namespace

absl::Status CommandBufferCmdSequence::Record(
    const Thunk::ExecuteParams& execute_params,
    const CommandBufferCmd::RecordParams& record_params,
    se::CommandBuffer* command_buffer, RecordMode mode) {
  VLOG(3) << "Record " << commands_.size() << " commands into command buffer"
          << "; mode=" << RecordModeString(mode);
  uint64_t start_micros = tsl::Env::Default()->NowMicros();

  if (mode == RecordMode::kExclusive) {
    if (command_buffer->state() == se::CommandBuffer::State::kFinalized) {
      TF_RETURN_IF_ERROR(command_buffer->Update());
    }
  }

  // Track the number of commands recorded between barriers.
  absl::flat_hash_map<ExecutionScopeId, int64_t> num_recorded_commands;

  for (CommandInfo& command : commands_) {
    // TODO(batzor): Uncomment this. Dependency: CollectiveCmd.
    // if (execute_params.mock_collectives &&
    //     dynamic_cast<CollectiveCmd*>(command.cmd.get())) {
    //   continue;
    // }

    ExecutionScopeId execution_scope_id =
        command.cmd->GetExecutionScope(record_params);
    std::optional<tsl::profiler::ScopedAnnotation> annotation =
        GetKernelAnnotation(command.cmd->profile_annotation());

    if (command.requires_barrier) {
      VLOG(3) << "Add command buffer barrier after "
              << num_recorded_commands[execution_scope_id]
              << " recorded commands into the execution scope #"
              << execution_scope_id.value();
      TF_RETURN_IF_ERROR(command_buffer->Barrier(execution_scope_id));
      num_recorded_commands.erase(execution_scope_id);
    }
    VLOG(5) << "Record command buffer with scope id "
            << execution_scope_id.value();

    TF_RETURN_IF_ERROR(
        command.cmd->Record(execute_params, record_params, command_buffer));
    ++num_recorded_commands[execution_scope_id];
  }

  if (mode == RecordMode::kExclusive) {
    TF_RETURN_IF_ERROR(command_buffer->Finalize());
  }

  uint64_t end_micros = tsl::Env::Default()->NowMicros();
  VLOG(3) << "Recorded " << commands_.size()
          << " commands into command buffer in " << (end_micros - start_micros)
          << " μs; mode=" << RecordModeString(mode);

  return absl::OkStatus();
}

const absl::flat_hash_set<BufferUse>& CommandBufferCmdSequence::buffers()
    const {
  return buffers_;
}

const absl::flat_hash_set<BufferAllocation::Index>&
CommandBufferCmdSequence::allocs_indices() const {
  return allocs_indices_;
}

std::vector<bool> CommandBufferCmdSequence::barriers() const {
  std::vector<bool> barriers;
  absl::c_transform(commands_, std::back_inserter(barriers),
                    [](auto& command) { return command.requires_barrier; });
  return barriers;
}

//===----------------------------------------------------------------------===//
// TracedCommandBuffer
//===----------------------------------------------------------------------===//

TracedCommandBuffer::TracedCommandBuffer(
    const CommandBufferCmd* trace_cmd,
    CommandBufferCmd::BufferUseVector buffers, int64_t capacity)
    : trace_cmd_(trace_cmd), capacity_(capacity), entries_(capacity) {
  CHECK_GT(capacity, 0) << "capacity must be larger than 0";
  // Collect unique buffer allocation indices in a set first and convert to
  // vector as flat hash set iteration has measurable overheads.
  absl::flat_hash_set<BufferAllocation::Index> allocs_indices;
  for (auto& buffer : buffers) allocs_indices.insert(buffer.slice().index());
  allocs_indices_.assign(allocs_indices.begin(), allocs_indices.end());
}

absl::StatusOr<se::CommandBuffer*> TracedCommandBuffer::GetOrTraceCommandBuffer(
    const BufferAllocations* buffer_allocation, se::StreamExecutor* executor,
    se::Stream* stream, absl::FunctionRef<absl::Status(se::Stream*)> trace) {
  // Collect memory addresses for relevant allocations.
  absl::InlinedVector<se::DeviceMemoryBase, 4> allocs;
  allocs.reserve(allocs_indices_.size());
  for (auto& index : allocs_indices_) {
    allocs.emplace_back(buffer_allocation->GetDeviceAddress(index));
  }

  // Moves entry at `i` position to front and moves entries in `[0, i)` range
  // one element to the right. Returns reference to the first entry.
  auto shift_right = [&](size_t i) -> Entry& {
    if (i == 0) return entries_[0];

    Entry entry = std::move(entries_[i]);
    do {
      entries_[i] = std::move(entries_[i - 1]);
    } while (--i > 0);

    return entries_[0] = std::move(entry);
  };

  for (size_t i = 0; i < capacity_; ++i) {
    // Found entry for a given allocations, move it to front and return a
    // pointer to cached command buffer.
    if (ABSL_PREDICT_TRUE(absl::c_equal(entries_[i].recorded_allocs, allocs) &&
                          entries_[i].command_buffer)) {
      VLOG(6) << "Command buffer trace cache hit for command "
              << trace_cmd_->ToString();
      return shift_right(i).command_buffer.get();
    }

    // Create a new entry by calling a user-provided tracing function, move it
    // to front and return a pointer to cached command buffer.
    if (entries_[i].command_buffer == nullptr) {
      TF_ASSIGN_OR_RETURN(
          entries_[i].command_buffer,
          se::TraceCommandBufferFactory::Create(executor, stream, trace));
      entries_[i].recorded_allocs.assign(allocs.begin(), allocs.end());
      VLOG(6) << "Command buffer trace cache create new item for command "
              << trace_cmd_->ToString();
      return shift_right(i).command_buffer.get();
    }
  }

  // Create a new entry by calling a user-provided tracing function, replace the
  // last entry with it, move it to front and return a pointer to cached command
  // buffer.
  TF_ASSIGN_OR_RETURN(
      entries_[capacity_ - 1].command_buffer,
      se::TraceCommandBufferFactory::Create(executor, stream, trace));
  entries_[capacity_ - 1].recorded_allocs.assign(allocs.begin(), allocs.end());
  VLOG(6) << "Command buffer trace cache does replacement for command "
          << trace_cmd_->ToString();
  return shift_right(capacity_ - 1).command_buffer.get();
}

//===----------------------------------------------------------------------===//
// TracedCommandBufferCmd
//===----------------------------------------------------------------------===//

TracedCommandBufferCmd::TracedCommandBufferCmd(
    CommandBufferCmdType cmd_type, ExecutionStreamId execution_stream_id)
    : CommandBufferCmd(cmd_type, execution_stream_id) {}

absl::Status TracedCommandBufferCmd::AddTracedCommandBuffer(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer,
    absl::FunctionRef<absl::Status(se::Stream*)> trace) {
  auto traced_cmd =
      record_params.state.GetOrCreate<TracedCommandBuffer>(this, [&] {
        const auto& debug_options = zkx::GetDebugOptionsFromFlags();
        return std::make_unique<TracedCommandBuffer>(
            this, buffers(), debug_options.zkx_cmd_buffer_trace_cache_size());
      });

  TF_ASSIGN_OR_RETURN(
      auto nested_cmd,
      traced_cmd->GetOrTraceCommandBuffer(
          execute_params.buffer_allocations, execute_params.stream->parent(),
          execute_params.command_buffer_trace_stream, trace));

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "Add nested command buffer to execution scope: "
          << execution_scope_id.value();
  return command_buffer->AddNestedCommandBuffer(execution_scope_id,
                                                *nested_cmd);
}

//===----------------------------------------------------------------------===//
// ComputationId
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): PTX kernel should be replaced with CUDA C++ kernel but
// today we accidentally try to build them without CUDA support. We need to
// clean our build and testing infrastructure first.

// PTX kernel compiled from:
//
// __global__ void memset32(int64_t n, uint32_t value, uint32_t* dst)
// {
//   int i = blockIdx.x*blockDim.x + threadIdx.x;
//   if (i < n) dst[i] = value;
// }
//
// Easiest way to get PTX from C++ is to use https://godbolt.org.
inline constexpr std::string_view kMemset32Kernel = R"(
.version 4.0
.target sm_50
.address_size 64

.visible .entry memset32(
        .param .u64 memset32_param_0,
        .param .u32 memset32_param_1,
        .param .u64 memset32_param_2
)
{
        .reg .pred      %p<2>;
        .reg .b32       %r<6>;
        .reg .b64       %rd<7>;
        .loc    1 3 0

        ld.param.u64    %rd3, [memset32_param_0];
        ld.param.u32    %r1, [memset32_param_1];
        ld.param.u64    %rd2, [memset32_param_2];
        .loc    1 5 3
        mov.u32         %r2, %ctaid.x;
        mov.u32         %r3, %ntid.x;
        mov.u32         %r4, %tid.x;
        mad.lo.s32      %r5, %r2, %r3, %r4;
        .loc    1 6 3
        cvt.s64.s32     %rd1, %r5;
        setp.ge.s64     %p1, %rd1, %rd3;
        @%p1 bra        $L__BB0_2;

        .loc    1 5 3
        cvta.to.global.u64      %rd4, %rd2;
        .loc    1 6 3
        shl.b64         %rd5, %rd1, 2;
        add.s64         %rd6, %rd4, %rd5;
        st.global.u32   [%rd6], %r1;

$L__BB0_2:
        .loc    1 7 1
        ret;

})";

ComputationIdCmd::ComputationIdCmd(ExecutionStreamId execution_stream_id,
                                   BufferAllocation::Slice dest, Kind kind)
    : CommandBufferCmd(CommandBufferCmdType::kComputationIdCmd,
                       execution_stream_id),
      dest_(dest),
      kind_(kind) {}

CommandBufferCmd::BufferUseVector ComputationIdCmd::buffers() {
  return {{dest_, MemoryAccess::kWrite}};
}

absl::Status ComputationIdCmd::Initialize(const Thunk::InitializeParams& params,
                                          StateManager& state) {
  auto cuda_cc = std::get_if<stream_executor::CudaComputeCapability>(
      &params.executor->GetDeviceDescription().gpu_compute_capability());
  if (cuda_cc != nullptr) {
    {
      absl::MutexLock lock(&mutex_);
      if (memset_kernels_.contains(params.executor)) return absl::OkStatus();
    }

    TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Kernel> kernel,
                        CreateKernel("memset32", 3, kMemset32Kernel,
                                     /*cubin_data=*/{}, params.executor,
                                     /*shared_mem_bytes=*/0));

    absl::MutexLock lock(&mutex_);
    memset_kernels_.emplace(params.executor, std::move(kernel));
  }
  return absl::OkStatus();
}

absl::Status ComputationIdCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dest_);

  GlobalDeviceId global_device_id =
      execute_params.collective_params->global_device_id;
  TF_ASSIGN_OR_RETURN(
      const DeviceAssignment::LogicalID logical_id,
      execute_params.collective_params->device_assn->LogicalIdForDevice(
          global_device_id));

  uint32_t value = kind_ == Kind::kReplica ? logical_id.replica_id
                                           : logical_id.computation_id;

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "ComputationIdCmd"
          << ": kind=" << (kind_ == Kind::kReplica ? "replica" : "partition")
          << "; value=" << value
          << "; execution_scope_id=" << execution_scope_id.value();
  VLOG(5) << "  Id: " << dest_ << " (" << dst.opaque() << ")";
  auto cuda_cc = std::get_if<stream_executor::CudaComputeCapability>(
      &execute_params.stream->parent()
           ->GetDeviceDescription()
           .gpu_compute_capability());

  if (cuda_cc != nullptr) {
    se::Kernel* memset_kernel = [&] {
      absl::MutexLock lock(&mutex_);
      return memset_kernels_[execute_params.stream->parent()].get();
    }();

    if (memset_kernel == nullptr) {
      return absl::InternalError(
          "Memset kernel not loaded on a command buffer executor");
    }

    auto args = se::PackKernelArgs(/*shmem_bytes=*/0, int64_t{1}, value, dst);
    return command_buffer->Launch(execution_scope_id, se::ThreadDim(1),
                                  se::BlockDim(1), *memset_kernel, *args);
  } else {
    return command_buffer->Memset(execution_scope_id, &dst, value,
                                  /*num_elements=*/1);
  }
}

//===----------------------------------------------------------------------===//
// LaunchCmd
//===----------------------------------------------------------------------===//

LaunchCmd::LaunchCmd(ExecutionStreamId execution_stream_id,
                     std::string kernel_name,
                     absl::Span<const BufferAllocation::Slice> args,
                     absl::Span<const MemoryAccess> args_access,
                     LaunchDimensions dims, int64_t shmem_bytes)
    : CommandBufferCmd(CommandBufferCmdType::kLaunchCmd, execution_stream_id),
      kernel_name_(std::move(kernel_name)),
      args_(args.begin(), args.end()),
      args_access_(args_access.begin(), args_access.end()),
      dims_(dims),
      shmem_bytes_(shmem_bytes) {}

absl::Status LaunchCmd::Initialize(const Thunk::InitializeParams& params,
                                   StateManager& state) {
  {
    absl::MutexLock lock(&mutex_);
    if (kernels_.contains(params.executor)) return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<se::Kernel> kernel,
      CreateKernel(kernel_name_, args_.size(), params.src.text,
                   params.src.binary, params.executor, shmem_bytes_));

  absl::MutexLock lock(&mutex_);
  kernels_.emplace(params.executor, std::move(kernel));
  return absl::OkStatus();
}

absl::Status LaunchCmd::Record(const Thunk::ExecuteParams& execute_params,
                               const RecordParams& record_params,
                               se::CommandBuffer* command_buffer) {
  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "LaunchCmd: kernel=" << kernel_name_
          << "; shmem_bytes=" << shmem_bytes_
          << "; execution_scope_id=" << execution_scope_id.value();

  se::Kernel* kernel = [&] {
    absl::MutexLock lock(&mutex_);
    return kernels_[execute_params.stream->parent()].get();
  }();

  if (kernel == nullptr) {
    return absl::InternalError(absl::StrCat(
        "Kernel not loaded on a command buffer executor: ", kernel_name_));
  }

  absl::InlinedVector<se::DeviceMemoryBase, 4> buffers;
  for (const BufferAllocation::Slice& arg : args_) {
    se::DeviceMemoryBase buf =
        execute_params.buffer_allocations->GetDeviceAddress(arg);
    VLOG(5) << "  Arg: " << arg << ": " << buf.opaque();
    buffers.push_back(buf);
  }

  TF_ASSIGN_OR_RETURN(auto kernel_args,
                      se::PackKernelArgs(buffers, shmem_bytes_));

  return command_buffer->Launch(execution_scope_id,
                                dims_.thread_counts_per_block(),
                                dims_.block_counts(), *kernel, *kernel_args);
}

CommandBufferCmd::BufferUseVector LaunchCmd::buffers() {
  BufferUseVector buffers;
  for (int32_t i = 0; i < args_.size(); ++i) {
    buffers.emplace_back(args_[i], args_access_[i]);
  }
  return buffers;
}

//===----------------------------------------------------------------------===//
// MemcpyDeviceToDeviceCmd
//===----------------------------------------------------------------------===//

MemcpyDeviceToDeviceCmd::MemcpyDeviceToDeviceCmd(
    ExecutionStreamId execution_stream_id, BufferAllocation::Slice dst,
    BufferAllocation::Slice src, int64_t num_bytes)
    : CommandBufferCmd(CommandBufferCmdType::kMemcpyDeviceToDeviceCmd,
                       execution_stream_id),
      dst_(dst),
      src_(src),
      num_bytes_(num_bytes) {}

absl::Status MemcpyDeviceToDeviceCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dst_);
  se::DeviceMemoryBase src =
      execute_params.buffer_allocations->GetDeviceAddress(src_);

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "MemcpyDeviceToDeviceCmd: num_bytes = " << num_bytes_
          << "; execution_scope_id=" << execution_scope_id.value();
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";
  VLOG(5) << "  Src: " << src_ << " (" << src.opaque() << ")";

  if (num_bytes_ == 0) {
    VLOG(5) << "Skip recording MemcpyDeviceToDeviceCmd command of 0 bytes";
    return absl::OkStatus();
  }

  return command_buffer->MemcpyDeviceToDevice(execution_scope_id, &dst, src,
                                              num_bytes_);
}

CommandBufferCmd::BufferUseVector MemcpyDeviceToDeviceCmd::buffers() {
  return {{dst_, MemoryAccess::kWrite}, {src_, MemoryAccess::kRead}};
}

//===----------------------------------------------------------------------===//
// MemzeroCmd
//===----------------------------------------------------------------------===//

MemzeroCmd::MemzeroCmd(ExecutionStreamId execution_stream_id,
                       BufferAllocation::Slice dst)
    : CommandBufferCmd(CommandBufferCmdType::kMemzeroCmd, execution_stream_id),
      dst_(dst) {}

absl::Status MemzeroCmd::Record(const Thunk::ExecuteParams& execute_params,
                                const RecordParams& record_params,
                                se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dst_);

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "MemzeroCmd: execution_scope_id=" << execution_scope_id.value();
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";

  if (dst_.size() == 0) {
    VLOG(5) << "Skip recording MemzeroCmd command of 0 bytes";
    return absl::OkStatus();
  }

  return command_buffer->Memset(execution_scope_id, &dst, uint8_t{0},
                                /*num_elements=*/dst_.size());
}

CommandBufferCmd::BufferUseVector MemzeroCmd::buffers() {
  return {{dst_, MemoryAccess::kWrite}};
}

//===----------------------------------------------------------------------===//
// Memset32Cmd
//===----------------------------------------------------------------------===//

Memset32Cmd::Memset32Cmd(ExecutionStreamId execution_stream_id,
                         BufferAllocation::Slice dst, uint32_t bit_pattern)
    : CommandBufferCmd(CommandBufferCmdType::kMemset32Cmd, execution_stream_id),
      dst_(dst),
      bit_pattern_(bit_pattern) {}

absl::Status Memset32Cmd::Record(const Thunk::ExecuteParams& execute_params,
                                 const RecordParams& record_params,
                                 se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dst_);

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "Memset32Cmd: bit_pattern=" << bit_pattern_
          << "; execution_scope_id=" << execution_scope_id.value();
  VLOG(5) << "  Dst: " << dst_ << " (" << dst.opaque() << ")";

  if (dst_.size() == 0) {
    VLOG(5) << "Skip recording Memset32Cmd command of 0 bytes";
    return absl::OkStatus();
  }

  return command_buffer->Memset(
      execution_scope_id, &dst, bit_pattern_,
      /*num_elements=*/dst_.size() / sizeof(uint32_t));
}

CommandBufferCmd::BufferUseVector Memset32Cmd::buffers() {
  return {{dst_, MemoryAccess::kWrite}};
}

//===----------------------------------------------------------------------===//
// IfCmd
//===----------------------------------------------------------------------===//

IfCmd::IfCmd(ExecutionStreamId execution_stream_id,
             BufferAllocation::Slice pred,
             CommandBufferCmdSequence then_commands)
    : CommandBufferCmd(CommandBufferCmdType::kIfCmd, execution_stream_id),
      pred_(pred),
      then_commands_(std::move(then_commands)) {}

absl::Status IfCmd::Initialize(const Thunk::InitializeParams& params,
                               StateManager& state) {
  return then_commands_.Initialize(params, state);
}

absl::Status IfCmd::Record(const Thunk::ExecuteParams& execute_params,
                           const RecordParams& record_params,
                           se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase pred =
      execute_params.buffer_allocations->GetDeviceAddress(pred_);

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "IfCmd: execution_scope_id=" << execution_scope_id.value();
  VLOG(5) << "  pred: " << pred_ << " (" << pred.opaque() << ")";

  return command_buffer->If(
      execution_scope_id, se::DeviceMemory<bool>(pred),
      CreateBuilder(&then_commands_, &execute_params, &record_params));
}

bool IfCmd::force_update() { return then_commands_.force_update(); }

CommandBufferCmd::BufferUseVector IfCmd::buffers() {
  absl::flat_hash_set<BufferUse> buffers;
  buffers.emplace(pred_, MemoryAccess::kRead);
  buffers.insert(then_commands_.buffers().begin(),
                 then_commands_.buffers().end());
  return {buffers.begin(), buffers.end()};
}

//===----------------------------------------------------------------------===//
// IfElseCmd
//===----------------------------------------------------------------------===//

IfElseCmd::IfElseCmd(ExecutionStreamId execution_stream_id,
                     BufferAllocation::Slice pred,
                     CommandBufferCmdSequence then_commands,
                     CommandBufferCmdSequence else_commands)
    : CommandBufferCmd(CommandBufferCmdType::kIfElseCmd, execution_stream_id),
      pred_(pred),
      then_commands_(std::move(then_commands)),
      else_commands_(std::move(else_commands)) {}

absl::Status IfElseCmd::Initialize(const Thunk::InitializeParams& params,
                                   StateManager& state) {
  TF_RETURN_IF_ERROR(then_commands_.Initialize(params, state));
  TF_RETURN_IF_ERROR(else_commands_.Initialize(params, state));
  return absl::OkStatus();
}

absl::Status IfElseCmd::Record(const Thunk::ExecuteParams& execute_params,
                               const RecordParams& record_params,
                               se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase pred =
      execute_params.buffer_allocations->GetDeviceAddress(pred_);

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "IfElseCmd: execution_scope_id=" << execution_scope_id.value();
  VLOG(5) << "  pred: " << pred_ << " (" << pred.opaque() << ")";

  return command_buffer->IfElse(
      execution_scope_id, se::DeviceMemory<bool>(pred),
      CreateBuilder(&then_commands_, &execute_params, &record_params),
      CreateBuilder(&else_commands_, &execute_params, &record_params));
}

bool IfElseCmd::force_update() {
  return (then_commands_.force_update() || else_commands_.force_update());
}

CommandBufferCmd::BufferUseVector IfElseCmd::buffers() {
  absl::flat_hash_set<BufferUse> buffers;
  buffers.emplace(pred_, MemoryAccess::kRead);
  buffers.insert(then_commands_.buffers().begin(),
                 then_commands_.buffers().end());
  buffers.insert(else_commands_.buffers().begin(),
                 else_commands_.buffers().end());
  return {buffers.begin(), buffers.end()};
}

//===----------------------------------------------------------------------===//
// CaseCmd
//===----------------------------------------------------------------------===//

CaseCmd::CaseCmd(ExecutionStreamId execution_stream_id,
                 BufferAllocation::Slice index,
                 std::vector<CommandBufferCmdSequence> branches_commands)
    : CommandBufferCmd(CommandBufferCmdType::kCaseCmd, execution_stream_id),
      index_(index),
      branches_commands_(std::move(branches_commands)) {}

absl::Status CaseCmd::Initialize(const Thunk::InitializeParams& params,
                                 StateManager& state) {
  for (auto& branch : branches_commands_) {
    TF_RETURN_IF_ERROR(branch.Initialize(params, state));
  }
  return absl::OkStatus();
}

absl::Status CaseCmd::Record(const Thunk::ExecuteParams& execute_params,
                             const RecordParams& record_params,
                             se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase index =
      execute_params.buffer_allocations->GetDeviceAddress(index_);

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "CaseCmd: execution_scope_id=" << execution_scope_id.value();
  VLOG(5) << "  index: " << index_ << " (" << index.opaque() << ")";

  return command_buffer->Case(execution_scope_id,
                              se::DeviceMemory<int32_t>(index),
                              CreateBuilders(absl::MakeSpan(branches_commands_),
                                             &execute_params, &record_params));
}

bool CaseCmd::force_update() {
  return absl::c_any_of(branches_commands_,
                        [](const auto& seq) { return seq.force_update(); });
}

CommandBufferCmd::BufferUseVector CaseCmd::buffers() {
  absl::flat_hash_set<BufferUse> buffers;
  buffers.emplace(index_, MemoryAccess::kRead);
  for (auto& branch : branches_commands_) {
    buffers.insert(branch.buffers().begin(), branch.buffers().end());
  }
  return {buffers.begin(), buffers.end()};
}

//===----------------------------------------------------------------------===//
// ForCmd
//===----------------------------------------------------------------------===//

ForCmd::ForCmd(ExecutionStreamId execution_stream_id, int32_t num_iterations,
               BufferAllocation::Slice loop_counter,
               CommandBufferCmdSequence body_commands)
    : CommandBufferCmd(CommandBufferCmdType::kForCmd, execution_stream_id),
      num_iterations_(num_iterations),
      loop_counter_(loop_counter),
      body_commands_(std::move(body_commands)) {}

absl::Status ForCmd::Initialize(const Thunk::InitializeParams& params,
                                StateManager& state) {
  return body_commands_.Initialize(params, state);
}

absl::Status ForCmd::Record(const Thunk::ExecuteParams& execute_params,
                            const RecordParams& record_params,
                            se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase loop_counter =
      execute_params.buffer_allocations->GetDeviceAddress(loop_counter_);

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "ForCmd: num_iterations=" << num_iterations_
          << "; body_commands=" << body_commands_.size()
          << "; execution_scope_id=" << execution_scope_id.value();
  VLOG(5) << "  loop_counter: " << loop_counter_ << " ("
          << loop_counter.opaque() << ")";

  return command_buffer->For(
      execution_scope_id, num_iterations_,
      se::DeviceMemory<int32_t>(loop_counter),
      CreateBuilder(&body_commands_, &execute_params, &record_params));
}

bool ForCmd::force_update() { return body_commands_.force_update(); }

CommandBufferCmd::BufferUseVector ForCmd::buffers() {
  absl::flat_hash_set<BufferUse> buffers;
  buffers.emplace(loop_counter_, MemoryAccess::kWrite);
  buffers.insert(body_commands_.buffers().begin(),
                 body_commands_.buffers().end());
  return {buffers.begin(), buffers.end()};
}

//===----------------------------------------------------------------------===//
// WhileCmd
//===----------------------------------------------------------------------===//

WhileCmd::WhileCmd(ExecutionStreamId execution_stream_id,
                   BufferAllocation::Slice pred,
                   CommandBufferCmdSequence cond_commands,
                   CommandBufferCmdSequence body_commands)
    : CommandBufferCmd(CommandBufferCmdType::kWhileCmd, execution_stream_id),
      pred_(pred),
      cond_commands_(std::move(cond_commands)),
      body_commands_(std::move(body_commands)) {}

absl::Status WhileCmd::Initialize(const Thunk::InitializeParams& params,
                                  StateManager& state) {
  TF_RETURN_IF_ERROR(cond_commands_.Initialize(params, state));
  return body_commands_.Initialize(params, state);
}

absl::Status WhileCmd::Record(const Thunk::ExecuteParams& execute_params,
                              const RecordParams& record_params,
                              se::CommandBuffer* command_buffer) {
  se::DeviceMemoryBase pred =
      execute_params.buffer_allocations->GetDeviceAddress(pred_);

  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);
  VLOG(5) << "WhileCmd: cond_commands=" << cond_commands_.size()
          << " body_commands=" << body_commands_.size()
          << "; execution_scope_id=" << execution_scope_id.value();
  VLOG(5) << "  pred: " << pred_ << " (" << pred.opaque() << ")";

  return command_buffer->While(
      execution_scope_id, se::DeviceMemory<bool>(pred),
      CreateExecutionScopeBuilder(&cond_commands_, &execute_params,
                                  &record_params),
      CreateBuilder(&body_commands_, &execute_params, &record_params));
}

bool WhileCmd::force_update() {
  return (cond_commands_.force_update() || body_commands_.force_update());
}

CommandBufferCmd::BufferUseVector WhileCmd::buffers() {
  absl::flat_hash_set<BufferUse> buffers;
  buffers.emplace(pred_, MemoryAccess::kWrite);
  buffers.insert(cond_commands_.buffers().begin(),
                 cond_commands_.buffers().end());
  buffers.insert(body_commands_.buffers().begin(),
                 body_commands_.buffers().end());
  return {buffers.begin(), buffers.end()};
}

//===----------------------------------------------------------------------===//
// BarrierCmd
//===----------------------------------------------------------------------===//

BarrierCmd::BarrierCmd(ExecutionStreamId execution_stream_id,
                       ExecutionStreamId from_stream_id)
    : CommandBufferCmd(CommandBufferCmdType::kBarrierCmd, execution_stream_id),
      from_stream_id_(from_stream_id) {}

absl::Status BarrierCmd::Record(const Thunk::ExecuteParams& execute_params,
                                const RecordParams& record_params,
                                se::CommandBuffer* command_buffer) {
  VLOG(5) << "BarrierCmd from stream " << from_stream_id_.value()
          << " to stream " << execution_stream_id().value();
  if (from_stream_id_ != execution_stream_id()) {
    TF_RETURN_IF_ERROR(command_buffer->Barrier(
        CommandBufferCmd::GetExecutionScope(record_params, from_stream_id_),
        CommandBufferCmd::GetExecutionScope(record_params,
                                            execution_stream_id())));
  }
  return absl::OkStatus();
}

BarrierCmd::BufferUseVector BarrierCmd::buffers() { return {}; }

//===----------------------------------------------------------------------===//
// DynamicSliceFusionCmd
//===----------------------------------------------------------------------===//

DynamicSliceFusionCmd::DynamicSliceFusionCmd(
    ExecutionStreamId execution_stream_id,
    std::unique_ptr<CommandBufferCmdSequence> embedded_commands,
    std::vector<std::optional<BufferAllocation::Slice>> arguments,
    std::vector<std::unique_ptr<BufferAllocation>> fake_allocations,
    std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>> offsets,
    std::vector<std::optional<Shape>> orig_shapes,
    std::vector<std::optional<Shape>> sliced_shapes,
    std::vector<std::optional<uint64_t>> offset_byte_sizes)
    : CommandBufferCmd(CommandBufferCmdType::kDynamicSliceFusionCmd,
                       execution_stream_id),
      embedded_commands_(std::move(embedded_commands)),
      fake_allocations_(std::move(fake_allocations)) {
  // Zip all arguments together to create a list of SliceDef.
  for (auto [arg, offset, orig_shape, sliced_shape, offset_byte_size] :
       llvm::zip_equal(arguments, offsets, orig_shapes, sliced_shapes,
                       offset_byte_sizes)) {
    slices_.push_back(DynamicSliceThunk::SliceDef{
        std::move(arg),
        std::move(offset),
        std::move(orig_shape),
        std::move(sliced_shape),
        std::move(offset_byte_size),
    });
  }

  for (auto [argument_idx, slice] : llvm::enumerate(slices_)) {
    embedded_to_origin_slice_map_[argument_idx] = slice.embedded_thunk_argument;
  }

  // Find how many offsets we might have to transfer from device to host and
  // pre-compute host allocation requirements.
  for (DynamicSliceThunk::SliceDef& slice : slices_) {
    offsets_allocs_base_.push_back(offsets_allocs_size_);
    if (slice.sliced_shape.has_value()) {
      offsets_allocs_size_ += slice.sliced_shape->rank() * sizeof(int64_t);
    }
  }
}

// Force update the command when there is any non-constant value slice offset,
// because the memory address might changed if the offset is loop
// iterator or operator outputs even if the parent command's memory pointers
// do not change.
bool DynamicSliceFusionCmd::force_update() {
  return !absl::c_all_of(slices_, [](const DynamicSliceThunk::SliceDef& slice) {
    if (!slice.offsets.has_value()) return true;
    return absl::c_all_of(slice.offsets.value(),
                          [](DynamicSliceThunk::Offset offset) {
                            return std::holds_alternative<int64_t>(offset);
                          });
  });
}

absl::Status DynamicSliceFusionCmd::Initialize(
    const Thunk::InitializeParams& params, StateManager& state) {
  TF_RETURN_IF_ERROR(embedded_commands_->Initialize(params, state));
  absl::MutexLock lock(&mutex_);
  if (offsets_allocs_.contains(params.executor)) return absl::OkStatus();

  VLOG(2) << "Allocate " << offsets_allocs_size_
          << " bytes for transferring offsets on executor: " << params.executor;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<se::MemoryAllocation> allocation,
      params.executor->HostMemoryAllocate(offsets_allocs_size_));
  offsets_allocs_.emplace(params.executor, std::move(allocation));
  return absl::OkStatus();
}

absl::Status DynamicSliceFusionCmd::Prepare(
    const Thunk::PrepareParams& params,
    Thunk::ResourceRequestsInterface& resource_requests) {
  for (DynamicSliceThunk::SliceDef& slice : slices_) {
    if (slice.offsets.has_value()) {
      TF_RET_CHECK(slice.embedded_thunk_argument.has_value());
      TF_RET_CHECK(slice.orig_shape.has_value());
      TF_RET_CHECK(slice.sliced_shape.has_value());
      TF_RET_CHECK(slice.offset_byte_size.has_value());

      TF_RET_CHECK(slice.orig_shape->IsArray());
      TF_RET_CHECK(slice.sliced_shape->IsArray());

      TF_RET_CHECK(slice.offsets->size() == slice.orig_shape->rank());
      TF_RET_CHECK(slice.sliced_shape->rank() == slice.orig_shape->rank());
    }
  }
  TF_RETURN_IF_ERROR(embedded_commands_->Prepare(params, resource_requests));
  return absl::OkStatus();
}

absl::Status DynamicSliceFusionCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, se::CommandBuffer* command_buffer) {
  se::Stream& stream = *execute_params.stream;
  ExecutionScopeId execution_scope_id = GetExecutionScope(record_params);

  const BufferAllocations& orig_allocations =
      *execute_params.buffer_allocations;
  absl::InlinedVector<se::DeviceMemoryBase, 8> slice_buffers(
      slices_.size(), se::DeviceMemoryBase());

  // Get memory allocation for copying offsets from device.
  int64_t* offsets_alloc = [&] {
    absl::MutexLock lock(&mutex_);
    return reinterpret_cast<int64_t*>(
        offsets_allocs_.at(stream.parent())->opaque());
  }();

  auto offset_value = [&](int64_t arg_idx, int64_t offset_idx) -> int64_t& {
    return offsets_alloc[offsets_allocs_base_.at(arg_idx) + offset_idx];
  };

  VLOG(2) << "Execute address computation thunk: slices=" << slices_.size();
  for (auto [argument_idx, slice] : llvm::enumerate(slices_)) {
    // Skip arguments that do not have buffer slices (tokens).
    if (!slice.embedded_thunk_argument.has_value()) {
      continue;
    }

    // `argument_buffer` will contain the original offset for slice
    // `argument_slice` within `orig_allocations`
    se::DeviceMemoryBase argument_buffer =
        orig_allocations.GetDeviceAddress(*slice.embedded_thunk_argument);

    // If argument is not sliced, just use the original buffer.
    if (!slice.offsets.has_value()) {
      slice_buffers[argument_idx] = argument_buffer;
      continue;
    }

    const Shape& src_shape = *slice.orig_shape;
    const Shape& dst_shape = *slice.sliced_shape;

    absl::InlinedVector<int64_t, 4> slice_starts;
    slice_starts.reserve(dst_shape.rank());

    // Number of issues d2h transfers to copy offset values from device to
    // host.
    int64_t num_transfers = 0;

    // Get offset for `argument_idx`-th argument, which has `dst_shape.rank()`
    // components.
    for (auto [offset_idx, values] : llvm::enumerate(llvm::zip(
             *slice.offsets, src_shape.dimensions(), dst_shape.dimensions()))) {
      auto [offset, src_dim, dst_dim] = values;
      if (int64_t* const_offset = std::get_if<int64_t>(&offset)) {
        // Forward slice offsets that are known constant values
        VLOG(2) << "  - arg " << argument_idx << "[" << offset_idx
                << "]: constant offset = " << *const_offset;
        offset_value(argument_idx, offset_idx) = *const_offset;

      } else {
        // Transfer slice offset value from device to host.
        auto alloc_slice = std::get<BufferAllocation::Slice>(offset);
        VLOG(2) << "  - arg " << argument_idx << "[" << offset_idx
                << "]: transfer offset from device " << alloc_slice.ToString();

        se::DeviceMemoryBase offset_src =
            orig_allocations.GetDeviceAddress(alloc_slice);
        int64_t* offset_dst = &offset_value(argument_idx, offset_idx);

        // Copy the `offset_idx`-th component of the offset for the
        // `argument_idx`-th argument from device to host.
        TF_RETURN_IF_ERROR(
            stream.Memcpy(offset_dst, offset_src, *slice.offset_byte_size));
        ++num_transfers;
      }
    }

    // Wait for the completion of all transfers.
    if (num_transfers > 0) {
      VLOG(2) << "Wait for completion of " << num_transfers << " transfer";
      TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());
    }

    // Clamp start indices:
    // start_indices[i] = min(max(start_indices[i], 0),
    //                        operand.dimension_size[i] - size_indices[i])
    for (auto [offset_idx, values] : llvm::enumerate(
             llvm::zip(src_shape.dimensions(), dst_shape.dimensions()))) {
      auto [src_dim, dst_dim] = values;
      int64_t start_index =
          std::min(std::max(offset_value(argument_idx, offset_idx), int64_t{0}),
                   src_dim - dst_dim);
      VLOG(2) << "arg idx: " << argument_idx << " offset_idx " << offset_idx
              << " with offset_value " << offset_value(argument_idx, offset_idx)
              << " start_idx: " << start_index << " src_dim: " << src_dim
              << " dst_dim:" << dst_dim;
      slice_starts.push_back(start_index);
    }

    // Compute new slice. No need to copy the content to new buffers as we can
    // reuse the original buffers since slices are contiguous.
    int64_t new_size = ShapeUtil::ByteSizeOf(dst_shape);

    int64_t new_offset = 0;
    for (auto [start, stride] :
         llvm::zip(slice_starts, *ShapeUtil::ByteStrides(src_shape))) {
      new_offset += start * stride;
    }

    VLOG(2) << "Create sliced argument " << argument_idx << " of shape "
            << slice.sliced_shape->ToString()
            << " by slicing argument of shape " << slice.orig_shape->ToString()
            << " at offset " << new_offset << " with " << new_size;
    slice_buffers[argument_idx] =
        argument_buffer.GetByteSlice(new_offset, new_size);
  }

  // Safe to create a local BufferAllocations here since buffers are only
  // slices of bigger ones allocated elsewhere.
  BufferAllocations slice_allocations(slice_buffers,
                                      orig_allocations.device_ordinal(),
                                      orig_allocations.memory_allocator());

  Thunk::ExecuteParams new_params =
      Thunk::ExecuteParams::CloneWithNewAllocations(execute_params,
                                                    slice_allocations);
  auto nested_command_buffer =
      execute_params.stream->parent()
          ->CreateCommandBuffer(se::CommandBuffer::Mode::kNested)
          .value();
  TF_RETURN_IF_ERROR(embedded_commands_->Record(new_params, record_params,
                                                nested_command_buffer.get()));
  return command_buffer->AddNestedCommandBuffer(execution_scope_id,
                                                *nested_command_buffer);
}

CommandBufferCmd::BufferUseVector DynamicSliceFusionCmd::buffers() {
  CommandBufferCmd::BufferUseVector buffers;
  auto embed_buffers = embedded_commands_->buffers();
  for (auto buffer_usage : embed_buffers) {
    CHECK(embedded_to_origin_slice_map_[buffer_usage.slice().index()]
              .has_value());
    buffers.emplace_back(
        embedded_to_origin_slice_map_[buffer_usage.slice().index()].value(),
        buffer_usage.access());
  }
  return buffers;
}

}  // namespace zkx::gpu
