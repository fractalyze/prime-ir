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

#include "zkx/backends/gpu/runtime/kernel_thunk.h"

#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/STLExtras.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/service/gpu/stream_executor_util.h"
#include "zkx/stream_executor/device_memory.h"

namespace zkx::gpu {

//===----------------------------------------------------------------------===//
// KernelThunk
//===----------------------------------------------------------------------===//

KernelThunk::KernelThunk(const HloInstruction* instr, std::string kernel_name,
                         absl::Span<const KernelArgument> kernel_arguments,
                         LaunchDimensions launch_dimensions,
                         std::optional<se::ClusterDim> cluster_dim,
                         int64_t shmem_bytes,
                         std::optional<se::gpu::TmaMetadata> tma_metadata)
    : Thunk(Kind::kKernel, Thunk::ThunkInfo::WithProfileAnnotation(instr)),
      kernel_name_(std::move(kernel_name)),
      launch_dimensions_(std::move(launch_dimensions)),
      cluster_dim_(std::move(cluster_dim)),
      shmem_bytes_(shmem_bytes),
      tma_metadata_(std::move(tma_metadata)) {
  args_.reserve(kernel_arguments.size());
  written_.reserve(kernel_arguments.size());
  for (const auto& kernel_argument : kernel_arguments) {
    if (!kernel_argument.first_with_same_slice().has_value()) {
      args_.push_back(kernel_argument.slice());
      written_.push_back(kernel_argument.written());
    }
  }
}

std::string KernelThunk::ToString(int) const {
  return absl::StrFormat(
      "kernel = %s, launch dimensions = %s, cluster_dim = %s", kernel_name_,
      launch_dimensions_.ToString(),
      cluster_dim_.has_value() ? cluster_dim_->ToString() : "nullopt");
}

absl::Status KernelThunk::Initialize(const InitializeParams& params) {
  absl::MutexLock lock(&mutex_);

  // Load the kernel into the device if necessary.
  //
  // We could alternatively do this within ExecuteOnStream, but doing it here
  // lets the time spent loading the kernel not count towards our execution
  // profiles.
  auto it = kernel_cache_.find(params.executor);
  if (kernel_cache_.end() == it) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<se::Kernel> kernel,
        CreateKernel(kernel_name_, args_.size(), params.src.text,
                     params.src.binary, params.executor, shmem_bytes_));

    kernel_cache_.emplace(params.executor, std::move(kernel));
  }

  return absl::OkStatus();
}

namespace {

void PrintBufferContents(se::Stream* stream,
                         absl::Span<const se::DeviceMemoryBase> buffer_args) {
  int input_idx = 0;
  for (const se::DeviceMemoryBase& buf : buffer_args) {
    auto host_buffer = std::make_unique<char[]>(buf.size());
    CHECK_OK(stream->Memcpy(host_buffer.get(), buf, buf.size()));
    CHECK_OK(stream->BlockHostUntilDone());

    std::string buffer_contents;
    for (int i = 0; i < buf.size(); i++) {
      absl::StrAppendFormat(&buffer_contents, "%x ",
                            static_cast<unsigned>(host_buffer[i]));
    }
    VLOG(100) << "BUF(" << input_idx++ << ") = " << buffer_contents;
  }
}

}  // namespace

absl::Status KernelThunk::ExecuteOnStream(const ExecuteParams& params) {
  // Load the kernel.
  se::StreamExecutor* executor = params.stream->parent();
  LaunchDimensions launch_dimensions;
  std::optional<se::ClusterDim> cluster_dim;
  se::Kernel* kernel = nullptr;

  TF_ASSIGN_OR_RETURN(
      se::Stream * stream,
      GetStreamForExecution(Thunk::execution_stream_id(), params));

  {
    absl::MutexLock lock(&mutex_);
    auto it = kernel_cache_.find(executor);
    CHECK(it != kernel_cache_.end())
        << "Initialize() not called for StreamExecutor " << executor;
    launch_dimensions = launch_dimensions_;
    cluster_dim = cluster_dim_;
    kernel = it->second.get();
  }

  VLOG(3) << "Launching " << kernel->name();
  absl::InlinedVector<se::DeviceMemoryBase, 4> buffer_args;
  se::gpu::TmaMetadata tma_metadata =
      tma_metadata_.value_or(se::gpu::TmaMetadata{});
  for (const auto& [idx, arg] : llvm::enumerate(args_)) {
    se::DeviceMemoryBase buf = params.buffer_allocations->GetDeviceAddress(arg);
    VLOG(3) << "  Arg: alloc #" << arg.index() << ", offset: " << arg.offset()
            << ": " << buf.opaque() << " (" << buf.size() << "B)";
    auto it = tma_metadata.arg_index_to_tma_info.find(idx);
    if (it != tma_metadata.arg_index_to_tma_info.end()) {
      se::gpu::TmaDescriptor tma_desc = it->second;
      TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase tensor_map,
                          executor->CreateTensorMap(tma_desc, buf.opaque()));
      VLOG(3) << "  Using TensorMap for arg #" << arg.index() << ": "
              << tma_desc.ToString() << "; buffer: " << tensor_map.opaque()
              << " (" << tensor_map.size() << "B)";
      buffer_args.push_back(tensor_map);
    } else {
      buffer_args.push_back(buf);
    }
  }

  if (VLOG_IS_ON(100)) {
    PrintBufferContents(stream, buffer_args);
  }

  if (cluster_dim.has_value()) {
    return ExecuteKernelOnStream(*kernel, buffer_args, launch_dimensions,
                                 cluster_dim.value(), stream);
  } else {
    return ExecuteKernelOnStream(*kernel, buffer_args, launch_dimensions,
                                 stream);
  }
}

}  // namespace zkx::gpu
