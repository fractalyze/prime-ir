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

#include "zkx/backends/gpu/runtime/command_buffer_thunk.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/lib/profiler_lock.h"
#include "zkx/backends/gpu/runtime/command_buffer_cmd.h"
#include "zkx/backends/gpu/runtime/memset_thunk.h"
#include "zkx/backends/gpu/runtime/sequential_thunk.h"
#include "zkx/backends/gpu/runtime/thunk.h"
#include "zkx/runtime/buffer_use.h"
#include "zkx/service/buffer_assignment.h"
#include "zkx/service/gpu/buffer_allocations.h"
#include "zkx/service/gpu/launch_dimensions.h"
#include "zkx/service/platform_util.h"
#include "zkx/service/service_executable_run_options.h"
#include "zkx/stream_executor/command_buffer.h"
#include "zkx/stream_executor/device_description.h"
#include "zkx/stream_executor/device_memory.h"
#include "zkx/stream_executor/device_memory_allocator.h"
#include "zkx/stream_executor/gpu/gpu_test_kernels_fatbin.h"
#include "zkx/stream_executor/kernel_spec.h"
#include "zkx/stream_executor/platform.h"
#include "zkx/stream_executor/platform_manager.h"
#include "zkx/stream_executor/semantic_version.h"
#include "zkx/stream_executor/stream_executor.h"
#include "zkx/stream_executor/stream_executor_memory_allocator.h"

namespace zkx::gpu {

using MemoryAccess = BufferUse::MemoryAccess;
using KernelArgsPacking = se::MultiKernelLoaderSpec::KernelArgsPacking;

namespace {
se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

struct OwningExecutableSource {
  std::string text;
  std::vector<uint8_t> binary;

  explicit operator Thunk::ExecutableSource() const { return {text, binary}; }
};

absl::StatusOr<OwningExecutableSource> ExecutableSource() {
  TF_ASSIGN_OR_RETURN(std::vector<uint8_t> fatbin,
                      se::gpu::GetGpuTestKernelsFatbin());
  return OwningExecutableSource{/*text=*/{},
                                /*binary=*/fatbin};
}

// Some of the tests rely on CUDA 12.3+ features.
bool IsAtLeastCuda12300(const se::StreamExecutor* executor) {
  const auto& device_description = executor->GetDeviceDescription();
  const auto* cuda_cc = std::get_if<se::CudaComputeCapability>(
      &device_description.gpu_compute_capability());
  if (cuda_cc != nullptr) {
    if (device_description.driver_version() >=
        stream_executor::SemanticVersion(12, 3, 0)) {
      return true;
    }
  }

  return false;
}

// Give a short aliases to execution threads.
constexpr auto s0 = ExecutionStreamId(0);
constexpr auto s1 = ExecutionStreamId(1);
}  // namespace

TEST(CommandBufferThunkTest, MemcpyCmd) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42, b=0
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);

  ASSERT_TRUE(stream->Memset32(&a, 42, byte_length).ok());
  ASSERT_TRUE(stream->MemZero(&b, byte_length).ok());

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  // Prepare commands sequence for constructing command buffer.
  CommandBufferCmdSequence commands;
  commands.Emplace<MemcpyDeviceToDeviceCmd>(s0, slice_b, slice_a, byte_length);

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo());

  se::StreamExecutorMemoryAllocator allocator(executor);
  ServiceExecutableRunOptions run_options;
  BufferAllocations allocations({a, b}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  // Execute command buffer thunk and verify that it copied the memory.
  ASSERT_TRUE(thunk.ExecuteOnStream(params).ok());
  ASSERT_TRUE(stream->BlockHostUntilDone().ok());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42));

  // Try to update the command buffer with the same buffers.
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));

  // Thunk execution should automatically update underlying command buffer.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42));
}

TEST(CommandBufferThunkTest, MemzeroCmd) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);

  // Prepare commands sequence for constructing command buffer.
  CommandBufferCmdSequence commands;
  commands.Emplace<MemzeroCmd>(s0, slice_a);

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({a}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  // Execute command buffer thunk and verify that it zeroes the memory.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `a` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 0));
}

TEST(CommandBufferThunkTest, Memset32Cmd) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);

  // Prepare commands sequence for constructing command buffer.
  CommandBufferCmdSequence commands;
  commands.Emplace<Memset32Cmd>(s0, slice_a, int32_t{84});

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations(
      {a}, 0, static_cast<se::DeviceMemoryAllocator*>(&allocator));

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  // Execute command buffer thunk and verify that it set the memory.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `a` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 84));
}

// Verifies that when profiler is active and command buffers are disabled during
// profiling, the thunk falls back to sequential execution.
TEST(CommandBufferThunkTest, Memset32CmdCommandBuffersDisabledDuringProfiling) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);

  auto memset_thunk =
      std::make_unique<Memset32BitValueThunk>(Thunk::ThunkInfo(), 84, slice_a);
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(memset_thunk));
  auto seq_thunks =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));
  // Prepare commands sequence for constructing command buffer that should not
  // be used.
  CommandBufferCmdSequence commands;
  commands.Emplace<Memset32Cmd>(s0, slice_a, int32_t{12});

  constexpr bool kProfileCommandBuffersEnabled = false;
  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo(),
                           std::move(seq_thunks),
                           kProfileCommandBuffersEnabled);

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({a}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  // Acquire profiler lock to simulate active profiling session.
  TF_ASSERT_OK_AND_ASSIGN(auto profiler_lock,
                          tsl::profiler::ProfilerLock::Acquire());
  // Execute command buffer thunk and verify that it set the memory.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `a` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 84));
}

// Verifies that when profiler is active but command buffers are enabled during
// profiling, the thunk still uses command buffer execution.
TEST(CommandBufferThunkTest, Memset32CmdCommandBuffersEnabledDuringProfiling) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);

  auto memset_thunk =
      std::make_unique<Memset32BitValueThunk>(Thunk::ThunkInfo(), 84, slice_a);
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(memset_thunk));
  auto seq_thunks =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));
  // Prepare commands sequence for constructing command buffer that should not
  // be used.
  CommandBufferCmdSequence commands;
  commands.Emplace<Memset32Cmd>(s0, slice_a, int32_t{12});

  constexpr bool kProfileCommandBuffersEnabled = true;
  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo(),
                           std::move(seq_thunks),
                           kProfileCommandBuffersEnabled);

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({a}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  // Acquire profiler lock to simulate active profiling session.
  TF_ASSERT_OK_AND_ASSIGN(auto profiler_lock,
                          tsl::profiler::ProfilerLock::Acquire());
  // Execute command buffer thunk and verify that it set the memory.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `a` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 12));
}

TEST(CommandBufferThunkTest, Memset32CmdOnDifferentStreams) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(2, 0);
  TF_ASSERT_OK(stream->MemZero(&a, 2 * sizeof(int32_t)));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc(/*index=*/0, a.size(), /*color=*/0);
  BufferAllocation::Slice slice0(&alloc, 0 * sizeof(int32_t), sizeof(int32_t));
  BufferAllocation::Slice slice1(&alloc, 1 * sizeof(int32_t), sizeof(int32_t));

  // Prepare commands sequence for constructing command buffer.
  CommandBufferCmdSequence commands;
  commands.Emplace<Memset32Cmd>(s0, slice0, int32_t{12});
  commands.Emplace<Memset32Cmd>(s1, slice1, int32_t{34});

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({a}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  // Execute command buffer thunk and verify that it set the memory.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `a` data back to host.
  std::vector<int32_t> dst(2, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), a, a.size()));

  ASSERT_EQ(dst, std::vector<int32_t>({12, 34}));
}

TEST(CommandBufferThunkTest, LaunchCmd) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42, b=0
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  auto args = {slice_a, slice_a, slice_b};  // b = a + a
  auto args_access = {MemoryAccess::kRead, MemoryAccess::kRead,
                      MemoryAccess::kWrite};

  // Prepare commands sequence for constructing command buffer.
  CommandBufferCmdSequence commands;
  commands.Emplace<LaunchCmd>(s0, "AddI32", args, args_access,
                              LaunchDimensions(1, 4),
                              /*shmem_bytes=*/0);

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({a, b}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(OwningExecutableSource source, ExecutableSource());
  TF_ASSERT_OK(
      thunk.Initialize({executor, static_cast<Thunk::ExecutableSource>(source),
                        &allocations, stream.get()}));

  // Execute command buffer thunk and verify that it added the value.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Prepare buffer allocation for updating command buffer: c=0
  se::DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);
  TF_ASSERT_OK(stream->MemZero(&c, byte_length));

  // Update buffer allocation #1 to buffer `c`.
  allocations = BufferAllocations({a, c}, 0, &allocator);

  // Thunk execution should automatically update underlying command buffer.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `c` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Try to update the command buffer with the same buffers.
  TF_ASSERT_OK(stream->MemZero(&c, byte_length));

  // Thunk execution should automatically update underlying command buffer.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `c` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));
}

// TODO(batzor): Implement this. Dependency: CustomKernel
// TEST(CommandBufferThunkTest, CustomAddKernelLaunchCmd) {
// }

// TODO(batzor): Uncomment this. Dependency: DynamicSliceFusion
TEST(CommandBufferThunkTest, DynamicSliceFusionCmd) {
  se::StreamExecutor* executor = GpuExecutor();

  if (!IsAtLeastCuda12300(executor)) {
    GTEST_SKIP() << "CUDA graph tracing is not supported";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t lhs_length = sizeof(int32_t) * 8;
  int64_t fake_lhs_length = sizeof(int32_t) * 4;
  int64_t rhs_length = sizeof(int32_t) * 4;
  int64_t out_length = sizeof(int32_t) * 4;

  // Prepare arguments:
  // lhs = [0, 0, 0, 0, 1, 2, 3, 4]
  // rhs = [10, 10, 10, 10]
  se::DeviceMemory<int32_t> lhs = executor->AllocateArray<int32_t>(8);
  std::vector<int32_t> lhs_arr{0, 0, 0, 0, 1, 2, 3, 4};
  TF_ASSERT_OK(stream->Memcpy(&lhs, lhs_arr.data(), lhs_length));

  se::DeviceMemory<int32_t> rhs = executor->AllocateArray<int32_t>(4);
  std::vector<int32_t> rhs_arr(4, 10);
  TF_ASSERT_OK(stream->Memcpy(&rhs, rhs_arr.data(), rhs_length));

  se::DeviceMemory<int32_t> out = executor->AllocateArray<int32_t>(4);
  TF_ASSERT_OK(stream->MemZero(&out, out_length));

  // Prepare buffer allocations for recording command buffer.
  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations(3);
  fake_allocations[0] = std::make_unique<BufferAllocation>(
      /*index=*/0, fake_lhs_length, /*color=*/0);
  fake_allocations[1] =
      std::make_unique<BufferAllocation>(/*index=*/1, rhs_length, /*color=*/0);
  fake_allocations[2] =
      std::make_unique<BufferAllocation>(/*index=*/2, out_length,
                                         /*color=*/0);
  BufferAllocation::Slice fake_slice_lhs(fake_allocations[0].get(), 0,
                                         fake_lhs_length);
  BufferAllocation::Slice fake_slice_rhs(fake_allocations[1].get(), 0,
                                         rhs_length);
  BufferAllocation::Slice fake_slice_out(fake_allocations[2].get(), 0,
                                         out_length);

  // Prepare commands sequence for constructing command buffer.
  std::unique_ptr<CommandBufferCmdSequence> embed_commands =
      std::make_unique<CommandBufferCmdSequence>();
  auto args = {fake_slice_lhs, fake_slice_rhs, fake_slice_out};
  auto args_access = {MemoryAccess::kRead, MemoryAccess::kRead,
                      MemoryAccess::kWrite};
  embed_commands->Emplace<LaunchCmd>(s0, "AddI32", args, args_access,
                                     LaunchDimensions(1, 4),
                                     /*shmem_bytes=*/0);

  BufferAllocation alloc_lhs(/*index=*/0, lhs_length, /*color=*/0);
  BufferAllocation alloc_rhs(/*index=*/1, rhs_length, /*color=*/0);
  BufferAllocation alloc_out(/*index=*/2, out_length, /*color=*/0);
  BufferAllocation::Slice slice_lhs(&alloc_lhs, 0, lhs_length);
  BufferAllocation::Slice slice_rhs(&alloc_rhs, 0, rhs_length);
  BufferAllocation::Slice slice_out(&alloc_out, 0, out_length);

  std::vector<DynamicSliceThunk::Offset> lhs_offsets = {
      DynamicSliceThunk::Offset(4l)};

  std::vector<std::optional<BufferAllocation::Slice>> arguments = {
      std::optional<BufferAllocation::Slice>(slice_lhs),
      std::optional<BufferAllocation::Slice>(slice_rhs),
      std::optional<BufferAllocation::Slice>(slice_out)};

  std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>> offsets = {
      lhs_offsets, std::nullopt, std::nullopt};

  std::vector<std::optional<Shape>> orig_shapes = {
      ShapeUtil::MakeShape(PrimitiveType::S32, {8}), std::nullopt,
      std::nullopt};
  std::vector<std::optional<Shape>> sliced_shapes = {
      ShapeUtil::MakeShape(PrimitiveType::S32, {4}), std::nullopt,
      std::nullopt};
  std::vector<std::optional<uint64_t>> offset_byte_sizes = {
      sizeof(int64_t), std::nullopt, std::nullopt};

  CommandBufferCmdSequence commands;
  commands.Emplace<DynamicSliceFusionCmd>(
      s0, std::move(embed_commands), arguments, std::move(fake_allocations),
      offsets, orig_shapes, sliced_shapes, offset_byte_sizes);

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({lhs, rhs, out}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(OwningExecutableSource source, ExecutableSource());
  TF_ASSERT_OK(
      thunk.Initialize({executor, static_cast<Thunk::ExecutableSource>(source),
                        &allocations, stream.get()}));

  // Execute command buffer thunk and verify that it added the value.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `out` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), out, out_length));

  ASSERT_EQ(dst, std::vector<int32_t>({11, 12, 13, 14}));

  // Prepare buffer allocation for updating command buffer.
  se::DeviceMemory<int32_t> updated_out = executor->AllocateArray<int32_t>(4);
  TF_ASSERT_OK(stream->MemZero(&updated_out, out_length));

  // Update buffer allocation to updated `out` buffer.
  allocations = BufferAllocations({lhs, rhs, updated_out}, 0, &allocator);

  // Thunk execution should automatically update underlying command buffer.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `updated_out` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), updated_out, out_length));

  ASSERT_EQ(dst, std::vector<int32_t>({11, 12, 13, 14}));

  // Try to update the command buffer with the same buffers.
  TF_ASSERT_OK(stream->MemZero(&updated_out, out_length));

  // Thunk execution should automatically update underlying command buffer.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `updated_out` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), updated_out, out_length));

  ASSERT_EQ(dst, std::vector<int32_t>({11, 12, 13, 14}));
}

TEST(CommandBufferThunkTest, MultipleLaunchCmd) {
  se::StreamExecutor* executor = GpuExecutor();

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=42, b=0
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> d = executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));
  TF_ASSERT_OK(stream->Memset32(&c, 21, byte_length));
  TF_ASSERT_OK(stream->MemZero(&d, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_a(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/1, byte_length, /*color=*/0);
  BufferAllocation alloc_c(/*index=*/2, byte_length, /*color=*/0);
  BufferAllocation alloc_d(/*index=*/3, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);
  BufferAllocation::Slice slice_c(&alloc_c, 0, byte_length);
  BufferAllocation::Slice slice_d(&alloc_d, 0, byte_length);

  auto args = {slice_a, slice_a, slice_b};    // b = a + a
  auto args_1 = {slice_c, slice_c, slice_d};  // d = c + c
  auto args_access = {MemoryAccess::kRead, MemoryAccess::kRead,
                      MemoryAccess::kWrite};

  // Prepare commands sequence for constructing command buffer.
  CommandBufferCmdSequence commands;
  commands.Emplace<LaunchCmd>(s0, "AddI32", args, args_access,
                              LaunchDimensions(1, 4),
                              /*shmem_bytes=*/0);
  commands.Emplace<LaunchCmd>(s0, "AddI32", args_1, args_access,
                              LaunchDimensions(1, 4),
                              /*shmem_bytes=*/0);

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({a, b, c, d}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(OwningExecutableSource source, ExecutableSource());
  TF_ASSERT_OK(
      thunk.Initialize({executor, static_cast<Thunk::ExecutableSource>(source),
                        &allocations, stream.get()}));

  // Execute command buffer thunk and verify that it added the value.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Copy `d` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), d, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 21 + 21));

  BufferAllocation alloc_e(/*index=*/3, byte_length, /*color=*/0);
  BufferAllocation::Slice slice_e(&alloc_e, 0, byte_length);

  // Prepare buffer allocation for updating command buffer: e=0
  se::DeviceMemory<int32_t> e = executor->AllocateArray<int32_t>(length, 0);
  TF_ASSERT_OK(stream->MemZero(&e, byte_length));

  // Update buffer allocation #1 to buffer `c`.
  allocations = BufferAllocations({a, b, c, e}, 0, &allocator);

  // Thunk execution should automatically update underlying command buffer.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Copy `e` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), e, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 21 + 21));

  // Try to update the command buffer with the same buffers.
  TF_ASSERT_OK(stream->MemZero(&e, byte_length));

  // Thunk execution should automatically update underlying command buffer.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Copy `e` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), e, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 21 + 21));
}

TEST(CommandBufferThunkTest, IfCmd) {
  se::StreamExecutor* executor = GpuExecutor();

  if (!IsAtLeastCuda12300(executor)) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: pred=true, a=42, b=0
  se::DeviceMemory<bool> pred = executor->AllocateArray<bool>(1, 0);
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);

  constexpr bool kTrue = true;
  TF_ASSERT_OK(stream->Memcpy(&pred, &kTrue, 1));
  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_p(/*index=*/0, 1, /*color=*/0);
  BufferAllocation alloc_a(/*index=*/1, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/2, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_p(&alloc_p, 0, 1);
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  auto args = {slice_a, slice_a, slice_b};  // b = a + a
  auto args_access = {MemoryAccess::kRead, MemoryAccess::kRead,
                      MemoryAccess::kWrite};

  // Prepare commands sequence for `then` branch.
  CommandBufferCmdSequence then_commands;
  then_commands.Emplace<LaunchCmd>(s0, "AddI32", args, args_access,
                                   LaunchDimensions(1, 4),
                                   /*shmem_bytes=*/0);

  // Prepare commands sequence for thunk.
  CommandBufferCmdSequence commands;
  commands.Emplace<IfCmd>(s0, slice_p, std::move(then_commands));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({pred, a, b}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(OwningExecutableSource source, ExecutableSource());
  TF_ASSERT_OK(
      thunk.Initialize({executor, static_cast<Thunk::ExecutableSource>(source),
                        &allocations, stream.get()}));

  // Execute command buffer thunk and verify that it added the value.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Prepare buffer allocation for updating command buffer: c=0
  se::DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);
  TF_ASSERT_OK(stream->MemZero(&c, byte_length));

  // Update buffer allocation #2 to buffer `c`.
  allocations = BufferAllocations({pred, a, c}, 0, &allocator);

  // Thunk execution should automatically update underlying command buffer.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `c` data back to host.
  std::fill(dst.begin(), dst.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), c, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));
}

TEST(CommandBufferThunkTest, IfElseCmd) {
  se::StreamExecutor* executor = GpuExecutor();

  if (!IsAtLeastCuda12300(executor)) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: pred=true, a=42, b=0
  se::DeviceMemory<bool> pred = executor->AllocateArray<bool>(1, 0);
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);

  constexpr bool kTrue = true;
  TF_ASSERT_OK(stream->Memcpy(&pred, &kTrue, 1));
  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_p(/*index=*/0, 1, /*color=*/0);
  BufferAllocation alloc_a(/*index=*/1, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/2, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_p(&alloc_p, 0, 1);
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  // Prepare commands sequence for `then` & `else` branches.
  CommandBufferCmdSequence then_commands;
  CommandBufferCmdSequence else_commands;

  auto args_access = {MemoryAccess::kRead, MemoryAccess::kRead,
                      MemoryAccess::kWrite};

  {  // Then: b = a + a
    auto args = {slice_a, slice_a, slice_b};
    then_commands.Emplace<LaunchCmd>(s0, "AddI32", args, args_access,
                                     LaunchDimensions(1, 4),
                                     /*shmem_bytes=*/0);
  }

  {  // Else: b = b + b
    auto args = {slice_b, slice_b, slice_b};
    else_commands.Emplace<LaunchCmd>(s0, "AddI32", args, args_access,
                                     LaunchDimensions(1, 4),
                                     /*shmem_bytes=*/0);
  }

  // Prepare commands sequence for thunk.
  CommandBufferCmdSequence commands;
  commands.Emplace<IfElseCmd>(s0, slice_p, std::move(then_commands),
                              std::move(else_commands));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({pred, a, b}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(OwningExecutableSource source, ExecutableSource());
  TF_ASSERT_OK(
      thunk.Initialize({executor, static_cast<Thunk::ExecutableSource>(source),
                        &allocations, stream.get()}));

  // Execute command buffer thunk and verify that it added the value.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Change branch to `else` and check that it updated the `b` buffer.
  constexpr bool kFalse = false;
  TF_ASSERT_OK(stream->Memcpy(&pred, &kFalse, 1));

  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 2 * (42 + 42)));
}

TEST(CommandBufferThunkTest, CaseCmd) {
  se::StreamExecutor* executor = GpuExecutor();

  if (!IsAtLeastCuda12300(executor)) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: index=0, a=42, b=0
  se::DeviceMemory<int32_t> index = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&index, 0, sizeof(int32_t)));
  TF_ASSERT_OK(stream->Memset32(&a, 42, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_i(/*index=*/0, 1, /*color=*/0);
  BufferAllocation alloc_a(/*index=*/1, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/2, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_i(&alloc_i, 0, sizeof(int32_t));
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  // Prepare commands sequence for branches.
  std::vector<CommandBufferCmdSequence> branches(2);

  auto args_access = {MemoryAccess::kRead, MemoryAccess::kRead,
                      MemoryAccess::kWrite};

  {  // Case 0: b = a + a
    auto args = {slice_a, slice_a, slice_b};
    branches[0].Emplace<LaunchCmd>(s0, "AddI32", args, args_access,
                                   LaunchDimensions(1, 4),
                                   /*shmem_bytes=*/0);
  }

  {  // Case 1: b = b + b
    auto args = {slice_b, slice_b, slice_b};
    branches[1].Emplace<LaunchCmd>(s0, "AddI32", args, args_access,
                                   LaunchDimensions(1, 4),
                                   /*shmem_bytes=*/0);
  }

  // Prepare commands sequence for thunk.
  CommandBufferCmdSequence commands;
  commands.Emplace<CaseCmd>(s0, slice_i, std::move(branches));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({index, a, b}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(OwningExecutableSource source, ExecutableSource());
  TF_ASSERT_OK(
      thunk.Initialize({executor, static_cast<Thunk::ExecutableSource>(source),
                        &allocations, stream.get()}));

  // Execute command buffer thunk and verify that it added the value.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 42 + 42));

  // Change `index` to `1` and check that it updated the `b` buffer.
  TF_ASSERT_OK(stream->Memset32(&index, 1, sizeof(int32_t)));

  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));
  ASSERT_EQ(dst, std::vector<int32_t>(4, 2 * (42 + 42)));
}

TEST(CommandBufferThunkTest, ForCmd) {
  se::StreamExecutor* executor = GpuExecutor();

  if (!IsAtLeastCuda12300(executor)) {
    GTEST_SKIP() << "CUDA graph conditionals are not supported";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: loop_cnt=0, a=1, b=0
  se::DeviceMemory<int32_t> loop_cnt = executor->AllocateArray<int32_t>(1, 0);
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&loop_cnt, 0, sizeof(int32_t)));
  TF_ASSERT_OK(stream->Memset32(&a, 1, byte_length));
  TF_ASSERT_OK(stream->MemZero(&b, byte_length));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_cnt(/*index=*/0, 1, /*color=*/0);
  BufferAllocation alloc_a(/*index=*/1, byte_length, /*color=*/0);
  BufferAllocation alloc_b(/*index=*/2, byte_length, /*color=*/0);

  BufferAllocation::Slice slice_cnt(&alloc_cnt, 0, sizeof(int32_t));
  BufferAllocation::Slice slice_a(&alloc_a, 0, byte_length);
  BufferAllocation::Slice slice_b(&alloc_b, 0, byte_length);

  auto args = {slice_a, slice_b, slice_b};  // b = a + b
  auto args_access = {MemoryAccess::kRead, MemoryAccess::kRead,
                      MemoryAccess::kWrite};

  // Prepare commands sequence for loop `body`.
  CommandBufferCmdSequence body_commands;
  body_commands.Emplace<LaunchCmd>(s0, "AddI32", args, args_access,
                                   LaunchDimensions(1, 4),
                                   /*shmem_bytes=*/0);

  // Prepare commands sequence for thunk.
  CommandBufferCmdSequence commands;
  commands.Emplace<ForCmd>(s0, /*num_iterations=*/10, slice_cnt,
                           std::move(body_commands));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(commands), Thunk::ThunkInfo());

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(executor);
  BufferAllocations allocations({loop_cnt, a, b}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(OwningExecutableSource source, ExecutableSource());
  TF_ASSERT_OK(
      thunk.Initialize({executor, static_cast<Thunk::ExecutableSource>(source),
                        &allocations, stream.get()}));

  // Execute command buffer thunk and verify that it added the value 10 times.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy `b` data back to host.
  std::vector<int32_t> dst(4, 0);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), b, byte_length));

  ASSERT_EQ(dst, std::vector<int32_t>(4, 10));
}

TEST(CommandBufferThunkTest, WhileCmd) {
  // TODO(ezhulenev): Find a way to test WhileCmd: add a test only TraceCmd that
  // could allow us trace custom kernels to update while loop iterations. Or
  // maybe add a CustomLaunchCmd and wrap loop update into custom kernel.
}

// TODO(batzor): Implement this. Dependency: DynamicSliceFusion
// TEST_F(CmdBufferTest, DynamicSliceFusionCmd) {
// }

}  // namespace zkx::gpu
