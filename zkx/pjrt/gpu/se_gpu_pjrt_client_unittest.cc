/* Copyright 2022 The OpenXLA Authors.
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

#include "zkx/pjrt/gpu/se_gpu_pjrt_client.h"

#include <stdlib.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "mlir/IR/MLIRContext.h"

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/mem.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/tsl/platform/thread_pool.h"
#include "zkx/ffi/ffi.h"
#include "zkx/ffi/ffi_api.h"
#include "zkx/hlo/parser/hlo_parser.h"
#include "zkx/literal.h"
#include "zkx/literal_util.h"
#include "zkx/pjrt/distributed/client.h"
#include "zkx/pjrt/distributed/distributed.h"
#include "zkx/pjrt/distributed/in_memory_key_value_store.h"
#include "zkx/pjrt/gpu/gpu_topology.pb.h"
#include "zkx/pjrt/host_memory_spaces.h"
#include "zkx/pjrt/mlir_to_hlo.h"
#include "zkx/service/platform_util.h"
#include "zkx/shape_util.h"
#include "zkx/status_macros.h"
#include "zkx/stream_executor/cuda/cuda_compute_capability.h"
#include "zkx/stream_executor/device_memory.h"
#include "zkx/stream_executor/stream.h"
#include "zkx/tests/literal_test_util.h"
#include "zkx/types.h"
#include "zkx/util.h"
#include "zkx/zkx.pb.h"
#include "zkx/zkx_data.pb.h"

namespace zkx {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::Ge;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::SizeIs;

absl::StatusOr<std::unique_ptr<zkx::PjRtLoadedExecutable>> CompileExecutable(
    std::string_view program, PjRtClient& client,
    CompileOptions compile_options = CompileOptions()) {
  TF_ASSIGN_OR_RETURN(auto hlo_module,
                      ParseAndReturnUnverifiedModule(program, {}));

  ZkxComputation zkx_computation(hlo_module->ToProto());
  return client.Compile(zkx_computation, compile_options);
}

// Given the result of a PjrtExecutable::Execute call (TF-status of vectors of
// vectors), extract the zeroth result from the zeroth device.
absl::StatusOr<std::shared_ptr<Literal>> ExtractSingleResult(
    absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>&
        result) {
  TF_RETURN_IF_ERROR(result.status());
  TF_RET_CHECK(result->size() == 1);
  std::vector<std::unique_ptr<PjRtBuffer>>& result_buffers = (*result)[0];
  TF_RET_CHECK(result_buffers.size() == 1);
  auto literal_or = result_buffers[0]->ToLiteralSync();
  if (!literal_or.status().ok()) return literal_or.status();
  return *literal_or;
}

static constexpr char const* kProgram = R"(HloModule HostTransfer
    ENTRY SendRecvSynchronous() -> s32[2] {
      in_chain = token[] after-all()

      data = s32[2] constant({2, 3})
      send = (s32[2], u32[], token[]) send(data, in_chain),
        channel_id=1,
        is_host_transfer=true,
        frontend_attributes={
          _zkx_host_transfer_handler_name="undef",
          _zkx_host_transfer_rendezvous="undef"
        }
      send-done = token[] send-done(send),
        channel_id=1, is_host_transfer=true

      recv = (s32[2], u32[], token[]) recv(send-done),
        channel_id=2,
        is_host_transfer=true,
        frontend_attributes={
          _zkx_host_transfer_handler_name="undef",
          _zkx_host_transfer_rendezvous="undef"
        }
      recv-done = (s32[2], token[]) recv-done(recv),
        channel_id=2, is_host_transfer=true

      ROOT result = s32[2] get-tuple-element(recv-done), index=0
    })";

TEST(StreamExecutorGpuClientTest, MemorySpace) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  ASSERT_GE(client->devices().size(), 1);

  for (auto* device : client->devices()) {
    TF_ASSERT_OK_AND_ASSIGN(auto* memory_space, device->default_memory_space());
    EXPECT_EQ(memory_space->kind(), StreamExecutorGpuHbmMemorySpace::kKind);
    EXPECT_EQ(memory_space->kind_id(),
              StreamExecutorGpuHbmMemorySpace::kKindId);
    EXPECT_THAT(
        device->memory_space_by_kind(StreamExecutorGpuHbmMemorySpace::kKind),
        IsOkAndHolds(memory_space));
    EXPECT_EQ(device->memory_spaces().size(), 2);
    auto* pinned = device->memory_spaces()[1];
    EXPECT_EQ(pinned->kind_id(), PinnedHostMemorySpace::kKindId);
    EXPECT_THAT(device->memory_space_by_kind(PinnedHostMemorySpace::kKind),
                IsOkAndHolds(pinned));
  }
}

TEST(StreamExecutorGpuClientTest, MemorySpacesUniqueIds) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  ASSERT_GE(client->devices().size(), 1);

  absl::flat_hash_map<int, std::string> memories;
  for (auto* device : client->devices()) {
    for (auto* memory_space : device->memory_spaces()) {
      std::string debug_string(memory_space->DebugString());
      auto [it, inserted] = memories.insert({memory_space->id(), debug_string});
      EXPECT_TRUE(inserted) << "Duplicate ids for memory spaces '" << it->second
                            << "' and '" << debug_string << "'";
    }
  }
}

TEST(StreamExecutorGpuClientTest, PropagateError) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  auto shape = ShapeUtil::MakeScalarShape(S32);
  absl::Status input_error = absl::InvalidArgumentError("input error");
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->CreateErrorBuffer(
          input_error, shape,
          *client->addressable_devices()[0]->default_memory_space()));

  static constexpr char const* kAddProgram =
      R"(
HloModule Add.6, entry_computation_layout={(s32[], s32[])->(s32[], s32[])}

ENTRY %Add.6 (a.1: s32[], b.2: s32[]) -> (s32[], s32[]) {
  %a.1 = s32[] parameter(0)
  %b.2 = s32[] parameter(1)
  %add.3 = s32[] add(s32[] %a.1, s32[] %b.2)
  %add.4 = s32[] add(s32[] %add.3, s32[] %add.3)
  ROOT %tuple.5 = (s32[], s32[]) tuple(s32[] %add.3, s32[] %add.4)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kAddProgram, *client));

  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      executable->Execute({{buffer.get(), buffer.get()}}, /*options=*/{}));

  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].size(), 1);
  EXPECT_EQ(result[0][0]->GetReadyFuture().Await(), input_error);
}

// TODO(b/372735047): Fix and re-enable.
TEST(StreamExecutorGpuClientTest, DISABLED_DonateWithControlDependency) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  auto shape = ShapeUtil::MakeScalarShape(S32);
  absl::Status input_error = absl::InvalidArgumentError("input error");
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->CreateErrorBuffer(
          input_error, shape,
          *client->addressable_devices()[0]->default_memory_space()));

  static constexpr char const* kAddProgram =
      R"(
HloModule Add.6, entry_computation_layout={(s32[], s32[])->(s32[], s32[])}

ENTRY %Add.6 (a.1: s32[], b.2: s32[]) -> (s32[], s32[]) {
  %a.1 = s32[] parameter(0)
  %b.2 = s32[] parameter(1)
  %add.3 = s32[] add(s32[] %a.1, s32[] %b.2)
  %add.4 = s32[] add(s32[] %add.3, s32[] %add.3)
  ROOT %tuple.5 = (s32[], s32[]) tuple(s32[] %add.3, s32[] %add.4)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kAddProgram, *client));

  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      executable->Execute({{buffer.get(), buffer.get()}}, /*options=*/{}));

  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].size(), 1);

  TF_ASSERT_OK_AND_ASSIGN(
      auto another_buffer,
      client->CreateErrorBuffer(
          input_error, shape,
          *client->addressable_devices()[0]->default_memory_space()));
  TF_ASSERT_OK_AND_ASSIGN(another_buffer,
                          another_buffer->DonateWithControlDependency(
                              result[0][0]->GetReadyFuture()));
  EXPECT_EQ(another_buffer->GetReadyFuture().Await(), input_error);
}

// TODO(chokobole): Enable this test. Dependency: SendThunk
TEST(StreamExecutorGpuClientTest, DISABLED_SendRecvChunked) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));

  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kProgram, *client));

  std::array<int32_t, 2> sent_value = {0, 0};

  // Send buffer to host.
  SendCallback send_callback = {
      /*channel_id=*/1, [&](const PjRtTransferMetadata& m, PjRtChunk chunk,
                            int64_t total_size_in_bytes, bool done) {
        int32_t* data = reinterpret_cast<int32_t*>(chunk.data());
        sent_value[0] = data[0];
        sent_value[1] = data[1];
        return absl::OkStatus();
      }};

  // Recv buffer from host.
  RecvCallback recv_callback = {
      /*channel_id=*/2, [&](const PjRtTransferMetadata& m,
                            std::unique_ptr<CopyToDeviceStream> stream) {
        auto chunk0 = PjRtChunk::AllocateDefault(sizeof(int32_t));
        *reinterpret_cast<int32_t*>(chunk0.data()) = 5;
        CHECK_OK(stream->AddChunk(std::move(chunk0)).Await());

        auto chunk1 = PjRtChunk::AllocateDefault(sizeof(int32_t));
        *reinterpret_cast<int32_t*>(chunk1.data()) = 6;
        CHECK_OK(stream->AddChunk(std::move(chunk1)).Await());

        return absl::OkStatus();
      }};

  // Callbacks for point-to-point communication ops.
  std::vector<std::vector<SendCallback>> send_callbacks = {{send_callback}};
  std::vector<std::vector<RecvCallback>> recv_callbacks = {{recv_callback}};

  ExecuteOptions opts;
  opts.send_callbacks = send_callbacks;
  opts.recv_callbacks = recv_callbacks;

  auto result = executable->Execute(/*argument_handles=*/{{}}, opts);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> result_literal,
                          ExtractSingleResult(result));
  EXPECT_EQ(sent_value[0], 2);
  EXPECT_EQ(sent_value[1], 3);
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>({5, 6}),
                                     *result_literal));
}

// TODO(chokobole): Enable this test. Dependency: SendThunk
TEST(StreamExecutorGpuClientTest, DISABLED_SendErrorNoDeadLock) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));

  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kProgram, *client));

  // Always-failing Send handler.
  SendCallback send_callback = {
      /*channel_id=*/1,
      [&](const PjRtTransferMetadata&, PjRtChunk, int64_t, bool) {
        return absl::InternalError("Uh-oh, can send chunk to host");
      }};

  // No-op Recv handler.
  RecvCallback recv_callback = {
      /*channel_id=*/2, [&](const PjRtTransferMetadata& m,
                            std::unique_ptr<CopyToDeviceStream> stream) {
        return absl::OkStatus();
      }};

  // Callbacks for point-to-point communication ops.
  std::vector<std::vector<SendCallback>> send_callbacks = {{send_callback}};
  std::vector<std::vector<RecvCallback>> recv_callbacks = {{recv_callback}};

  ExecuteOptions opts;
  opts.send_callbacks = send_callbacks;
  opts.recv_callbacks = recv_callbacks;

  // Check that send error safely rejected and we do not dead lock.
  auto result = executable->Execute(/*argument_handles=*/{{}}, opts);
  EXPECT_TRUE(absl::StrContains(result.status().message(),
                                "Uh-oh, can send chunk to host"));
}

// TODO(chokobole): Enable this test. Dependency: SendThunk
TEST(StreamExecutorGpuClientTest, DISABLED_RecvErrorNoDeadLock) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));

  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kProgram, *client));

  // No-op Send handler.
  SendCallback send_callback = {
      /*channel_id=*/1, [&](const PjRtTransferMetadata&, PjRtChunk, int64_t,
                            bool) { return absl::OkStatus(); }};

  // Invalid Recv handler that tries to add invalid chunk.
  RecvCallback recv_callback = {
      /*channel_id=*/2, [&](const PjRtTransferMetadata& m,
                            std::unique_ptr<CopyToDeviceStream> stream) {
        auto chunk = PjRtChunk::AllocateDefault(10 * sizeof(int32_t));
        stream->AddChunk(std::move(chunk)).Await().IgnoreError();
        // Return ok status to proceed to corresponding recv-done call.
        return absl::OkStatus();
      }};

  // Callbacks for point-to-point communication ops.
  std::vector<std::vector<SendCallback>> send_callbacks = {{send_callback}};
  std::vector<std::vector<RecvCallback>> recv_callbacks = {{recv_callback}};

  ExecuteOptions opts;
  opts.send_callbacks = send_callbacks;
  opts.recv_callbacks = recv_callbacks;

  // Check that invalid chunk safely rejected and we do not dead lock.
  auto result = executable->Execute(/*argument_handles=*/{{}}, opts);
  EXPECT_TRUE(absl::StrContains(result.status().message(),
                                "Adding chunk of size 40 would overflow buffer "
                                "of size 8 (0 already transferred)"));
}

// User-defined data type to be passed to FFI handler via the execute context
// side channel.
struct MemsetValue {
  explicit MemsetValue(int32_t value) : value(value) {}
  int32_t value;
};

static absl::Status MemsetFromValue(
    se::Stream* stream, ffi::Result<ffi::BufferR1<PrimitiveType::S32>> result,
    MemsetValue* memset_value) {
  uint32_t pattern;
  std::memcpy(&pattern, &memset_value->value, sizeof(pattern));

  se::DeviceMemoryBase base = result->device_memory();
  return stream->Memset32(&base, pattern, base.size());
}

ZKX_FFI_DEFINE_HANDLER(kMemsetFromValue, MemsetFromValue,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Ret<ffi::BufferR1<PrimitiveType::S32>>()
                           .Ctx<ffi::UserData<MemsetValue>>());

ZKX_FFI_REGISTER_HANDLER(ffi::GetZkxFfiApi(), "MemsetFromValue",
                         PlatformUtil::CanonicalPlatformName("GPU").value(),
                         kMemsetFromValue);

// TODO(chokobole): Enable this test. Dependency: CustomCallThunk
TEST(StreamExecutorGpuClientTest, DISABLED_ForwardUserDataToFfiHandler) {
  static constexpr char const* kProgram = R"(
    HloModule ffi_handler
    ENTRY main {
      ROOT %custom-call = s32[4] custom-call(),
                          custom_call_target="MemsetFromValue",
                          api_version=API_VERSION_TYPED_FFI
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kProgram, *client));

  ExecuteContext context;
  TF_ASSERT_OK(context.ffi_context().Emplace<MemsetValue>(42));

  ExecuteOptions opts;
  opts.context = &context;

  auto result = executable->Execute(/*argument_handles=*/{{}}, opts);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> result_literal,
                          ExtractSingleResult(result));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR1<int32_t>({42, 42, 42, 42}), *result_literal));
}

static absl::Status MemsetFromAttr(
    se::Stream* stream, int32_t attr,
    ffi::Result<ffi::BufferR1<PrimitiveType::S32>> result) {
  uint32_t pattern;
  std::memcpy(&pattern, &attr, sizeof(pattern));

  se::DeviceMemoryBase base = result->device_memory();
  return stream->Memset32(&base, pattern, base.size());
}

ZKX_FFI_DEFINE_HANDLER(kMemsetFromAttr, MemsetFromAttr,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Attr<int32_t>("attr")
                           .Ret<ffi::BufferR1<PrimitiveType::S32>>());

ZKX_FFI_REGISTER_HANDLER(ffi::GetZkxFfiApi(), "MemsetFromAttr",
                         PlatformUtil::CanonicalPlatformName("GPU").value(),
                         kMemsetFromAttr);

// TODO(chokobole): Enable this test. Dependency: CustomCallThunk
TEST(StreamExecutorGpuClientTest, DISABLED_PassAttrToFfiHandler) {
  static constexpr char const* kProgram = R"(
    HloModule ffi_handler
    ENTRY main {
      ROOT %custom-call = s32[4] custom-call(),
          custom_call_target="MemsetFromAttr",
          api_version=API_VERSION_TYPED_FFI,
          backend_config={"custom_call_backend_config": {"attributes": "{attr = 3.0 : s32}"}}
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kProgram, *client));

  ExecuteOptions opts;
  auto result = executable->Execute(/*argument_handles=*/{{}}, opts);
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> result_literal,
                          ExtractSingleResult(result));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR1<int32_t>({3, 3, 3, 3}), *result_literal));
}

TEST(StreamExecutorGpuClientTest, ToLiteralAsync) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);

  auto src_literal = LiteralUtil::CreateR1<int32_t>({41, 42, 43, 44});
  TF_ASSERT_OK_AND_ASSIGN(
      auto transfer_manager,
      client->CreateBuffersForAsyncHostToDevice(
          {src_literal.shape()}, client->addressable_devices()[0]));
  auto buffer = transfer_manager->RetrieveBuffer(0);

  absl::Mutex mu;
  auto literal = std::make_shared<Literal>(
      ShapeUtil::DeviceShapeToHostShape(buffer->on_device_shape()));
  bool got_literal = false;

  TF_ASSERT_OK(
      transfer_manager->TransferLiteralToBuffer(0, src_literal, [&]() {}));

  buffer->ToLiteral(literal.get()).OnReady([&](absl::Status s) {
    absl::MutexLock l(&mu);
    TF_ASSERT_OK(s);
    got_literal = true;
  });
  buffer.reset();

  {
    absl::MutexLock l(&mu);
    mu.Await(absl::Condition(&got_literal));
  }

  ASSERT_TRUE(ShapeUtil::Compatible(src_literal.shape(), literal->shape()));
  ASSERT_EQ(src_literal.data<int32_t>(),
            literal->Relayout(src_literal.shape().layout()).data<int32_t>());
}

TEST(StreamExecutorGpuClientTest, ToLiteralAsyncWithNonCompactLayout) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);

  Shape transposed_shape = ShapeUtil::MakeShapeWithDenseLayout(
      S32, {2, 3}, /*minor_to_major=*/{0, 1});
  Literal src_literal = LiteralUtil::CreateR2WithLayout<int32_t>(
      {{3, 14, 25}, {36, 47, 58}}, transposed_shape.layout());

  PjRtClient::ShapeSpec spec;
  spec.element_type = src_literal.shape().element_type();
  spec.dims = DimensionVector(src_literal.shape().dimensions().begin(),
                              src_literal.shape().dimensions().end());
  std::vector<std::optional<Layout>> device_layouts = {
      std::make_optional(transposed_shape.layout())};
  TF_ASSERT_OK_AND_ASSIGN(
      auto transfer_manager,
      client->CreateBuffersForAsyncHostToDevice(
          {spec}, device_layouts,
          client->addressable_devices()[0]->memory_spaces()[0]));
  auto buffer = transfer_manager->RetrieveBuffer(0);

  absl::Mutex mu;
  auto literal = std::make_shared<Literal>(
      ShapeUtil::DeviceShapeToHostShape(buffer->on_device_shape()));
  bool got_literal = false;

  TF_ASSERT_OK(
      transfer_manager->TransferLiteralToBuffer(0, src_literal, [&]() {}));

  buffer->ToLiteral(literal.get()).OnReady([&](absl::Status s) {
    absl::MutexLock l(&mu);
    TF_ASSERT_OK(s);
    got_literal = true;
  });
  buffer.reset();

  {
    absl::MutexLock l(&mu);
    mu.Await(absl::Condition(&got_literal));
  }

  ASSERT_TRUE(ShapeUtil::Compatible(src_literal.shape(), literal->shape()));
  ASSERT_EQ(src_literal.data<int32_t>(),
            literal->Relayout(src_literal.shape().layout()).data<int32_t>());
}

TEST(StreamExecutorGpuClientTest, ToLiteralAsyncBeforeBufferReady) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);

  auto src_literal = LiteralUtil::CreateR1<int32_t>({41, 42, 43, 44});
  TF_ASSERT_OK_AND_ASSIGN(
      auto transfer_manager,
      client->CreateBuffersForAsyncHostToDevice(
          {src_literal.shape()}, client->addressable_devices()[0]));
  auto buffer = transfer_manager->RetrieveBuffer(0);

  absl::Mutex mu;
  auto literal = std::make_shared<Literal>(
      ShapeUtil::DeviceShapeToHostShape(buffer->on_device_shape()));
  bool got_literal = false;

  buffer->ToLiteral(literal.get()).OnReady([&](absl::Status s) {
    absl::MutexLock l(&mu);
    TF_ASSERT_OK(s);
    got_literal = true;
  });

  absl::SleepFor(absl::Milliseconds(10));
  ASSERT_FALSE(got_literal);
  TF_ASSERT_OK(
      transfer_manager->TransferLiteralToBuffer(0, src_literal, [&]() {}));

  buffer.reset();

  {
    absl::MutexLock l(&mu);
    mu.Await(absl::Condition(&got_literal));
  }

  ASSERT_TRUE(ShapeUtil::Compatible(src_literal.shape(), literal->shape()));
  ASSERT_EQ(src_literal.data<int32_t>(),
            literal->Relayout(src_literal.shape().layout()).data<int32_t>());
}

TEST(StreamExecutorGpuClientTest, FromHostAsync) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);

  std::vector<Literal> src_literals;
  std::vector<Shape> src_shapes;
  for (int i = 0; i < 4; ++i) {
    std::vector<int32_t> data(i + 1);
    std::iota(data.begin(), data.end(), static_cast<int32_t>(i + 10));
    src_literals.emplace_back(LiteralUtil::CreateR1<int32_t>(data));
    src_shapes.push_back(src_literals.back().shape());
  }
  TF_ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                          client->CreateBuffersForAsyncHostToDevice(
                              src_shapes, client->addressable_devices()[0]));
  std::vector<std::unique_ptr<PjRtBuffer>> buffers;
  for (int i = 0; i < src_shapes.size(); ++i) {
    buffers.emplace_back(transfer_manager->RetrieveBuffer(i));
  }

  for (int i = 0; i < src_shapes.size(); ++i) {
    TF_ASSERT_OK(transfer_manager->TransferRawDataToBuffer(
        i,
        std::string_view(static_cast<char*>(src_literals[i].untyped_data()),
                         src_literals[i].size_bytes()),
        [&]() {}));
  }

  absl::Mutex mu;
  std::vector<std::shared_ptr<Literal>> literals;
  int got_literal_count = 0;
  int got_callback_count = 0;

  for (auto& buffer : buffers) {
    literals.push_back(std::make_shared<Literal>(
        ShapeUtil::DeviceShapeToHostShape(buffer->on_device_shape())));
    buffer->ToLiteral(literals.back().get()).OnReady([&](absl::Status s) {
      absl::MutexLock l(&mu);
      TF_ASSERT_OK(s);
      ++got_literal_count;
    });
    buffer->GetReadyFuture().OnReady([&](absl::Status s) {
      absl::MutexLock l(&mu);
      TF_ASSERT_OK(s);
      ++got_callback_count;
    });
    buffer.reset();
  }

  {
    auto done = [&]() {
      return got_literal_count == src_literals.size() &&
             got_callback_count == src_literals.size();
    };
    absl::MutexLock l(&mu);
    mu.Await(absl::Condition(&done));
  }

  for (int i = 0; i < src_literals.size(); ++i) {
    ASSERT_TRUE(
        ShapeUtil::Compatible(src_literals[i].shape(), literals[i]->shape()));
    ASSERT_EQ(src_literals[i].data<int32_t>(),
              literals[i]
                  ->Relayout(src_literals[i].shape().layout())
                  .data<int32_t>());
  }
}

TEST(StreamExecutorGpuClientTest, FromHostAsyncPinnedHost) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(
      auto* pinned_memory_space,
      client->addressable_devices()[0]->memory_space_by_kind(
          PinnedHostMemorySpace::kKind));

  std::vector<Literal> src_literals;
  std::vector<Shape> src_shapes;
  for (int i = 0; i < 4; ++i) {
    std::vector<int32_t> data(i + 1);
    std::iota(data.begin(), data.end(), static_cast<int32_t>(i + 10));
    src_literals.emplace_back(LiteralUtil::CreateR1<int32_t>(data));
    src_shapes.push_back(src_literals.back().shape());
  }
  TF_ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                          client->CreateBuffersForAsyncHostToDevice(
                              src_shapes, pinned_memory_space));
  std::vector<std::unique_ptr<PjRtBuffer>> buffers;
  for (int i = 0; i < src_shapes.size(); ++i) {
    buffers.emplace_back(transfer_manager->RetrieveBuffer(i));
  }

  for (int i = 0; i < src_shapes.size(); ++i) {
    TF_ASSERT_OK(transfer_manager->TransferRawDataToBuffer(
        i,
        std::string_view(static_cast<char*>(src_literals[i].untyped_data()),
                         src_literals[i].size_bytes()),
        [&]() {}));
  }

  absl::Mutex mu;
  std::vector<std::shared_ptr<Literal>> literals;
  int got_literal_count = 0;
  int got_callback_count = 0;

  for (auto& buffer : buffers) {
    literals.push_back(std::make_shared<Literal>(
        ShapeUtil::DeviceShapeToHostShape(buffer->on_device_shape())));
    buffer->ToLiteral(literals.back().get()).OnReady([&](absl::Status s) {
      absl::MutexLock l(&mu);
      TF_ASSERT_OK(s);
      ++got_literal_count;
    });
    buffer->GetReadyFuture().OnReady([&](absl::Status s) {
      absl::MutexLock l(&mu);
      TF_ASSERT_OK(s);
      ++got_callback_count;
    });
    buffer.reset();
  }

  {
    auto done = [&]() {
      return got_literal_count == src_literals.size() &&
             got_callback_count == src_literals.size();
    };
    absl::MutexLock l(&mu);
    mu.Await(absl::Condition(&done));
  }

  for (int i = 0; i < src_literals.size(); ++i) {
    ASSERT_TRUE(
        ShapeUtil::Compatible(src_literals[i].shape(), literals[i]->shape()));
    ASSERT_EQ(src_literals[i].data<int32_t>(),
              literals[i]
                  ->Relayout(src_literals[i].shape().layout())
                  .data<int32_t>());
  }
}

TEST(StreamExecutorGpuClientTest, FromHostAsyncPinnedHostChunked) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  ASSERT_THAT(client->addressable_devices(), SizeIs(Gt(0)));
  TF_ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memspace,
      client->addressable_devices()[0]->memory_space_by_kind(
          PinnedHostMemorySpace::kKind));
  std::vector<int32_t> data{1, 3, 5, 7, 11, 13, 17, 19};
  Shape shape = ShapeUtil::MakeShape(S32, {static_cast<int64_t>(data.size())});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager> txm,
      client->CreateBuffersForAsyncHostToDevice({shape}, memspace));
  std::unique_ptr<PjRtBuffer> buf = txm->RetrieveBuffer(0);
  ASSERT_THAT(buf->GetReadyFuture().IsReady(), Eq(false));

  std::string_view raw_view(reinterpret_cast<char*>(data.data()),
                            data.size() * sizeof(data[0]));
  int offset = 0;
  while (true) {
    int end = offset + 3;  // unaligned chunk size
    if (end > raw_view.size()) {
      end = raw_view.size();
    }
    int sz = end - offset;
    bool reaches_end = end == raw_view.size();
    TF_ASSERT_OK(txm->TransferRawDataToSubBuffer(
        /*buffer_index=*/0, raw_view.data() + offset, offset, sz, reaches_end,
        /*on_done=*/[]() {}));
    if (reaches_end) {
      break;
    }
    offset = end;
  }
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> lit, buf->ToLiteralSync());
  EXPECT_THAT(lit->data<int32_t>(), ElementsAreArray(data));
}

TEST(StreamExecutorGpuClientTest, DeleteBufferThenFulfillBufferNoDeadLock) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  ASSERT_THAT(client->addressable_devices(), SizeIs(Gt(0)));
  TF_ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memspace,
      client->addressable_devices()[0]->memory_space_by_kind(
          PinnedHostMemorySpace::kKind));
  std::vector<int32_t> data{1, 3, 5, 7, 11, 13, 17, 19};
  Shape shape = ShapeUtil::MakeShape(S32, {static_cast<int64_t>(data.size())});
  std::vector<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
      txms;
  for (int i = 0; i < 10000; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager> txm,
        client->CreateBuffersForAsyncHostToDevice({shape}, memspace));
    std::unique_ptr<PjRtBuffer> buf = txm->RetrieveBuffer(0);
    ASSERT_THAT(buf->GetReadyFuture().IsReady(), Eq(false));
    txms.push_back(std::move(txm));
    // Delete the buffer
  }

  // At this point, we have 10000 buffers pending deallocation.

  std::string_view raw_view(reinterpret_cast<char*>(data.data()),
                            data.size() * sizeof(data[0]));
  for (auto& txm : txms) {
    int offset = 0;
    while (true) {
      int end = offset + 3;  // unaligned chunk size
      if (end > raw_view.size()) {
        end = raw_view.size();
      }
      int sz = end - offset;
      bool reaches_end = end == raw_view.size();
      TF_ASSERT_OK(txm->TransferRawDataToSubBuffer(
          /*buffer_index=*/0, raw_view.data() + offset, offset, sz, reaches_end,
          /*on_done=*/[]() {}));
      if (reaches_end) {
        break;
      }
      offset = end;
    }
  }
}

TEST(StreamExecutorGpuClientTest, CopyRawToHostFullBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  auto literal = LiteralUtil::CreateR1<int32_t>({41, 42});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostLiteral(literal, client->memory_spaces()[0]));

  TF_ASSERT_OK_AND_ASSIGN(int64_t size, buffer->GetOnDeviceSizeInBytes());
  void* dst =
      tsl::port::AlignedMalloc(size, tsl::Allocator::kAllocatorAlignment);

  auto result = buffer->CopyRawToHost(dst, 0, size);
  TF_EXPECT_OK(result.Await());
  EXPECT_EQ(*(static_cast<int32_t*>(dst)), 41);
  EXPECT_EQ(*(static_cast<int32_t*>(dst) + 1), 42);

  tsl::port::AlignedSizedFree(dst, tsl::Allocator::kAllocatorAlignment, size);
}

TEST(StreamExecutorGpuClientTest, CopyRawToHostSubBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  auto literal = LiteralUtil::CreateR1<int32_t>({41, 42});

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostLiteral(literal, client->memory_spaces()[0]));
  TF_ASSERT_OK_AND_ASSIGN(int64_t size, buffer->GetOnDeviceSizeInBytes());
  void* dst =
      tsl::port::AlignedMalloc(size, tsl::Allocator::kAllocatorAlignment);

  auto result = buffer->CopyRawToHost(dst, 0, sizeof(int32_t));
  TF_EXPECT_OK(result.Await());
  EXPECT_EQ(*(static_cast<int32_t*>(dst)), 41);

  tsl::port::AlignedSizedFree(dst, tsl::Allocator::kAllocatorAlignment, size);
}

TEST(StreamExecutorGpuClientTest, CopyRawToHostOutOfRange) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  auto literal = LiteralUtil::CreateR1<int32_t>({41, 42});

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostLiteral(literal, client->memory_spaces()[0]));
  TF_ASSERT_OK_AND_ASSIGN(int64_t size, buffer->GetOnDeviceSizeInBytes());
  void* dst =
      tsl::port::AlignedMalloc(size, tsl::Allocator::kAllocatorAlignment);

  auto result = buffer->CopyRawToHost(dst, 1, size);
  EXPECT_THAT(result.Await(), StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("invalid offset 1")));
  tsl::port::AlignedSizedFree(dst, tsl::Allocator::kAllocatorAlignment, size);
}

TEST(StreamExecutorGpuClientTest, CopyRawToHostFuture) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  auto literal = LiteralUtil::CreateR1<int32_t>({41, 42});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostLiteral(literal, client->memory_spaces()[0]));

  auto dst_promise = PjRtFuture<void*>::CreatePromise();
  PjRtFuture<void*> dst_future(dst_promise);

  TF_ASSERT_OK_AND_ASSIGN(int64_t size, buffer->GetOnDeviceSizeInBytes());
  auto ready = buffer->GetReadyFuture();
  auto result = buffer->CopyRawToHostFuture(dst_future, 0, size);

  // Drop the buffer before fulfilling `dst`. The transfer should still keep the
  // buffer alive.
  buffer.reset();
  ready.OnReady([dst_promise = std::move(dst_promise),
                 size](absl::Status status) mutable {
    void* dst =
        tsl::port::AlignedMalloc(size, tsl::Allocator::kAllocatorAlignment);
    dst_promise.Set(dst);
  });

  TF_EXPECT_OK(result.Await());
  TF_ASSERT_OK_AND_ASSIGN(auto* dst, dst_future.Await());
  EXPECT_EQ(*(static_cast<int32_t*>(dst)), 41);
  EXPECT_EQ(*(static_cast<int32_t*>(dst) + 1), 42);

  tsl::port::AlignedSizedFree(dst, tsl::Allocator::kAllocatorAlignment, size);
}

// TODO(chokobole): Enable this test. Dependency: requires >= 2 physical GPUs
TEST(StreamExecutorGpuClientTest, DISABLED_AsyncCopyToDevice) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 2);

  // d0 is the device we will perform local/remote sends from.
  auto* d0 = client->addressable_devices()[0];
  // d1 is the device we will perform local/remote recvs, where the recv
  // sync flag may be contended.
  auto* d1 = client->addressable_devices()[1];

  auto src_literal = LiteralUtil::CreateR1<int32_t>({41, 42, 43, 44});
  TF_ASSERT_OK_AND_ASSIGN(
      auto transfer_manager,
      client->CreateBuffersForAsyncHostToDevice({src_literal.shape()}, d0));
  auto src_buffer = transfer_manager->RetrieveBuffer(0);
  // CopyToMemorySpace won't be enqueued until src_buffer is available.
  auto local_recv_buffer =
      *src_buffer->CopyToMemorySpace(*d1->default_memory_space());

  TF_ASSERT_OK(
      transfer_manager->TransferLiteralToBuffer(0, src_literal, []() {}));

  auto literal = std::make_shared<Literal>(src_literal.shape());

  auto local_recv_literal = local_recv_buffer->ToLiteral(literal.get());
  TF_EXPECT_OK(local_recv_literal.Await());

  ASSERT_TRUE(ShapeUtil::Compatible(src_literal.shape(), literal->shape()));
  ASSERT_EQ(src_literal.data<int32_t>(),
            literal->Relayout(src_literal.shape().layout()).data<int32_t>());
}

TEST(StreamExecutorGpuClientTest, CreateMixOfErrorBuffers) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 1);

  std::vector<Literal> src_literals;
  std::vector<Shape> src_shapes;
  for (int i = 0; i < 4; ++i) {
    std::vector<int32_t> data(i + 1);
    std::iota(data.begin(), data.end(), static_cast<int32_t>(i + 10));
    src_literals.emplace_back(LiteralUtil::CreateR1<int32_t>(data));
    src_shapes.push_back(src_literals.back().shape());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      auto transfer_manager,
      client->CreateBuffersForAsyncHostToDevice(
          src_shapes, client->addressable_devices()[0]->memory_spaces()[0]));
  std::vector<std::unique_ptr<PjRtBuffer>> buffers;
  for (int i = 0; i < src_shapes.size(); ++i) {
    buffers.emplace_back(transfer_manager->RetrieveBuffer(i));
  }

  absl::Mutex mu;
  int got_callback_count = 0;
  for (int i = 0; i < 4; ++i) {
    auto& buffer = buffers[i];
    if (i == 0 || i == 3) {
      TF_ASSERT_OK(transfer_manager->TransferLiteralToBuffer(i, src_literals[i],
                                                             [&]() {}));
      buffer->GetReadyFuture().OnReady([&](absl::Status s) {
        absl::MutexLock l(&mu);
        TF_ASSERT_OK(s);
        ++got_callback_count;
      });
    } else {
      absl::Status error = absl::InternalError(absl::StrCat("error ", i));
      transfer_manager->SetBufferError(i, error);
      buffer->GetReadyFuture().OnReady(
          [error, &mu, &got_callback_count](absl::Status s) {
            absl::MutexLock l(&mu);
            ASSERT_EQ(s, error);
            ++got_callback_count;
          });
    }
    buffer.reset();
  }

  {
    auto done = [&]() { return got_callback_count == src_literals.size(); };
    absl::MutexLock l(&mu);
    QCHECK(mu.AwaitWithTimeout(absl::Condition(&done), absl::Seconds(60)));
  }
}

TEST(GpuTopology, FromProto) {
  GpuTopologyProto msg;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      R"pb(
        device_ids: [ 3, 2, 1 ]
        platform_version: "platform_version"
        num_slices: 2
        num_hosts_per_slice: 1
        num_devices_per_host: 3
      )pb",
      &msg));

  std::unique_ptr<const GpuTopology> gpu_topology = GpuTopology::FromProto(msg);
  EXPECT_THAT(gpu_topology->device_ids(), ElementsAre(3, 2, 1));
  EXPECT_THAT(gpu_topology->platform_version(), "platform_version");
  EXPECT_THAT(gpu_topology->num_slices(), 2);
  EXPECT_THAT(gpu_topology->num_hosts_per_slice(), 1);
  EXPECT_THAT(gpu_topology->num_devices_per_host(), 3);
}

TEST(GpuTopology, ToProto) {
  GpuTopology gpu_topology(/*gpu_device_ids=*/{3, 2, 1},
                           /*platform_version=*/"platform_version",
                           /*num_slices=*/2,
                           /*num_hosts_per_slice=*/1,
                           /*num_devices_per_host=*/3);
  GpuTopologyProto msg = gpu_topology.ToProto();
  EXPECT_THAT(msg.device_ids(), ElementsAre(3, 2, 1));
  EXPECT_THAT(msg.platform_version(), "platform_version");
  EXPECT_THAT(msg.num_slices(), 2);
  EXPECT_THAT(msg.num_hosts_per_slice(), 1);
  EXPECT_THAT(msg.num_devices_per_host(), 3);
}

// TODO(chokobole): Enable this test. Dependency: requires >= 2 physical GPUs
TEST(StreamExecutorGpuClientTest, DISABLED_DistributedInit) {
  auto kv_store = std::make_shared<InMemoryKeyValueStore>();
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "DistributeInit", 4);

  int num_nodes = 2;
  for (int i = 0; i < num_nodes; i++) {
    thread_pool.Schedule([kv_store, i, num_nodes] {
      GpuClientOptions options;
      options.node_id = i;
      options.num_nodes = num_nodes;
      options.kv_store = kv_store;
      TF_ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(options));
      EXPECT_TRUE(client->platform_name() == "cuda" ||
                  client->platform_name() == "rocm");
      EXPECT_EQ(client->addressable_device_count(), 2);
      EXPECT_EQ(client->device_count(), 4);
    });
  }
}

// TODO(chokobole): Enable this test. Dependency: requires >= 2 physical GPUs
TEST(StreamExecutorGpuClientTest, DISABLED_GetAllocatorStatsTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  ASSERT_GE(client->addressable_devices().size(), 2);

  for (auto device : client->addressable_devices()) {
    const Literal literal = LiteralUtil::CreateR0<int32_t>(0);
    TF_ASSERT_OK_AND_ASSIGN(auto* memory_space, device->default_memory_space());
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<PjRtBuffer> buffer,
        client->BufferFromHostLiteral(literal, memory_space));

    auto stats = device->GetAllocatorStats();
    TF_ASSERT_OK(stats.status());
    ASSERT_GT(stats.value().peak_bytes_in_use, 0);
  }
}

TEST(StreamExecutorGpuClientTest, GpuDeviceDescriptionTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  for (int device_index = 0; device_index < client->device_count();
       device_index++) {
    auto coords =
        static_cast<PjRtStreamExecutorDevice*>(client->devices()[device_index])
            ->description()
            .coords();
    EXPECT_EQ(coords[0], device_index);
  }
}

// TODO(chokobole): Enable this test. Dependency: requires >= 2 physical GPUs
// (mock NCCL still reads local_topology.devices().size() for devices_per_host)
TEST(StreamExecutorGpuClientTest, DISABLED_MockNcclClientTest) {
  const int num_nodes = 4;
  GpuClientOptions options;
  options.num_nodes = num_nodes;
  options.enable_mock_nccl = true;
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(options));

  auto devices_per_host = client->addressable_device_count();
  EXPECT_EQ(devices_per_host, 2);
  EXPECT_EQ(client->device_count(), devices_per_host * num_nodes);
  for (int i = 0; i < client->device_count(); i++) {
    auto device = client->devices()[i];
    auto slice_index =
        std::get<int64_t>(device->Attributes().at("slice_index"));
    auto host_index = device->process_index();
    EXPECT_EQ(slice_index, host_index);
  }
}

// TODO(chokobole): Enable this test. Dependency: requires >= 2 physical GPUs
TEST(StreamExecutorGpuClientTest, DISABLED_MockNcclClientWithGpuTopologyTest) {
  GpuClientOptions options;
  options.enable_mock_nccl = true;
  options.num_nodes = 8;
  options.mock_gpu_topology = "2x4x2";
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(options));

  auto devices_per_host = client->addressable_device_count();
  EXPECT_EQ(devices_per_host, 2) << "This test requires 2 local GPUs.";

  TF_ASSERT_OK_AND_ASSIGN(const PjRtTopologyDescription* topology,
                          client->GetTopologyDescription());
  const StreamExecutorGpuTopologyDescription& gpu_topology =
      tensorflow::down_cast<const StreamExecutorGpuTopologyDescription&>(
          *topology);

  EXPECT_EQ(gpu_topology.gpu_topology().num_slices(), 2);
  EXPECT_EQ(gpu_topology.gpu_topology().num_hosts_per_slice(), 4);
  EXPECT_EQ(gpu_topology.gpu_topology().num_devices_per_host(), 2);
}

constexpr char kMlirDistributedSum[] = R"(
module @jit_f attributes {mhlo.num_partitions = 8 : i32,
                          mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8xi32> {
      mhlo.layout_mode = "default",
      mhlo.sharding = "{devices=[8]0,1,2,3,4,5,6,7}"}) -> (tensor<i32> {
          jax.result_info = "",
          mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.reduce(%arg0 init: %c)
        applies stablehlo.add across dimensions = [0]
            : (tensor<8xi32>, tensor<i32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
})";

// TODO(chokobole): Enable this test. Dependency: requires >= 2 physical GPUs
TEST(StreamExecutorGpuClientTest,
     DISABLED_MockNcclClientWithGpuTopologyExecuteTest) {
  GpuClientOptions client_options;
  client_options.enable_mock_nccl = true;
  client_options.num_nodes = 4;
  client_options.mock_gpu_topology = "2x2x2";
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(client_options));

  auto devices_per_host = client->addressable_device_count();
  EXPECT_EQ(devices_per_host, 2) << "This test requires 2 local GPUs.";

  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseMlirModuleString(kMlirDistributedSum, context));

  CompileOptions options;
  options.executable_build_options.set_num_partitions(8)
      .set_use_spmd_partitioning(true)
      .set_allow_spmd_sharding_propagation_to_output({true});
  TF_ASSERT_OK_AND_ASSIGN(auto executable, client->Compile(*module, options));

  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {1}, {0});
  std::vector<std::unique_ptr<PjRtBuffer>> inputs;
  std::vector<std::vector<PjRtBuffer*>> input_ptrs;
  for (int i = 0; i < devices_per_host; i++) {
    auto device = client->addressable_devices()[i];
    std::vector<int32_t> data{i};
    TF_ASSERT_OK_AND_ASSIGN(
        auto input,
        client->BufferFromHostBuffer(
            data.data(), shape.element_type(), shape.dimensions(),
            /*byte_strides=*/std::nullopt,
            PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/nullptr,
            *device->default_memory_space(), /*device_layout=*/nullptr));
    input_ptrs.push_back({input.get()});
    inputs.push_back(std::move(input));
  }

  // Test that running the program does not crash/hang.
  TF_ASSERT_OK(
      executable->Execute(absl::MakeSpan(input_ptrs), ExecuteOptions()));
}

TEST(StreamExecutorGpuClientTest, MockNcclClientWithGpuTopologyMismatchTest) {
  GpuClientOptions options;
  options.enable_mock_nccl = true;
  options.num_nodes = 16;
  options.mock_gpu_topology = "2x4";
  EXPECT_FALSE(GetStreamExecutorGpuClient(options).ok());
}

TEST(StreamExecutorGpuClientTest, BufferFromHostBufferPinnedMemory) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  std::vector<int32_t> data{1, 2, 3, 4};
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  TF_ASSERT_OK_AND_ASSIGN(
      auto* pinned_memory_space,
      client->addressable_devices()[0]->memory_space_by_kind(
          PinnedHostMemorySpace::kKind));
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          pinned_memory_space, /*device_layout=*/nullptr));

  EXPECT_EQ(buffer->memory_space()->kind(), "pinned_host");
  EXPECT_TRUE(buffer->IsOnCpu());

  TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteralSync());
  std::vector<int32_t> expected{1, 2, 3, 4};
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                     *literal));
}

TEST(StreamExecutorGpuClientTest, CopyToPinnedHostMemorySpace) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  std::vector<int32_t> data{1, 2, 3, 4};
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  auto device = client->addressable_devices()[0];
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          *device->default_memory_space(), /*device_layout=*/nullptr));

  EXPECT_EQ(buffer->memory_space()->kind(), "device");

  auto* pinned_memory_space = device->memory_spaces()[1];
  EXPECT_EQ(pinned_memory_space->kind_id(), PinnedHostMemorySpace::kKindId);
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          buffer->CopyToMemorySpace(pinned_memory_space));

  EXPECT_EQ(result->memory_space()->kind(), "pinned_host");
  EXPECT_TRUE(result->IsOnCpu());

  TF_ASSERT_OK_AND_ASSIGN(auto literal, result->ToLiteralSync());
  std::vector<int32_t> expected{1, 2, 3, 4};
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                     *literal));
}

TEST(StreamExecutorGpuClientTest, CopyToPinnedHostMemorySpaceInt4) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  std::vector<int8_t> data{1, 2, 3, 4};
  Shape shape = ShapeUtil::MakeShape(S4, {4});
  auto device = client->addressable_devices()[0];
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          *device->default_memory_space(), /*device_layout=*/nullptr));

  EXPECT_EQ(buffer->memory_space()->kind(), "device");

  auto* pinned_memory_space = device->memory_spaces()[1];
  EXPECT_EQ(pinned_memory_space->kind_id(), PinnedHostMemorySpace::kKindId);
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          buffer->CopyToMemorySpace(pinned_memory_space));

  EXPECT_EQ(result->memory_space()->kind(), "pinned_host");
  EXPECT_TRUE(result->IsOnCpu());

  TF_ASSERT_OK_AND_ASSIGN(auto literal, result->ToLiteralSync());
  std::vector<s4> expected{s4(1), s4(2), s4(3), s4(4)};
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR1<s4>(expected), *literal));
}

TEST(StreamExecutorGpuClientTest, OpaqueDeviceMemoryDataPointer) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  ASSERT_THAT(client->addressable_devices(), SizeIs(Gt(0)));
  PjRtDevice* device = client->addressable_devices()[0];
  TF_ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memspace,
      device->memory_space_by_kind(PinnedHostMemorySpace::kKind));

  // Create a pinned_host buffer
  std::vector<int32_t> int32_data{12, 34, 56, 78};
  Shape shape = ShapeUtil::MakeShapeWithType<int32_t>({4});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buf,
      client->BufferFromHostBuffer(
          static_cast<const void*>(int32_data.data()), shape.element_type(),
          shape.dimensions(), /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr, memspace,
          /*device_layout=*/nullptr));
  ASSERT_THAT(buf->IsOnCpu(), true);
  TF_ASSERT_OK_AND_ASSIGN(size_t buf_sz, buf->GetOnDeviceSizeInBytes());
  ASSERT_THAT(buf_sz, Ge(sizeof(int32_t) * 4));

  // Check that OpaqueDeviceMemoryDataPointer() points to actual data
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtBuffer::ExternalReference> ref,
                          buf->AcquireExternalReference());
  TF_ASSERT_OK(buf->GetReadyFuture().Await());
  const int32_t* int32_ptr =
      reinterpret_cast<const int32_t*>(ref->OpaqueDeviceMemoryDataPointer());
  EXPECT_EQ(*int32_ptr, 12);
  EXPECT_EQ(*(int32_ptr + 1), 34);
  EXPECT_EQ(*(int32_ptr + 2), 56);
  EXPECT_EQ(*(int32_ptr + 3), 78);

  // Copy raw to device using OpaqueDeviceMemoryDataPointer(), and then read
  // back to host; expect to get back the same data
  TF_ASSERT_OK_AND_ASSIGN(PjRtMemorySpace * default_ms,
                          device->default_memory_space());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager> txm,
      client->CreateBuffersForAsyncHostToDevice({shape}, default_ms));
  TF_ASSERT_OK(txm->TransferRawDataToBuffer(
      /*buffer_index=*/0,
      std::string_view(
          static_cast<const char*>(ref->OpaqueDeviceMemoryDataPointer()),
          buf_sz),
      /*on_done=*/[]() {}));
  std::unique_ptr<PjRtBuffer> hbm_buf = txm->RetrieveBuffer(0);
  EXPECT_THAT(hbm_buf->GetOnDeviceSizeInBytes(), IsOkAndHolds(buf_sz));
  EXPECT_THAT(hbm_buf->HostShape(), IsOkAndHolds(shape));
  TF_ASSERT_OK(hbm_buf->GetReadyFuture().Await());
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> literal,
                          hbm_buf->ToLiteralSync());
  EXPECT_THAT(literal->data<int32_t>(), ElementsAreArray(int32_data));
}

namespace {

absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateDeviceBufferForTest(
    PjRtClient* client) {
  auto device = client->addressable_devices()[0];
  TF_EXPECT_OK(device->default_memory_space());

  std::vector<int32_t> data{1, 2, 3, 4};
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {4}, {0});
  TF_ASSIGN_OR_RETURN(
      auto input,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr, *device->default_memory_space(),
          /*device_layout=*/nullptr));
  EXPECT_EQ(input->memory_space()->kind(), "device");
  return input;
}

constexpr char const* kD2HProgram = R"(
  HloModule f

  ENTRY main.5 {
    p = s32[4]{0} parameter(0)
    ROOT cc = s32[4] custom-call(p),
        custom_call_target="annotate_device_placement",
        frontend_attributes={_zkx_buffer_placement="pinned_host"}
  }
)";

constexpr char const* kD2HProgramTupleOutput = R"(
  HloModule f

  ENTRY main.5 {
    p = s32[4]{0} parameter(0)
    cc = s32[4] custom-call(p),
        custom_call_target="annotate_device_placement",
        frontend_attributes={_zkx_buffer_placement="pinned_host"}
    ROOT tuple = (s32[4]{0}, s32[4]{0}) tuple(s32[4]{0} p, s32[4]{0} cc)
  }
)";

constexpr char const* kCollectiveMemorySpaceOutput = R"(

  HloModule jit__psum, entry_computation_layout={(s32[1,4]{1,0})->s32[4]{0}}

  region_0.3 {
    Arg_0.0 = s32[] parameter(0)
    Arg_1.0 = s32[] parameter(1)
    ROOT add.0 = s32[] add(Arg_0.0, Arg_1.0)
  }

  ENTRY main.10_spmd {
    param = s32[1,4]{1,0} parameter(0)
    reshape = s32[4]{0} reshape(param)
    ROOT all-reduce = s32[4]{0} all-reduce(reshape), channel_id=1, to_apply=region_0.3
  }

)";

}  // namespace

// TODO(chokobole): Enable this test. Dependency: CustomCallThunk
TEST(StreamExecutorGpuClientTest, DISABLED_ExecutePinnedHostOutputTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  TF_ASSERT_OK_AND_ASSIGN(auto input, CreateDeviceBufferForTest(client.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kD2HProgram, *client));
  TF_ASSERT_OK_AND_ASSIGN(
      auto result, executable->Execute({{input.get()}}, ExecuteOptions()));

  std::vector<std::unique_ptr<PjRtBuffer>>& result_buffers = result[0];
  EXPECT_EQ(result_buffers[0]->memory_space()->kind(), "pinned_host");

  TF_ASSERT_OK_AND_ASSIGN(auto memory_stats,
                          executable->GetCompiledMemoryStats());
  EXPECT_EQ(memory_stats.output_size_in_bytes, 0);
  EXPECT_EQ(memory_stats.host_output_size_in_bytes, 16);
}

// TODO(chokobole): Enable this test. Dependency: CustomCallThunk
TEST(StreamExecutorGpuClientTest, DISABLED_ExecutePinnedHostOutputTupleTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  TF_ASSERT_OK_AND_ASSIGN(auto input, CreateDeviceBufferForTest(client.get()));

  // Build the output shape with the correct memory space set.
  Shape host_shape = input->on_device_shape();
  host_shape.mutable_layout()->set_memory_space(Layout::kHostMemorySpace);
  Shape out_shape =
      ShapeUtil::MakeTupleShape({input->on_device_shape(), host_shape});

  // Set the result layout so that the compiler assertions on memory
  // spaces pass.
  CompileOptions options;
  options.executable_build_options.set_result_layout(out_shape);

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      CompileExecutable(kD2HProgramTupleOutput, *client, options));

  // Untuple the result so that we get separate buffers.
  // This is how JAX invokes XLA.
  ExecuteOptions execute_options;
  execute_options.untuple_result = true;
  TF_ASSERT_OK_AND_ASSIGN(
      auto result, executable->Execute({{input.get()}}, execute_options));

  std::vector<std::unique_ptr<PjRtBuffer>>& result_buffers = result[0];
  EXPECT_EQ(result_buffers.size(), 2);
  EXPECT_EQ(result_buffers[0]->memory_space()->kind(), "device");
  EXPECT_EQ(result_buffers[1]->memory_space()->kind(), "pinned_host");
}

// TODO(chokobole): Enable this test. Dependency: CustomCallThunk
TEST(StreamExecutorGpuClientTest,
     DISABLED_ExecutablePinnedHostOutputMemoryKindTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kD2HProgram, *client));

  TF_ASSERT_OK_AND_ASSIGN(auto memory_kinds,
                          executable->GetOutputMemoryKinds());
  EXPECT_EQ(memory_kinds.size(), 1);
  EXPECT_EQ(memory_kinds[0].size(), 1);
  EXPECT_EQ(memory_kinds[0][0], "pinned_host");
}

// Verify the output device memory kind with collective memory space shape when
// NCCL user buffer is enabled.
// TODO(chokobole): Enable this test. Dependency: NcclAllReduceThunk
TEST(StreamExecutorGpuClientTest,
     DISABLED_ExecutableCollectiveMemoryOutputMemoryKindTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  CompileOptions options;
  options.executable_build_options.mutable_debug_options()
      ->set_zkx_gpu_enable_nccl_user_buffers(true);

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      CompileExecutable(kCollectiveMemorySpaceOutput, *client, options));
  std::vector<int32_t> data{1, 2, 3, 4};
  // Build the input shape with the correct memory space set.
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {1, 4},
                                                    /*minor_to_major=*/{1, 0});
  shape.mutable_layout()->set_memory_space(Layout::kDefaultMemorySpace);

  auto device = client->addressable_devices()[0];
  TF_EXPECT_OK(device->default_memory_space());
  TF_ASSERT_OK_AND_ASSIGN(
      auto input,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr, *device->default_memory_space(),
          /*device_layout=*/nullptr));
  EXPECT_EQ(input->memory_space()->kind(), "device");

  TF_ASSERT_OK_AND_ASSIGN(auto memory_kinds,
                          executable->GetOutputMemoryKinds());
  EXPECT_EQ(memory_kinds.size(), 1);
  EXPECT_EQ(memory_kinds[0].size(), 1);
  EXPECT_EQ(memory_kinds[0][0], "device");

  TF_ASSERT_OK_AND_ASSIGN(
      auto result, executable->Execute({{input.get()}}, ExecuteOptions()));
  std::vector<std::unique_ptr<PjRtBuffer>>& result_buffers = result[0];
  EXPECT_EQ(result_buffers[0]->memory_space()->kind(), "device");
  Shape result_shape = result_buffers[0]->on_device_shape();
  auto memory_space = result_shape.layout().memory_space();
  EXPECT_EQ(memory_space, 1);
}

// TODO(chokobole): Enable this test. Dependency: CustomCallThunk
TEST(StreamExecutorGpuClientTest,
     DISABLED_ExecutablePinnedHostTupleOutputMemoryKindTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));

  // Build the output shape with the correct memory space set.
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {4}, {0});
  Shape host_shape = shape;
  host_shape.mutable_layout()->set_memory_space(Layout::kHostMemorySpace);
  Shape out_shape = ShapeUtil::MakeTupleShape({shape, host_shape});

  // Set the result layout so that the compiler assertions on memory
  // spaces pass.
  CompileOptions options;
  options.executable_build_options.set_result_layout(out_shape);

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      CompileExecutable(kD2HProgramTupleOutput, *client, options));

  TF_ASSERT_OK_AND_ASSIGN(auto memory_kinds,
                          executable->GetOutputMemoryKinds());
  EXPECT_EQ(memory_kinds.size(), 1);
  EXPECT_EQ(memory_kinds[0].size(), 2);
  EXPECT_EQ(memory_kinds[0][0], "device");
  EXPECT_EQ(memory_kinds[0][1], "pinned_host");
}

// TODO(chokobole): Enable this test. Dependency: stablehlo.custom_call
TEST(StreamExecutorGpuClientTest,
     DISABLED_MlirParameterHostMemorySpaceIsSetInHlo) {
  constexpr char kMlirH2D[] =
      R"(
    func.func public @main(%arg0: tensor<8x2xi32> {
            mhlo.layout_mode = "{1,0}",
            mhlo.memory_kind = "pinned_host",
            mhlo.sharding = "{devices=[2,2]<=[4]}"
        }) -> (tensor<8x2xi32> {
            jax.result_info = "",
            mhlo.layout_mode = "default",
            mhlo.memory_kind = "device",
            mhlo.sharding = "{devices=[2,2]<=[4]}"}) {
      %0 = stablehlo.custom_call @annotate_device_placement(%arg0) {
              has_side_effect = true,
              mhlo.frontend_attributes = {_zkx_buffer_placement = "device"}
          } : (tensor<8x2xi32>) -> tensor<8x2xi32>
      return %0 : tensor<8x2xi32>
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));

  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseMlirModuleString(kMlirH2D, context));

  TF_ASSERT_OK_AND_ASSIGN(auto executable, client->Compile(*module, {}));
  TF_ASSERT_OK_AND_ASSIGN(auto modules, executable->GetHloModules());

  auto first_param_layout =
      modules[0]->entry_computation_layout().parameter_layout(0).layout();
  EXPECT_EQ(first_param_layout.memory_space(), Layout::kHostMemorySpace);
  auto result_layout =
      modules[0]->entry_computation_layout().result_layout().layout();
  EXPECT_EQ(result_layout.memory_space(), Layout::kDefaultMemorySpace);
}

// TODO(chokobole): Enable this test. Dependency: stablehlo.custom_call
TEST(StreamExecutorGpuClientTest,
     DISABLED_MlirResultHostMemorySpaceIsSetInHlo) {
  constexpr char kMlirD2H[] =
      R"(
    func.func public @main(%arg0: tensor<8x2xi32> {
            mhlo.layout_mode = "{1,0}",
            mhlo.memory_kind = "device",
            mhlo.sharding = "{devices=[2,2]<=[4]}"
        }) -> (tensor<8x2xi32> {
            jax.result_info = "",
            mhlo.layout_mode = "default",
            mhlo.memory_kind = "pinned_host",
            mhlo.sharding = "{devices=[2,2]<=[4]}"}) {
      %0 = stablehlo.custom_call @annotate_device_placement(%arg0) {
              has_side_effect = true,
              mhlo.frontend_attributes = {_zkx_buffer_placement = "pinned_host"}
          } : (tensor<8x2xi32>) -> tensor<8x2xi32>
      return %0 : tensor<8x2xi32>
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));

  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseMlirModuleString(kMlirD2H, context));

  TF_ASSERT_OK_AND_ASSIGN(auto executable, client->Compile(*module, {}));
  TF_ASSERT_OK_AND_ASSIGN(auto modules, executable->GetHloModules());

  auto first_param_layout =
      modules[0]->entry_computation_layout().parameter_layout(0).layout();
  EXPECT_EQ(first_param_layout.memory_space(), Layout::kDefaultMemorySpace);
  auto result_layout =
      modules[0]->entry_computation_layout().result_layout().layout();
  EXPECT_EQ(result_layout.memory_space(), Layout::kHostMemorySpace);
}

TEST(StreamExecutorGpuClientTest, ProfileExecution) {
  static constexpr char const* kProgram = R"(
    HloModule profiled
      ENTRY main {
      c0 = s32[] constant(20)
      c1 = s32[] constant(21)
      ROOT res = s32[] add(c0, c1)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CompileExecutable(kProgram, *client));
  ExecutionProfile profile;
  ExecuteOptions opts;
  opts.execution_profile = &profile;
  TF_ASSERT_OK(executable->Execute(/*argument_handles=*/{{}}, opts));
  EXPECT_GT(profile.compute_time_ns(), 0);
}

// TODO(chokobole): Enable this test. Dependency: RunLayoutAssignmentPasses()
TEST(StreamExecutorGpuClientTest, DISABLED_MlirAutoResultLayoutIsSet) {
  constexpr char kMlirWithParameterLayout[] =
      R"(
    func.func public @main(%arg0: tensor<2x4x2xi32> {
            mhlo.layout_mode = "{2, 1, 0}"
        }) -> (tensor<2x2x4xi32> {
            jax.result_info = "",
            mhlo.layout_mode = "auto"}) {
      %0 = stablehlo.transpose %arg0, dims = [0, 2, 1]
          : (tensor<2x4x2xi32>) -> tensor<2x2x4xi32>
      return %0 : tensor<2x2x4xi32>
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));

  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseMlirModuleString(kMlirWithParameterLayout, context));

  TF_ASSERT_OK_AND_ASSIGN(auto executable, client->Compile(*module, {}));
  TF_ASSERT_OK_AND_ASSIGN(auto modules, executable->GetHloModules());

  auto result_layout =
      modules[0]->entry_computation_layout().result_layout().layout();
  EXPECT_EQ(result_layout, Layout({1, 2, 0}));
}

// TODO(chokobole): Enable this test. Dependency: RunLayoutAssignmentPasses
TEST(StreamExecutorGpuClientTest, DISABLED_MlirAutoParameterLayoutIsSet) {
  constexpr char kMlirWithParameterLayout[] =
      R"(
    func.func public @main(%arg0: tensor<2x4x2xi32> {
            mhlo.layout_mode = "auto"
        }) -> (tensor<2x2x4xi32> {
            jax.result_info = "",
            mhlo.layout_mode = "{2, 1, 0}"}) {
      %0 = stablehlo.transpose %arg0, dims = [0, 2, 1]
          : (tensor<2x4x2xi32>) -> tensor<2x2x4xi32>
      return %0 : tensor<2x2x4xi32>
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));

  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseMlirModuleString(kMlirWithParameterLayout, context));

  TF_ASSERT_OK_AND_ASSIGN(auto executable, client->Compile(*module, {}));
  TF_ASSERT_OK_AND_ASSIGN(auto modules, executable->GetHloModules());

  auto first_param_layout =
      modules[0]->entry_computation_layout().parameter_layout(0).layout();
  EXPECT_EQ(first_param_layout, Layout({1, 2, 0}));
}

// TODO(chokobole): Enable this test. Dependency: LayoutNormalizationPass
TEST(StreamExecutorGpuClientTest, DISABLED_MlirParameterLayoutIsSetInHlo) {
  constexpr char kMlirWithParameterLayout[] =
      R"(
    func.func public @main(%arg0: tensor<2x2x2xi32> {
            mhlo.layout_mode = "{0, 2, 1}"
        }) -> (tensor<2x2x2xi32> {
            jax.result_info = "",
            mhlo.layout_mode = "default"}) {
      return %arg0 : tensor<2x2x2xi32>
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));

  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseMlirModuleString(kMlirWithParameterLayout, context));

  TF_ASSERT_OK_AND_ASSIGN(auto executable, client->Compile(*module, {}));
  TF_ASSERT_OK_AND_ASSIGN(auto modules, executable->GetHloModules());

  auto first_param_layout =
      modules[0]->entry_computation_layout().parameter_layout(0).layout();
  EXPECT_EQ(first_param_layout, Layout({0, 2, 1}));
}

// TODO(chokobole): Enable this test. Dependency: LayoutNormalizationPass
TEST(StreamExecutorGpuClientTest,
     DISABLED_MlirParameterLayoutFromOptionsIsSetInHlo) {
  constexpr char kMlirCopy[] =
      R"(
    func.func public @main(%arg0: tensor<2x2x2xi32> {
            mhlo.layout_mode = "default"
        }) -> (tensor<2x2x2xi32> {
            jax.result_info = "",
            mhlo.layout_mode = "default"}) {
      return %arg0 : tensor<2x2x2xi32>
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));

  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseMlirModuleString(kMlirCopy, context));

  CompileOptions options;
  options.argument_layouts = {
      {ShapeUtil::MakeShapeWithDenseLayout(S32, {2, 2, 2}, {0, 2, 1})}};
  TF_ASSERT_OK_AND_ASSIGN(auto executable, client->Compile(*module, options));
  TF_ASSERT_OK_AND_ASSIGN(auto modules, executable->GetHloModules());

  auto first_param_layout =
      modules[0]->entry_computation_layout().parameter_layout(0).layout();
  EXPECT_EQ(first_param_layout, Layout({0, 2, 1}));
}

// clang-format off
// TODO(chokobole): Enable this test. Dependency: MAP_STABLEHLO_TO_HLO(CustomCallOp)
// clang-format on
TEST(StreamExecutorGpuClientTest,
     DISABLED_MlirResultHostMemorySpaceIsSetInHloWithShardingPropagation) {
  constexpr std::string_view mlir_mul_explicit_sharding_layout_and_memory =
      R"mlir(
  module @jit_f attributes {
      mhlo.num_partitions = 2 : i32,
      mhlo.num_replicas = 1 : i32
  } {
    func.func public @main(%arg0: tensor<8x2xi32> {
            mhlo.layout_mode = "{1,0}",
            mhlo.memory_kind = "device",
            mhlo.sharding = "{devices=[1,2]<=[2]}"
        }) -> (tensor<8x2xi32> {
            jax.result_info = "",
            mhlo.layout_mode = "{0,1}",
            mhlo.memory_kind = "pinned_host"
        }) {
      %c = stablehlo.constant dense<2> : tensor<i32>
      %0 = stablehlo.broadcast_in_dim %c, dims = []
          : (tensor<i32>) -> tensor<8x2xi32>
      %1 = stablehlo.multiply %arg0, %0 : tensor<8x2xi32>
      %2 = stablehlo.custom_call @Sharding(%1) {
              mhlo.sharding = "{devices=[1,2]<=[2]}"
          } : (tensor<8x2xi32>) -> tensor<8x2xi32>
      %3 = stablehlo.custom_call @annotate_device_placement(%2) {
              has_side_effect = true,
              mhlo.frontend_attributes = {
                  _zkx_buffer_placement = "pinned_host"
              }
          } : (tensor<8x2xi32>) -> tensor<8x2xi32>
      return %3 : tensor<8x2xi32>
    }
  })mlir";

  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseMlirModuleString(
                       mlir_mul_explicit_sharding_layout_and_memory, context));
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));

  CompileOptions options;
  options.executable_build_options.set_num_partitions(2)
      .set_use_spmd_partitioning(true)
      .set_allow_spmd_sharding_propagation_to_output({true});

  TF_ASSERT_OK_AND_ASSIGN(auto executable, client->Compile(*module, options));
  TF_ASSERT_OK_AND_ASSIGN(auto modules, executable->GetHloModules());

  auto first_param_layout =
      modules[0]->entry_computation_layout().parameter_layout(0).layout();
  EXPECT_EQ(first_param_layout.memory_space(), Layout::kDefaultMemorySpace);
  auto result_layout =
      modules[0]->entry_computation_layout().result_layout().layout();
  EXPECT_EQ(result_layout,
            Layout({0, 1}).set_memory_space(Layout::kHostMemorySpace));

  // Verify that the executable's layout callback is null.
  // This is necessary for the executable to be serializable.
  EXPECT_EQ(executable->GetCompileOptions()
                .value()
                .executable_build_options.layout_canonicalization_callback(),
            nullptr);
}

TEST(StreamExecutorGpuClientTest, GetDefaultLayout) {
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  auto shape = ShapeUtil::MakeShape(S4, {2, 2});
  TF_ASSERT_OK_AND_ASSIGN(
      auto layout,
      client->GetDefaultLayout(shape.element_type(), shape.dimensions()));
  EXPECT_EQ(layout.element_size_in_bits(), 4);
}

// TODO(chokobole): Enable this test. Dependency: RunLayoutAssignmentPasses()
TEST(StreamExecutorGpuClientTest, DISABLED_AutoLayoutIsSupported) {
  const char* hlo_text = R"(
  HloModule DotLayout,
    entry_computation_layout={(s32[2,3,5],s32[3,4,5])->s32[5,2,4]{2,1,0}}

  ENTRY dot {
    p0 = s32[2,3,5]{2,1,0} parameter(0)
    p1 = s32[3,4,5]{2,1,0} parameter(1)
    ROOT dot.1330.10585 = s32[5,2,4]{2,1,0} dot(p0, p1),
      lhs_batch_dims={2}, lhs_contracting_dims={1},
      rhs_batch_dims={2}, rhs_contracting_dims={0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> m,
      ParseAndReturnUnverifiedModule(
          hlo_text, {}, HloParserOptions().set_fill_missing_layouts(false)));

  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetStreamExecutorGpuClient(GpuClientOptions()));
  CompileOptions compile_options;
  compile_options.executable_build_options.mutable_debug_options()
      ->set_zkx_pjrt_allow_auto_layout_in_hlo(true);
  ZkxComputation computation = m->ToProto();
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          client->Compile(computation, compile_options));
  TF_ASSERT_OK_AND_ASSIGN(auto layouts, executable->GetParameterLayouts());
  // Check that the assigned layouts are not default.
  EXPECT_NE(layouts[0]->ToString(), "{2,1,0}");
  EXPECT_NE(layouts[1]->ToString(), "{2,1,0}");
}

struct ShardedAutotuningTestInfo {
  bool use_zkx_computation;
  int num_active_nodes;
  int num_nodes_using_cache;

  static std::string Name(
      const ::testing::TestParamInfo<ShardedAutotuningTestInfo>& info) {
    return absl::StrFormat(
        "computation_%d_active_%d_cache_%d", info.param.use_zkx_computation,
        info.param.num_active_nodes, info.param.num_nodes_using_cache);
  }
};

class ShardedAutotuningTest
    : public ::testing::TestWithParam<ShardedAutotuningTestInfo> {
 public:
  static constexpr int kNumNodes = 2;
};

namespace {

const char* test_binary_name;

}  // namespace

// TODO(chokobole): Enable this test. Dependency: requires >= 2 physical GPUs
TEST_P(ShardedAutotuningTest, DISABLED_ShardedAutotuningWorks) {
  ShardedAutotuningTestInfo param = GetParam();

  std::string cache_dir;
  CHECK(tsl::Env::Default()->LocalTempFilename(&cache_dir));

  tsl::setenv("TF_CPP_VMODULE", "gemm_fusion_autotuner=1", /*overwrite=*/true);

  // Compile twice to test both empty and non-empty disk cache.
  for (int iteration = 0; iteration < 2; ++iteration) {
    tsl::SubProcess child[kNumNodes];
    for (int node_id = 0; node_id < kNumNodes; ++node_id) {
      std::vector<std::string> argv;
      argv.reserve(6);
      argv.push_back(test_binary_name);
      argv.push_back(absl::StrFormat("--node_id=%d", node_id));
      argv.push_back(absl::StrFormat("--use_zkx_computation=%d",
                                     param.use_zkx_computation));
      argv.push_back(
          absl::StrFormat("--num_active_nodes=%d", param.num_active_nodes));
      argv.push_back(absl::StrFormat("--num_nodes_using_cache=%d",
                                     param.num_nodes_using_cache));
      argv.push_back(absl::StrFormat("--cache_dir=%s", cache_dir));
      child[node_id].SetProgram(test_binary_name, argv);
      child[node_id].SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
      child[node_id].SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
      ASSERT_TRUE(child[node_id].Start()) << "node " << node_id;
    }
    for (int node_id = 0; node_id < kNumNodes; ++node_id) {
      std::string stdout_str;
      std::string stderr_str;
      int child_status =
          child[node_id].Communicate(nullptr, &stdout_str, &stderr_str);
      if (WIFEXITED(child_status) &&
          WEXITSTATUS(child_status) ==
              static_cast<int>(absl::StatusCode::kFailedPrecondition)) {
        GTEST_SKIP() << "Requires Ampere+ GPU.";
      }
      EXPECT_EQ(child_status, 0) << " node " << node_id << "\nstdout:\n"
                                 << stdout_str << "\nstderr:\n"
                                 << stderr_str;
      if (node_id < param.num_active_nodes) {
        int num_fusions_to_autotune = (node_id == 0) ? 1 : 0;
        if (iteration > 0 && node_id < param.num_nodes_using_cache) {
          num_fusions_to_autotune = 0;
        }
        EXPECT_THAT(stderr_str,
                    HasSubstr(absl::StrFormat(
                        "Rank %d / %d: autotuning %d / 1 fusions", node_id,
                        param.num_active_nodes, num_fusions_to_autotune)));
      } else {
        EXPECT_THAT(stderr_str, Not(HasSubstr("autotuning")));
      }
    }
  }
}

absl::Status ShardedAutotuningWorksTestBody(const int node_id,
                                            const int num_active_nodes,
                                            const int num_nodes_using_cache,
                                            std::string_view cache_dir,
                                            bool use_zkx_computation) {
  std::unique_ptr<DistributedRuntimeService> service;
  if (node_id == 0) {
    TF_ASSIGN_OR_RETURN(
        service,
        GetDistributedRuntimeService(
            "[::]:12345", CoordinationServiceImpl::Options{
                              .num_nodes = ShardedAutotuningTest::kNumNodes}));
  }

  DistributedRuntimeClient::Options distributed_options;
  distributed_options.node_id = node_id;
  distributed_options.init_timeout = absl::Seconds(120);
  auto distributed_client =
      GetDistributedRuntimeClient("127.0.0.1:12345", distributed_options);
  QCHECK_OK(distributed_client->Connect());
  GpuClientOptions options;
  options.node_id = node_id;
  options.allowed_devices = {node_id};
  options.num_nodes = ShardedAutotuningTest::kNumNodes;
  options.kv_store = GetDistributedKeyValueStore(distributed_client,
                                                 /*key_prefix=*/"gpu:");
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                      GetStreamExecutorGpuClient(options));
  TF_RET_CHECK(client->platform_name() == "cuda");
  TF_ASSIGN_OR_RETURN(
      se::CudaComputeCapability cc,
      se::CudaComputeCapability::FromString(std::get<std::string>(
          client->addressable_devices().front()->Attributes().at(
              "compute_capability"))));
  if (!cc.IsAtLeastAmpere()) {
    return absl::FailedPreconditionError("Ampere+ GPU required");
  }
  TF_RET_CHECK(client->addressable_device_count() == 1);
  TF_RET_CHECK(client->device_count() == ShardedAutotuningTest::kNumNodes);

  if (node_id >= num_active_nodes) {
    // Inactive nodes connect to the coordination service but don't compile.
    return absl::OkStatus();
  }

  CompileOptions compile_options;
  compile_options.executable_build_options.set_num_replicas(num_active_nodes);
  DebugOptions& debug_options =
      *compile_options.executable_build_options.mutable_debug_options();
  debug_options.set_zkx_gpu_shard_autotuning(true);

  if (node_id < num_nodes_using_cache) {
    debug_options.set_zkx_gpu_per_fusion_autotune_cache_dir(
        std::string(cache_dir));
  }

  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      ParseMlirModuleString(R"mlir(
      func.func public @main(%arg0: tensor<2x32x32xf16>) ->
      (tensor<2x32x32xf16> {jax.result_info = ""}) {
        %0 = stablehlo.dot_general %arg0, %arg0, batching_dims = [0] x [0],
        contracting_dims = [2] x [1]
          : (tensor<2x32x32xf16>, tensor<2x32x32xf16>) ->
          tensor<2x32x32xf16>
        return %0 : tensor<2x32x32xf16>
      })mlir",
                                            context));
  std::unique_ptr<PjRtLoadedExecutable> executable;
  if (use_zkx_computation) {
    ZkxComputation computation;
    TF_RETURN_IF_ERROR(MlirToZkxComputation(*module, computation,
                                            /*use_tuple_args=*/false,
                                            /*return_tuple=*/false,
                                            /*use_shardy=*/false));
    TF_ASSIGN_OR_RETURN(executable,
                        client->Compile(computation, compile_options));
  } else {
    TF_ASSIGN_OR_RETURN(executable, client->Compile(*module, compile_options));
  }

  const std::string optimized_hlo =
      executable->GetHloModules()->front()->ToString();
  TF_RET_CHECK(absl::StrContains(optimized_hlo, "triton_gemm"))
      << optimized_hlo;

  return absl::OkStatus();
}

INSTANTIATE_TEST_SUITE_P(
    ShardedAutotuningTest, ShardedAutotuningTest,
    ::testing::ValuesIn(std::vector<ShardedAutotuningTestInfo>{{true, 2, 0},
                                                               {false, 2, 0},
                                                               {false, 1, 0},
                                                               {false, 2, 1},
                                                               {false, 2, 2}}),
    ShardedAutotuningTestInfo::Name);

}  // namespace
}  // namespace zkx

int main(int argc, char* argv[]) {
  // Save name of binary so that it may invoke itself.
  zkx::test_binary_name = argv[0];
  int node_id = -1;
  int num_active_nodes = -1;
  int num_nodes_using_cache = -1;
  std::string cache_dir;
  bool use_zkx_computation = false;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("node_id", &node_id,
                "Node ID for ShardedAutotuningWorks test."),
      tsl::Flag("num_active_nodes", &num_active_nodes,
                "Test parameter for ShardedAutotuningWorks."),
      tsl::Flag("num_nodes_using_cache", &num_nodes_using_cache,
                "Test parameter for ShardedAutotuningWorks."),
      tsl::Flag("cache_dir", &cache_dir,
                "Test parameter for ShardedAutotuningWorks."),
      tsl::Flag("use_zkx_computation", &use_zkx_computation,
                "Test parameter for ShardedAutotuningWorks."),
  };
  zkx::AppendDebugOptionsFlags(&flag_list);
  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  tsl::Flags::Parse(&argc, argv, flag_list);
  testing::InitGoogleTest(&argc, argv);
  if (node_id >= 0) {
    absl::Status result = zkx::ShardedAutotuningWorksTestBody(
        node_id, num_active_nodes, num_nodes_using_cache, cache_dir,
        use_zkx_computation);
    if (!result.ok()) {
      LOG(ERROR) << result.ToString();
    }
    return result.raw_code();
  }
  return RUN_ALL_TESTS();
}
