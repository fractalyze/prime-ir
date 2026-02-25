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

#include "zkx/ffi/api/ffi.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/match.h"
#include "absl/synchronization/blocking_counter.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/thread_pool.h"
#include "zkx/executable_run_options.h"
#include "zkx/ffi/call_frame.h"
#include "zkx/ffi/execution_context.h"
#include "zkx/ffi/execution_state.h"
#include "zkx/ffi/ffi_api.h"
#include "zkx/ffi/type_id_registry.h"
#include "zkx/stream_executor/device_memory.h"
#include "zkx/stream_executor/device_memory_allocator.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace zkx::ffi {

enum class Int32BasedEnum : int32_t {
  kOne = 1,
  kTwo = 2,
};

namespace {

constexpr int64_t kI32MaxValue = std::numeric_limits<int32_t>::max();

}  // namespace

enum class Int64BasedEnum : int64_t {
  kOne = kI32MaxValue + 1,
  kTwo = kI32MaxValue + 2,
};

}  // namespace zkx::ffi

ZKX_FFI_REGISTER_ENUM_ATTR_DECODING(::zkx::ffi::Int32BasedEnum);
ZKX_FFI_REGISTER_ENUM_ATTR_DECODING(::zkx::ffi::Int64BasedEnum);

namespace zkx::ffi {

// Use integer types for ZKX (no floating point).
struct PairOfI32AndU32 {
  int32_t i32;
  uint32_t u32;
};

struct TupleOfI32 {
  int32_t i32_0;
  int32_t i32_1;
  int32_t i32_2;
  int32_t i32_3;
};

}  // namespace zkx::ffi

ZKX_FFI_REGISTER_STRUCT_ATTR_DECODING(
    ::zkx::ffi::PairOfI32AndU32, ::zkx::ffi::StructMember<int32_t>("i32"),
    ::zkx::ffi::StructMember<uint32_t>("u32"));
ZKX_FFI_REGISTER_STRUCT_ATTR_DECODING(
    ::zkx::ffi::TupleOfI32, ::zkx::ffi::StructMember<int32_t>("i32_0"),
    ::zkx::ffi::StructMember<int32_t>("i32_1"),
    ::zkx::ffi::StructMember<int32_t>("i32_2"),
    ::zkx::ffi::StructMember<int32_t>("i32_3"));

namespace zkx::ffi {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(FfiApiTest, DataTypeByteWidth) {
  EXPECT_EQ(0, ByteWidth(DataType::TOKEN));
  EXPECT_EQ(0, ByteWidth(DataType::INVALID));

  EXPECT_EQ(1, ByteWidth(DataType::PRED));

  EXPECT_EQ(1, ByteWidth(DataType::S8));
  EXPECT_EQ(2, ByteWidth(DataType::S16));
  EXPECT_EQ(4, ByteWidth(DataType::S32));
  EXPECT_EQ(8, ByteWidth(DataType::S64));

  EXPECT_EQ(1, ByteWidth(DataType::U8));
  EXPECT_EQ(2, ByteWidth(DataType::U16));
  EXPECT_EQ(4, ByteWidth(DataType::U32));
  EXPECT_EQ(8, ByteWidth(DataType::U64));
}

TEST(FfiApiTest, ErrorEnumValue) {
  // Verify that absl::StatusCode and zkx::ffi::ErrorCode use the same
  // integer value for encoding error (status) codes.
  auto encoded = [](auto value) { return static_cast<uint8_t>(value); };

  EXPECT_EQ(encoded(absl::StatusCode::kOk), encoded(ErrorCode::kOk));
  EXPECT_EQ(encoded(absl::StatusCode::kCancelled),
            encoded(ErrorCode::kCancelled));
  EXPECT_EQ(encoded(absl::StatusCode::kUnknown), encoded(ErrorCode::kUnknown));
  EXPECT_EQ(encoded(absl::StatusCode::kInvalidArgument),
            encoded(ErrorCode::kInvalidArgument));
  EXPECT_EQ(encoded(absl::StatusCode::kNotFound),
            encoded(ErrorCode::kNotFound));
  EXPECT_EQ(encoded(absl::StatusCode::kAlreadyExists),
            encoded(ErrorCode::kAlreadyExists));
  EXPECT_EQ(encoded(absl::StatusCode::kPermissionDenied),
            encoded(ErrorCode::kPermissionDenied));
  EXPECT_EQ(encoded(absl::StatusCode::kResourceExhausted),
            encoded(ErrorCode::kResourceExhausted));
  EXPECT_EQ(encoded(absl::StatusCode::kFailedPrecondition),
            encoded(ErrorCode::kFailedPrecondition));
  EXPECT_EQ(encoded(absl::StatusCode::kAborted), encoded(ErrorCode::kAborted));
  EXPECT_EQ(encoded(absl::StatusCode::kOutOfRange),
            encoded(ErrorCode::kOutOfRange));
  EXPECT_EQ(encoded(absl::StatusCode::kUnimplemented),
            encoded(ErrorCode::kUnimplemented));
  EXPECT_EQ(encoded(absl::StatusCode::kInternal),
            encoded(ErrorCode::kInternal));
  EXPECT_EQ(encoded(absl::StatusCode::kUnavailable),
            encoded(ErrorCode::kUnavailable));
  EXPECT_EQ(encoded(absl::StatusCode::kDataLoss),
            encoded(ErrorCode::kDataLoss));
  EXPECT_EQ(encoded(absl::StatusCode::kUnauthenticated),
            encoded(ErrorCode::kUnauthenticated));
}

TEST(FfiApiTest, Expected) {
  ErrorOr<int32_t> value(42);
  EXPECT_TRUE(value.has_value());
  EXPECT_FALSE(value.has_error());
  EXPECT_EQ(*value, 42);

  ErrorOr<int32_t> error(Error(ErrorCode::kInternal, "Test error"));
  EXPECT_FALSE(error.has_value());
  EXPECT_TRUE(error.has_error());
  EXPECT_THAT(error.error().message(), HasSubstr("Test error"));
}

TEST(FfiApiTest, FutureSetAvailable) {
  Promise promise;
  Future future(promise);

  promise.SetAvailable();
  future.OnReady([](const std::optional<Error>& error) {
    EXPECT_FALSE(error.has_value());
  });
}

TEST(FfiApiTest, FutureSetError) {
  Promise promise;
  Future future(promise);

  promise.SetError(Error(ErrorCode::kInternal, "Test error"));
  future.OnReady([](const std::optional<Error>& error) {
    EXPECT_TRUE(error.has_value());
    EXPECT_THAT(error->message(), HasSubstr("Test error"));
  });
}

TEST(FfiApiTest, FutureSetAvailableFromThreadPool) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "ffi-test", 2);

  Promise promise;
  Future future(promise);

  // We write and read to and from the shared variable to check that `OnReady`
  // callback is correctly synchronized with memory writes done in a thread
  // that completes the promise.
  int32_t value = 0;

  absl::BlockingCounter counter(1);

  future.OnReady([&](const std::optional<Error>& error) {
    EXPECT_FALSE(error.has_value());
    EXPECT_EQ(value, 42);
    counter.DecrementCount();
  });

  pool.Schedule([&]() {
    value = 42;
    promise.SetAvailable();
  });

  counter.Wait();
}

TEST(FfiApiTest, FutureSetErrorFromThreadPool) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "ffi-test", 2);

  Promise promise;
  Future future(promise);

  // We write and read to and from the shared variable to check that `OnReady`
  // callback is correctly synchronized with memory writes done in a thread
  // that completes the promise.
  int32_t value = 0;

  absl::BlockingCounter counter(1);

  future.OnReady([&](const std::optional<Error>& error) {
    EXPECT_TRUE(error.has_value());
    EXPECT_THAT(error->message(), HasSubstr("Test error"));
    EXPECT_EQ(value, 42);
    counter.DecrementCount();
  });

  pool.Schedule([&]() {
    value = 42;
    promise.SetError(Error(ErrorCode::kInternal, "Test error"));
  });

  counter.Wait();
}

TEST(FfiApiTest, FutureRace) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "ffi-test", 2);

  // Schedule `SetAvailable` and `OnReady` on a thread pool to detect
  // potential data races. Do this in a loop to make sure that we have
  // a good chance of triggering a data race if there is one.
  for (int32_t i = 0; i < 1000; ++i) {
    Promise promise;
    Future future(promise);

    absl::BlockingCounter counter(1);

    pool.Schedule([&]() { promise.SetAvailable(); });
    pool.Schedule([&]() {
      future.OnReady([&](const std::optional<Error>& error) {
        EXPECT_FALSE(error.has_value());
        counter.DecrementCount();
      });
    });

    counter.Wait();
  }
}

TEST(FfiApiTest, CountDownSuccess) {
  CountDownPromise counter(2);
  Future future(counter);
  EXPECT_FALSE(counter.CountDown());
  EXPECT_TRUE(counter.CountDown());
  future.OnReady([](const std::optional<Error>& error) {
    EXPECT_FALSE(error.has_value());
  });
}

TEST(FfiApiTest, CountDownError) {
  CountDownPromise counter(3);
  Future future(counter);
  EXPECT_FALSE(counter.CountDown());
  EXPECT_FALSE(counter.CountDown(Error(ErrorCode::kInternal, "Test error")));
  EXPECT_TRUE(counter.CountDown());
  future.OnReady([](const std::optional<Error>& error) {
    EXPECT_TRUE(error.has_value());
    EXPECT_THAT(error->message(), HasSubstr("Test error"));
  });
}

TEST(FfiApiTest, CountDownSuccessFromThreadPool) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "ffi-test", 2);

  CountDownPromise counter(2);
  Future future(counter);

  future.OnReady([](const std::optional<Error>& error) {
    EXPECT_FALSE(error.has_value());
  });

  for (int64_t i = 0; i < 2; ++i) {
    pool.Schedule([counter]() mutable { counter.CountDown(); });
  }
}

TEST(FfiApiTest, CountDownErrorFromThreadPool) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "ffi-test", 2);

  CountDownPromise counter(3);
  Future future(counter);

  future.OnReady([](const std::optional<Error>& error) {
    EXPECT_TRUE(error.has_value());
    EXPECT_THAT(error->message(), HasSubstr("Test error"));
  });

  pool.Schedule([counter]() mutable { counter.CountDown(); });
  pool.Schedule([counter]() mutable {
    counter.CountDown(Error(ErrorCode::kInternal, "Test error"));
  });
  pool.Schedule([counter]() mutable { counter.CountDown(); });
}

TEST(FfiApiTest, ReturnError) {
  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().To(
      []() { return Error(ErrorCode::kInternal, "Test error"); });

  auto status = Call(*handler, call_frame);
  EXPECT_EQ(status, absl::InternalError("Test error"));
}

TEST(FfiApiTest, RunId) {
  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Ctx<RunId>().To([&](RunId run_id) {
    EXPECT_EQ(run_id.run_id, 42);
    return Error::Success();
  });

  CallOptions options;
  options.run_id = zkx::RunId{42};

  auto status = Call(*handler, call_frame, options);

  TF_ASSERT_OK(status);
}

TEST(FfiApiTest, AnyBufferArgument) {
  std::vector<uint32_t> storage(4, 0);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(uint32_t));

  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(memory, PrimitiveType::U32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Arg<AnyBuffer>().To([&](auto buffer) {
    EXPECT_EQ(buffer.untyped_data(), storage.data());
    EXPECT_EQ(buffer.template typed_data<uint32_t>(),
              reinterpret_cast<uint32_t*>(storage.data()));
    EXPECT_EQ(buffer.template reinterpret_data<int32_t>(),
              reinterpret_cast<int32_t*>(storage.data()));
    EXPECT_EQ(buffer.dimensions().size(), 2);
    return Error::Success();
  });
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiApiTest, BufferArgument) {
  std::vector<uint32_t> storage(4, 0);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(uint32_t));

  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(memory, PrimitiveType::U32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Arg<BufferR2<U32>>().To([&](auto buffer) {
    EXPECT_EQ(buffer.typed_data(), storage.data());
    EXPECT_EQ(buffer.dimensions().size(), 2);
    return Error::Success();
  });
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiApiTest, AnyBufferResult) {
  std::vector<uint32_t> storage(4, 0);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(uint32_t));

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/1);
  builder.AddBufferRet(memory, PrimitiveType::U32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Ret<AnyBuffer>().To([&](Result<AnyBuffer> buffer) {
    EXPECT_EQ(buffer->untyped_data(), storage.data());
    EXPECT_EQ(buffer->template typed_data<uint32_t>(),
              reinterpret_cast<uint32_t*>(storage.data()));
    EXPECT_EQ(buffer->dimensions().size(), 2);
    return Error::Success();
  });
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiApiTest, MissingBufferArgument) {
  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Arg<BufferR1<U32>>().To(
      [](auto) { return Error::Success(); });
  auto status = Call(*handler, call_frame);

  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("Wrong number of arguments")));
}

TEST(FfiApiTest, WrongRankBufferArgument) {
  std::vector<int32_t> storage(4, 0);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(int32_t));

  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(memory, PrimitiveType::U32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Arg<BufferR1<U32>>().To(
      [](auto) { return Error::Success(); });
  auto status = Call(*handler, call_frame);

  EXPECT_THAT(status,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Wrong buffer rank: expected 1 but got 2")));
}

TEST(FfiApiTest, WrongTypeBufferArgument) {
  std::vector<int32_t> storage(4, 0);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(int32_t));

  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(memory, PrimitiveType::S32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Arg<BufferR2<U32>>().To(
      [](auto) { return Error::Success(); });
  auto status = Call(*handler, call_frame);

  EXPECT_THAT(
      status,
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Wrong buffer dtype: expected U32 but got S32")));
}

TEST(FfiApiTest, WrongNumberOfArguments) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("foo", 42);
  attrs.Insert("bar", 43);

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto handler =
      Ffi::Bind().Attr<int>("foo").To([](int foo) { return Error::Success(); });
  auto status = Call(*handler, call_frame);

  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("Wrong number of attributes")));
  EXPECT_THAT(status.message(), HasSubstr("foo"));
  EXPECT_THAT(status.message(), HasSubstr("bar"));
}

TEST(FfiApiTest, TokenArgument) {
  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(se::DeviceMemoryBase(), PrimitiveType::TOKEN,
                       /*dims=*/{});
  auto call_frame = builder.Build();

  auto fn = [&](Token tok) {
    EXPECT_EQ(tok.typed_data(), nullptr);
    EXPECT_EQ(tok.dimensions().size(), 0);
    return Error::Success();
  };

  auto handler = Ffi::Bind().Arg<Token>().To(fn);
  auto status = Call(*handler, call_frame);
  TF_ASSERT_OK(status);
}

TEST(FfiApiTest, RemainingArgs) {
  std::vector<uint32_t> storage(4, 0);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(uint32_t));

  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(memory, PrimitiveType::U32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto fn = [&](RemainingArgs args) {
    EXPECT_EQ(args.size(), 1);

    ErrorOr<AnyBuffer> arg0 = args.get<AnyBuffer>(0);
    ErrorOr<AnyBuffer> arg1 = args.get<AnyBuffer>(1);

    EXPECT_TRUE(arg0.has_value());
    EXPECT_FALSE(arg1.has_value());

    return Error::Success();
  };

  auto handler = Ffi::Bind().RemainingArgs().To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiApiTest, RemainingRets) {
  std::vector<uint32_t> storage(4, 0);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(uint32_t));

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/2);
  builder.AddBufferRet(memory, PrimitiveType::U32, /*dims=*/{2, 2});
  builder.AddBufferRet(memory, PrimitiveType::U32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto fn = [&](Result<AnyBuffer> ret, RemainingRets rets) {
    EXPECT_EQ(rets.size(), 1);

    ErrorOr<Result<AnyBuffer>> ret0 = rets.get<AnyBuffer>(0);
    ErrorOr<Result<AnyBuffer>> ret1 = rets.get<AnyBuffer>(1);

    EXPECT_TRUE(ret0.has_value());
    EXPECT_FALSE(ret1.has_value());

    return Error::Success();
  };

  auto handler = Ffi::Bind().Ret<AnyBuffer>().RemainingRets().To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiApiTest, OptionalArgs) {
  std::vector<uint32_t> storage(4, 0);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(uint32_t));

  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(memory, PrimitiveType::U32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  {  // Single optional argument.
    auto fn = [&](std::optional<AnyBuffer> arg0) {
      EXPECT_TRUE(arg0.has_value());
      return Error::Success();
    };

    auto handler = Ffi::Bind().OptionalArg<AnyBuffer>().To(fn);
    auto status = Call(*handler, call_frame);

    TF_ASSERT_OK(status);
  }

  {  // Two optional arguments.
    auto fn = [&](std::optional<AnyBuffer> arg0,
                  std::optional<AnyBuffer> arg1) {
      EXPECT_TRUE(arg0.has_value());
      EXPECT_FALSE(arg1.has_value());
      return Error::Success();
    };

    auto handler =
        Ffi::Bind().OptionalArg<AnyBuffer>().OptionalArg<AnyBuffer>().To(fn);
    auto status = Call(*handler, call_frame);

    TF_ASSERT_OK(status);
  }

  {  // Optional argument after a regular one.
    auto fn = [&](AnyBuffer arg0, std::optional<AnyBuffer> arg1) {
      EXPECT_FALSE(arg1.has_value());
      return Error::Success();
    };

    auto handler = Ffi::Bind().Arg<AnyBuffer>().OptionalArg<AnyBuffer>().To(fn);
    auto status = Call(*handler, call_frame);

    TF_ASSERT_OK(status);
  }

  {  // Remaining arguments after optional one.
    auto fn = [&](std::optional<AnyBuffer> arg0, RemainingArgs args) {
      EXPECT_TRUE(arg0.has_value());
      EXPECT_EQ(args.size(), 0);
      return Error::Success();
    };

    auto handler = Ffi::Bind().OptionalArg<AnyBuffer>().RemainingArgs().To(fn);
    auto status = Call(*handler, call_frame);

    TF_ASSERT_OK(status);
  }
}

TEST(FfiApiTest, OptionalRets) {
  std::vector<uint32_t> storage(4, 0);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(uint32_t));

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/1);
  builder.AddBufferRet(memory, PrimitiveType::U32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  {  // Single optional result.
    auto fn = [&](std::optional<Result<AnyBuffer>> ret0) {
      EXPECT_TRUE(ret0.has_value());
      return Error::Success();
    };

    auto handler = Ffi::Bind().OptionalRet<AnyBuffer>().To(fn);
    auto status = Call(*handler, call_frame);

    TF_ASSERT_OK(status);
  }

  {  // Two optional results.
    auto fn = [&](std::optional<Result<AnyBuffer>> ret0,
                  std::optional<Result<AnyBuffer>> ret1) {
      EXPECT_TRUE(ret0.has_value());
      EXPECT_FALSE(ret1.has_value());
      return Error::Success();
    };

    auto handler =
        Ffi::Bind().OptionalRet<AnyBuffer>().OptionalRet<AnyBuffer>().To(fn);
    auto status = Call(*handler, call_frame);

    TF_ASSERT_OK(status);
  }

  {  // Optional result after a regular one.
    auto fn = [&](Result<AnyBuffer> ret0,
                  std::optional<Result<AnyBuffer>> ret1) {
      EXPECT_FALSE(ret1.has_value());
      return Error::Success();
    };

    auto handler = Ffi::Bind().Ret<AnyBuffer>().OptionalRet<AnyBuffer>().To(fn);
    auto status = Call(*handler, call_frame);

    TF_ASSERT_OK(status);
  }

  {  // Remaining results after optional one.
    auto fn = [&](std::optional<Result<AnyBuffer>> ret0, RemainingRets rets) {
      EXPECT_TRUE(ret0.has_value());
      EXPECT_EQ(rets.size(), 0);
      return Error::Success();
    };

    auto handler = Ffi::Bind().OptionalRet<AnyBuffer>().RemainingRets().To(fn);
    auto status = Call(*handler, call_frame);

    TF_ASSERT_OK(status);
  }
}

TEST(FfiApiTest, AutoBinding) {
  static constexpr char kI32[] = "i32";

  auto handler = Ffi::BindTo(+[](AnyBuffer buffer, Attr<int32_t, kI32> foo) {
    EXPECT_EQ(*foo, 42);
    return Error::Success();
  });

  std::vector<uint32_t> storage(4, 0);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(uint32_t));

  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert(kI32, 42);

  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(memory, PrimitiveType::U32, /*dims=*/{2, 2});
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto status = Call(*handler, call_frame);
  TF_ASSERT_OK(status);
}

TEST(FfiApiTest, AutoBindingResult) {
  auto handler =
      Ffi::BindTo(+[](Result<AnyBuffer> buffer) { return Error::Success(); });

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/1);
  builder.AddBufferRet(se::DeviceMemoryBase(), PrimitiveType::U32, /*dims=*/{});
  auto call_frame = builder.Build();

  auto status = Call(*handler, call_frame);
  TF_ASSERT_OK(status);
}

TEST(FfiApiTest, AutoBindingStructs) {
  auto handler = Ffi::BindTo(+[](PairOfI32AndU32 attrs) {
    EXPECT_EQ(attrs.i32, 42);
    EXPECT_EQ(attrs.u32, 42u);
    return Error::Success();
  });

  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32", 42);
  attrs.Insert("u32", static_cast<uint32_t>(42));

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto status = Call(*handler, call_frame);
  TF_ASSERT_OK(status);
}

TEST(FfiApiTest, AutoBindingDictionary) {
  auto handler = Ffi::BindTo(+[](Dictionary attrs) {
    EXPECT_EQ(*attrs.get<int32_t>("i32"), 42);
    EXPECT_EQ(*attrs.get<uint32_t>("u32"), 42u);
    return Error::Success();
  });

  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32", 42);
  attrs.Insert("u32", static_cast<uint32_t>(42));

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto status = Call(*handler, call_frame);
  TF_ASSERT_OK(status);
}

// Use opaque struct to define a platform stream type just like platform
// stream for GPU backend (e.g. `CUstream_st`  and `cudaStream_t`).
struct TestStreamSt;
using TestStream = TestStreamSt*;

template <>
struct CtxBinding<TestStream> {
  using Ctx = PlatformStream<TestStream>;
};

TEST(FfiApiTest, BindingPlatformStreamInference) {
  // We only check that it compiles.
  (void)Ffi::BindTo(+[](TestStream stream) { return Error::Success(); });
}

TEST(FfiApiTest, ArrayAttr) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("arr0", std::vector<int8_t>({1, 2, 3, 4}));
  attrs.Insert("arr1", std::vector<int16_t>({1, 2, 3, 4}));
  attrs.Insert("arr2", std::vector<int32_t>({1, 2, 3, 4}));
  attrs.Insert("arr3", std::vector<int64_t>({1, 2, 3, 4}));
  attrs.Insert("arr4", std::vector<uint8_t>({1, 2, 3, 4}));
  attrs.Insert("arr5", std::vector<uint16_t>({1, 2, 3, 4}));
  attrs.Insert("arr6", std::vector<uint32_t>({1, 2, 3, 4}));
  attrs.Insert("arr7", std::vector<uint64_t>({1, 2, 3, 4}));

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](auto arr0, auto arr1, auto arr2, auto arr3, auto arr4,
                auto arr5, auto arr6, auto arr7) {
    EXPECT_EQ(arr0, Span<const int8_t>({1, 2, 3, 4}));
    EXPECT_EQ(arr1, Span<const int16_t>({1, 2, 3, 4}));
    EXPECT_EQ(arr2, Span<const int32_t>({1, 2, 3, 4}));
    EXPECT_EQ(arr3, Span<const int64_t>({1, 2, 3, 4}));
    EXPECT_EQ(arr4, Span<const uint8_t>({1, 2, 3, 4}));
    EXPECT_EQ(arr5, Span<const uint16_t>({1, 2, 3, 4}));
    EXPECT_EQ(arr6, Span<const uint32_t>({1, 2, 3, 4}));
    EXPECT_EQ(arr7, Span<const uint64_t>({1, 2, 3, 4}));
    return Error::Success();
  };

  auto handler = Ffi::Bind()
                     .Attr<Span<const int8_t>>("arr0")
                     .Attr<Span<const int16_t>>("arr1")
                     .Attr<Span<const int32_t>>("arr2")
                     .Attr<Span<const int64_t>>("arr3")
                     .Attr<Span<const uint8_t>>("arr4")
                     .Attr<Span<const uint16_t>>("arr5")
                     .Attr<Span<const uint32_t>>("arr6")
                     .Attr<Span<const uint64_t>>("arr7")
                     .To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiApiTest, AttrsAsDictionary) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32", 42);
  attrs.Insert("u32", static_cast<uint32_t>(42));
  attrs.Insert("str", "foo");

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](Dictionary dict) {
    EXPECT_EQ(dict.size(), 3);

    EXPECT_TRUE(dict.contains("i32"));
    EXPECT_TRUE(dict.contains("u32"));
    EXPECT_TRUE(dict.contains("str"));

    ErrorOr<int32_t> i32 = dict.get<int32_t>("i32");
    ErrorOr<uint32_t> u32 = dict.get<uint32_t>("u32");
    ErrorOr<std::string_view> str = dict.get<std::string_view>("str");

    EXPECT_TRUE(i32.has_value());
    EXPECT_TRUE(u32.has_value());
    EXPECT_TRUE(str.has_value());

    if (i32.has_value()) EXPECT_EQ(*i32, 42);
    if (u32.has_value()) EXPECT_EQ(*u32, 42u);
    if (str.has_value()) EXPECT_EQ(*str, "foo");

    EXPECT_FALSE(dict.contains("i64"));
    EXPECT_FALSE(dict.get<int64_t>("i32").has_value());
    EXPECT_FALSE(dict.get<int64_t>("i64").has_value());

    return Error::Success();
  };

  auto handler = Ffi::Bind().Attrs().To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiApiTest, DictionaryAttr) {
  CallFrameBuilder::AttributesMap dict0;
  dict0.try_emplace("i32", 42);

  CallFrameBuilder::AttributesMap dict1;
  dict1.try_emplace("u32", static_cast<uint32_t>(42));

  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("dict0", dict0);
  attrs.Insert("dict1", dict1);

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](Dictionary dict0, Dictionary dict1) {
    EXPECT_EQ(dict0.size(), 1);
    EXPECT_EQ(dict1.size(), 1);

    EXPECT_TRUE(dict0.contains("i32"));
    EXPECT_TRUE(dict1.contains("u32"));

    ErrorOr<int32_t> i32 = dict0.get<int32_t>("i32");
    ErrorOr<uint32_t> u32 = dict1.get<uint32_t>("u32");

    EXPECT_TRUE(i32.has_value());
    EXPECT_TRUE(u32.has_value());

    if (i32.has_value()) EXPECT_EQ(*i32, 42);
    if (u32.has_value()) EXPECT_EQ(*u32, 42u);

    return Error::Success();
  };

  auto handler =
      Ffi::Bind().Attr<Dictionary>("dict0").Attr<Dictionary>("dict1").To(fn);

  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiApiTest, StructAttr) {
  CallFrameBuilder::AttributesMap dict;
  dict.try_emplace("i32", 42);
  dict.try_emplace("u32", static_cast<uint32_t>(42));

  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("str", "foo");
  attrs.Insert("i32_and_u32", dict);

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](std::string_view str, PairOfI32AndU32 i32_and_u32) {
    EXPECT_EQ(str, "foo");
    EXPECT_EQ(i32_and_u32.i32, 42);
    EXPECT_EQ(i32_and_u32.u32, 42u);
    return Error::Success();
  };

  auto handler = Ffi::Bind()
                     .Attr<std::string_view>("str")
                     .Attr<PairOfI32AndU32>("i32_and_u32")
                     .To(fn);

  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiApiTest, AttrsAsStruct) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32", 42);
  attrs.Insert("u32", static_cast<uint32_t>(42));

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](PairOfI32AndU32 i32_and_u32) {
    EXPECT_EQ(i32_and_u32.i32, 42);
    EXPECT_EQ(i32_and_u32.u32, 42u);
    return Error::Success();
  };

  auto handler = Ffi::Bind().Attrs<PairOfI32AndU32>().To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiApiTest, PointerAttr) {
  std::string foo = "foo";

  // Test for convenience attr binding that casts i64 attribute to user-type
  // pointers. It's up to the user to guarantee that pointer is valid.
  auto ptr = reinterpret_cast<uintptr_t>(&foo);
  static_assert(sizeof(ptr) == sizeof(int64_t));

  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("ptr", static_cast<int64_t>(ptr));

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](const std::string* str) {
    EXPECT_EQ(*str, "foo");
    return Error::Success();
  };

  auto handler = Ffi::Bind().Attr<Pointer<std::string>>("ptr").To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiApiTest, EnumAttr) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32_one", static_cast<std::underlying_type_t<Int32BasedEnum>>(
                              Int32BasedEnum::kOne));
  attrs.Insert("i32_two", static_cast<std::underlying_type_t<Int32BasedEnum>>(
                              Int32BasedEnum::kTwo));
  attrs.Insert("i64_one", static_cast<std::underlying_type_t<Int64BasedEnum>>(
                              Int64BasedEnum::kOne));
  attrs.Insert("i64_two", static_cast<std::underlying_type_t<Int64BasedEnum>>(
                              Int64BasedEnum::kTwo));

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](Int32BasedEnum i32_one, Int32BasedEnum i32_two,
                Int64BasedEnum i64_one, Int64BasedEnum i64_two) {
    EXPECT_EQ(i32_one, Int32BasedEnum::kOne);
    EXPECT_EQ(i32_two, Int32BasedEnum::kTwo);
    EXPECT_EQ(i64_one, Int64BasedEnum::kOne);
    EXPECT_EQ(i64_two, Int64BasedEnum::kTwo);
    return Error::Success();
  };

  auto handler = Ffi::Bind()
                     .Attr<Int32BasedEnum>("i32_one")
                     .Attr<Int32BasedEnum>("i32_two")
                     .Attr<Int64BasedEnum>("i64_one")
                     .Attr<Int64BasedEnum>("i64_two")
                     .To(fn);

  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiApiTest, WrongEnumAttrType) {
  CallFrameBuilder::AttributesMap dict;
  dict.try_emplace("i32", 42);

  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32_enum1", dict);
  attrs.Insert("i32_enum0", 42u);

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [](Int32BasedEnum, Int32BasedEnum) { return Error::Success(); };

  auto handler = Ffi::Bind()
                     .Attr<Int32BasedEnum>("i32_enum0")
                     .Attr<Int32BasedEnum>("i32_enum1")
                     .To(fn);

  auto status = Call(*handler, call_frame);

  EXPECT_TRUE(absl::StrContains(
      status.message(),
      "Failed to decode all FFI handler operands (bad operands at: 0, 1)"))
      << "status.message():\n"
      << status.message() << "\n";

  EXPECT_TRUE(absl::StrContains(status.message(),
                                "Wrong scalar data type: expected S32 but got"))
      << "status.message():\n"
      << status.message() << "\n";

  EXPECT_TRUE(absl::StrContains(
      status.message(),
      "Wrong attribute type: expected scalar but got dictionary"))
      << "status.message():\n"
      << status.message() << "\n";
}

struct MyData {
  static TypeId id;
  std::string str;
};

TypeId MyData::id = {};  // zero-initialize type id
ZKX_FFI_REGISTER_TYPE(GetZkxFfiApi(), "my_data", &MyData::id);

TEST(FfiApiTest, UserData) {
  MyData data{"foo"};

  ExecutionContext execution_context;
  TF_ASSERT_OK(execution_context.Insert(
      TypeIdRegistry::TypeId(MyData::id.type_id), &data));

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  auto call_frame = builder.Build();

  auto fn = [&](MyData* data) {
    EXPECT_EQ(data->str, "foo");
    return Error::Success();
  };

  auto handler = Ffi::Bind().Ctx<UserData<MyData>>().To(fn);

  CallOptions options;
  options.execution_context = &execution_context;

  auto status = Call(*handler, call_frame, options);

  TF_ASSERT_OK(status);
}

struct MyState {
  static TypeId id;

  explicit MyState(int32_t value) : value(value) {}
  int32_t value;
};

TypeId MyState::id = {};  // zero-initialize type id
ZKX_FFI_REGISTER_TYPE(GetZkxFfiApi(), "state", &MyState::id);

TEST(FfiApiTest, StatefulHandler) {
  ExecutionState execution_state;

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  auto call_frame = builder.Build();

  CallOptions options;
  options.execution_state = &execution_state;

  // FFI instantiation handler that creates a state for FFI handler.
  auto instantiate =
      Ffi::BindInstantiate().To([]() -> ErrorOr<std::unique_ptr<MyState>> {
        return std::make_unique<MyState>(42);
      });

  // FFI execute handler that uses state created by the instantiation handler.
  auto execute = Ffi::Bind().Ctx<State<MyState>>().To([](MyState* state) {
    EXPECT_EQ(state->value, 42);
    return Error::Success();
  });

  // Create `State` and store it in the execution state.
  TF_ASSERT_OK(
      Call(*instantiate, call_frame, options, ExecutionStage::kInstantiate));

  // Check that state was created and forwarded to the execute handler.
  TF_ASSERT_OK(Call(*execute, call_frame, options));
}

TEST(FfiApiTest, ScratchAllocator) {
  static void* kAddr = reinterpret_cast<void*>(0xDEADBEEF);

  // A test only memory allocator that returns a fixed memory address.
  struct TestDeviceMemoryAllocator final : public se::DeviceMemoryAllocator {
    size_t count;

    TestDeviceMemoryAllocator()
        : se::DeviceMemoryAllocator(nullptr), count(0) {}

    absl::StatusOr<se::OwningDeviceMemory> Allocate(int, uint64_t size, bool,
                                                    int64_t) final {
      count++;
      return se::OwningDeviceMemory(se::DeviceMemoryBase(kAddr, size), 0, this);
    }

    absl::Status Deallocate(int, se::DeviceMemoryBase mem) final {
      count--;
      EXPECT_EQ(mem.opaque(), kAddr);
      return absl::OkStatus();
    }

    absl::StatusOr<se::Stream*> GetStream(int) final {
      return absl::UnimplementedError("Not implemented");
    }
  };

  auto fn = [&](ScratchAllocator scratch_allocator) {
    auto mem = scratch_allocator.Allocate(1024);
    EXPECT_EQ(*mem, kAddr);
    return Error::Success();
  };

  TestDeviceMemoryAllocator allocator;

  auto handler = Ffi::Bind().Ctx<ScratchAllocator>().To(fn);

  CallFrame call_frame =
      CallFrameBuilder(/*num_args=*/0, /*num_rets=*/0).Build();

  CallOptions options;
  options.backend_options = CallOptions::GpuOptions{nullptr, &allocator};

  auto status = Call(*handler, call_frame, options);

  TF_ASSERT_OK(status);
  EXPECT_EQ(allocator.count, 0);
}

TEST(FfiApiTest, ScratchAllocatorUnimplemented) {
  auto fn = [&](ScratchAllocator scratch_allocator) {
    auto mem = scratch_allocator.Allocate(1024);
    EXPECT_FALSE(mem.has_value());
    return Error::Success();
  };
  auto handler = Ffi::Bind().Ctx<ScratchAllocator>().To(fn);
  CallFrame call_frame =
      CallFrameBuilder(/*num_args=*/0, /*num_rets=*/0).Build();
  auto status = Call(*handler, call_frame);
  TF_ASSERT_OK(status);
}

TEST(FfiApiTest, ThreadPool) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "ffi-test", 2);
  Eigen::ThreadPoolDevice device(pool.AsEigenThreadPool(), pool.NumThreads());

  auto fn = [&](ThreadPool thread_pool) {
    // Check that we can get the size of the underlying thread pool.
    if (thread_pool.num_threads() != 2) {
      return Error::Internal("Wrong number of threads");
    }

    // Use a pair of blocking counters to check that scheduled task was executed
    // on a thread pool (it would deadlock if executed inline).
    absl::BlockingCounter prepare(1);
    absl::BlockingCounter execute(1);

    thread_pool.Schedule([&] {
      prepare.Wait();
      execute.DecrementCount();
    });

    prepare.DecrementCount();
    execute.Wait();

    return Error::Success();
  };

  auto handler = Ffi::Bind().Ctx<ThreadPool>().To(fn);
  CallFrame call_frame =
      CallFrameBuilder(/*num_args=*/0, /*num_rets=*/0).Build();

  CallOptions options;
  options.backend_options = CallOptions::CpuOptions{&device};

  auto status = Call(*handler, call_frame, options);
  TF_ASSERT_OK(status);
}

TEST(FfiApiTest, AsyncHandler) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "ffi-test", 2);
  Eigen::ThreadPoolDevice device(pool.AsEigenThreadPool(), pool.NumThreads());

  int32_t value = 0;

  // Handler completes execution asynchronously on a given thread pool.
  auto fn = [&](ThreadPool thread_pool) -> Future {
    Promise promise;
    Future future(promise);

    thread_pool.Schedule([&, promise = std::move(promise)]() mutable {
      value = 42;
      promise.SetAvailable();
    });

    return future;
  };

  auto handler = Ffi::Bind().Ctx<ThreadPool>().To(fn);
  CallFrame call_frame =
      CallFrameBuilder(/*num_args=*/0, /*num_rets=*/0).Build();

  CallOptions options;
  options.backend_options = CallOptions::CpuOptions{&device};

  {  // Synchronous call.
    absl::Status status = Call(*handler, call_frame, options);
    TF_ASSERT_OK(status);
    EXPECT_EQ(value, 42);
  }

  value = 0;  // reset value between calls

  {  // Asynchronous call.
    tsl::AsyncValueRef<tsl::Chain> async_value =
        CallAsync(*handler, call_frame, options);
    tsl::BlockUntilReady(async_value);
    ASSERT_TRUE(async_value.IsConcrete());
    EXPECT_EQ(value, 42);
  }
}

TEST(FfiApiTest, Metadata) {
  auto api = GetZkxFfiApi();
  auto handler = Ffi::BindTo([]() { return Error::Success(); });
  auto maybe_metadata = GetMetadata(*handler);
  EXPECT_TRUE(maybe_metadata.ok());
  auto metadata = maybe_metadata.value();
  EXPECT_EQ(metadata.api_version.major_version, api->api_version.major_version);
  EXPECT_EQ(metadata.api_version.minor_version, api->api_version.minor_version);
  EXPECT_EQ(metadata.traits, 0);
}

TEST(FfiApiTest, MetadataTraits) {
  auto handler = Ffi::BindTo([]() { return Error::Success(); },
                             {Traits::kCmdBufferCompatible});
  auto maybe_metadata = GetMetadata(*handler);
  EXPECT_TRUE(maybe_metadata.ok());
  auto metadata = maybe_metadata.value();
  EXPECT_EQ(metadata.api_version.major_version, ZKX_FFI_API_MAJOR);
  EXPECT_EQ(metadata.api_version.minor_version, ZKX_FFI_API_MINOR);
  EXPECT_EQ(metadata.traits, ZKX_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE);
}

}  // namespace zkx::ffi
