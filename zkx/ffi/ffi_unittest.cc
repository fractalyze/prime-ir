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

#include "zkx/ffi/ffi.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status_matchers.h"
#include "absl/strings/match.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"
#include "zkx/ffi/call_frame.h"
#include "zkx/ffi/ffi_api.h"
#include "zkx/stream_executor/device_memory.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

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
using ::testing::_;
using ::testing::HasSubstr;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

TEST(FfiTest, StaticHandlerRegistration) {
  static constexpr auto* noop = +[] { return absl::OkStatus(); };

  // Use explicit binding specification.
  ZKX_FFI_DEFINE_HANDLER(NoOp0, noop, Ffi::Bind(),
                         {Traits::kCmdBufferCompatible});

  // Automatically infer binding specification from function signature.
  ZKX_FFI_DEFINE_HANDLER(NoOp1, noop);

  ZKX_FFI_REGISTER_HANDLER(GetZkxFfiApi(), "no-op-0", "Host", NoOp0);
  ZKX_FFI_REGISTER_HANDLER(GetZkxFfiApi(), "no-op-1", "Host", NoOp1);

  auto handler0 = FindHandler("no-op-0", "Host");
  auto handler1 = FindHandler("no-op-1", "Host");

  TF_ASSERT_OK(handler0.status());
  TF_ASSERT_OK(handler1.status());

  ASSERT_EQ(handler0->traits, ZKX_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE);
  ASSERT_EQ(handler1->traits, 0);

  // Check that platform name was canonicalized an we can find handlers
  // registered for "Host" platform as "Cpu" handlers.
  TF_ASSERT_OK_AND_ASSIGN(auto handlers, StaticRegisteredHandlers("Cpu"));
  EXPECT_THAT(handlers,
              UnorderedElementsAre(Pair("no-op-0", _), Pair("no-op-1", _)));
}

TEST(FfiTest, RegistrationTraitsBackwardsCompatibility) {
  static constexpr auto* noop = +[] { return absl::OkStatus(); };
  ZKX_FFI_DEFINE_HANDLER(NoOp, noop, Ffi::Bind());
  ZKX_FFI_REGISTER_HANDLER(GetZkxFfiApi(), "traits-bwd-compat", "Host", NoOp,
                           ZKX_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE);
  auto handler = FindHandler("traits-bwd-compat", "Host");
  TF_ASSERT_OK(handler.status());
  ASSERT_EQ(handler->traits, ZKX_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE);
}

// Declare ZKX FFI handler as a function (extern "C" declaration).
ZKX_FFI_DECLARE_HANDLER_SYMBOL(NoOpHandler);

// Define ZKX FFI handler as a function forwarded to `NoOp` implementation.
static absl::Status NoOp() { return absl::OkStatus(); }
ZKX_FFI_DEFINE_HANDLER_SYMBOL(NoOpHandler, NoOp, Ffi::Bind());

TEST(FfiTest, StaticHandlerSymbolRegistration) {
  ZKX_FFI_REGISTER_HANDLER(GetZkxFfiApi(), "no-op-sym-0", "Host", NoOpHandler);

  // Use "Cpu" platform to check that platform name was canonicalized.
  auto handler0 = FindHandler("no-op-sym-0", "Cpu");

  TF_ASSERT_OK(handler0.status());
  ASSERT_EQ(handler0->traits, 0);
}

TEST(FfiTest, ForwardError) {
  auto call_frame = CallFrameBuilder(/*num_args=*/0, /*num_rets=*/0).Build();
  auto handler = Ffi::Bind().To([] { return absl::AbortedError("Ooops!"); });
  auto status = Call(*handler, call_frame);
  ASSERT_EQ(status.message(), "Ooops!");
}

TEST(FfiTest, CatchException) {
  auto call_frame = CallFrameBuilder(/*num_args=*/0, /*num_rets=*/0).Build();
  ZKX_FFI_DEFINE_HANDLER(
      handler,
      []() {
        throw std::runtime_error("Ooops!");
        return absl::OkStatus();
      },
      Ffi::Bind());
  auto status = Call(*handler, call_frame);
  ASSERT_EQ(status.message(), "ZKX FFI call failed: Ooops!");
}

TEST(FfiTest, CatchExceptionExplicit) {
  auto call_frame = CallFrameBuilder(/*num_args=*/0, /*num_rets=*/0).Build();
  auto handler = Ffi::Bind().To([]() {
    throw std::runtime_error("Ooops!");
    return absl::OkStatus();
  });
  auto status = Call(*handler, call_frame);
  ASSERT_EQ(status.message(), "ZKX FFI call failed: Ooops!");
}

TEST(FfiTest, WrongNumArgs) {
  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(se::DeviceMemoryBase(nullptr), PrimitiveType::U32, {});
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Arg<AnyBuffer>().Arg<AnyBuffer>().To(
      [](AnyBuffer, AnyBuffer) { return absl::OkStatus(); });

  auto status = Call(*handler, call_frame);

  ASSERT_EQ(status.message(),
            "Wrong number of arguments: expected 2 but got 1");
}

TEST(FfiTest, WrongNumAttrs) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32", 42);
  attrs.Insert("u32", static_cast<uint32_t>(42));

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Attr<int32_t>("i32").To(
      [](int32_t) { return absl::OkStatus(); });

  auto status = Call(*handler, call_frame);

  EXPECT_THAT(
      status,
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Wrong number of attributes: expected 1 but got 2")));
}

TEST(FfiTest, RunId) {
  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Ctx<RunId>().To([&](RunId run_id) {
    EXPECT_EQ(run_id.ToInt(), 42);
    return absl::OkStatus();
  });

  CallOptions options;
  options.run_id = RunId{42};

  auto status = Call(*handler, call_frame, options);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, BuiltinAttributes) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("pred", true);
  attrs.Insert("i8", static_cast<int8_t>(42));
  attrs.Insert("i16", static_cast<int16_t>(42));
  attrs.Insert("i32", static_cast<int32_t>(42));
  attrs.Insert("i64", static_cast<int64_t>(42));
  attrs.Insert("u32", static_cast<uint32_t>(42));
  attrs.Insert("u64", static_cast<uint64_t>(42));
  attrs.Insert("str", "foo");

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](bool pred, int8_t i8, int16_t i16, int32_t i32, int64_t i64,
                uint32_t u32, uint64_t u64, std::string_view str) {
    EXPECT_EQ(pred, true);
    EXPECT_EQ(i8, 42);
    EXPECT_EQ(i16, 42);
    EXPECT_EQ(i32, 42);
    EXPECT_EQ(i64, 42);
    EXPECT_EQ(u32, 42u);
    EXPECT_EQ(u64, 42u);
    EXPECT_EQ(str, "foo");
    return absl::OkStatus();
  };

  auto handler = Ffi::Bind()
                     .Attr<bool>("pred")
                     .Attr<int8_t>("i8")
                     .Attr<int16_t>("i16")
                     .Attr<int32_t>("i32")
                     .Attr<int64_t>("i64")
                     .Attr<uint32_t>("u32")
                     .Attr<uint64_t>("u64")
                     .Attr<std::string_view>("str")
                     .To(fn);

  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, BuiltinAttributesAutoBinding) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32", 42);
  attrs.Insert("u32", static_cast<uint32_t>(42));
  attrs.Insert("str", "foo");

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  static constexpr char kI32[] = "i32";
  static constexpr char kU32[] = "u32";
  static constexpr char kStr[] = "str";

  auto fn = [&](Attr<int32_t, kI32> i32, Attr<uint32_t, kU32> u32,
                Attr<std::string_view, kStr> str) {
    EXPECT_EQ(*i32, 42);
    EXPECT_EQ(*u32, 42u);
    EXPECT_EQ(*str, "foo");
    return absl::OkStatus();
  };

  auto handler = Ffi::BindTo(fn);
  auto status = Call(*handler, call_frame);
  TF_ASSERT_OK(status);
}

TEST(FfiTest, ArrayAttr) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("arr0", std::vector<int8_t>({1, 2, 3, 4}));
  attrs.Insert("arr1", std::vector<int16_t>({1, 2, 3, 4}));
  attrs.Insert("arr2", std::vector<int32_t>({1, 2, 3, 4}));
  attrs.Insert("arr3", std::vector<int64_t>({1, 2, 3, 4}));

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](auto arr0, auto arr1, auto arr2, auto arr3) {
    EXPECT_EQ(arr0, absl::Span<const int8_t>({1, 2, 3, 4}));
    EXPECT_EQ(arr1, absl::Span<const int16_t>({1, 2, 3, 4}));
    EXPECT_EQ(arr2, absl::Span<const int32_t>({1, 2, 3, 4}));
    EXPECT_EQ(arr3, absl::Span<const int64_t>({1, 2, 3, 4}));
    return absl::OkStatus();
  };

  auto handler = Ffi::Bind()
                     .Attr<absl::Span<const int8_t>>("arr0")
                     .Attr<absl::Span<const int16_t>>("arr1")
                     .Attr<absl::Span<const int32_t>>("arr2")
                     .Attr<absl::Span<const int64_t>>("arr3")
                     .To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, PointerAttr) {
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
    return absl::OkStatus();
  };

  auto handler = Ffi::Bind().Attr<Pointer<std::string>>("ptr").To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, AttrsAsDictionary) {
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

    absl::StatusOr<int32_t> i32 = dict.get<int32_t>("i32");
    absl::StatusOr<uint32_t> u32 = dict.get<uint32_t>("u32");
    absl::StatusOr<std::string_view> str = dict.get<std::string_view>("str");

    EXPECT_TRUE(i32.ok());
    EXPECT_TRUE(u32.ok());
    EXPECT_TRUE(str.ok());

    if (i32.ok()) EXPECT_EQ(*i32, 42);
    if (u32.ok()) EXPECT_EQ(*u32, 42u);
    if (str.ok()) EXPECT_EQ(*str, "foo");

    EXPECT_FALSE(dict.contains("i64"));
    EXPECT_FALSE(dict.get<int64_t>("i32").ok());
    EXPECT_FALSE(dict.get<int64_t>("i64").ok());

    return absl::OkStatus();
  };

  auto handler = Ffi::Bind().Attrs().To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, DictionaryAttr) {
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

    absl::StatusOr<int32_t> i32 = dict0.get<int32_t>("i32");
    absl::StatusOr<uint32_t> u32 = dict1.get<uint32_t>("u32");

    EXPECT_TRUE(i32.ok());
    EXPECT_TRUE(u32.ok());

    if (i32.ok()) EXPECT_EQ(*i32, 42);
    if (u32.ok()) EXPECT_EQ(*u32, 42u);

    return absl::OkStatus();
  };

  auto handler =
      Ffi::Bind().Attr<Dictionary>("dict0").Attr<Dictionary>("dict1").To(fn);

  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, StructAttr) {
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
    return absl::OkStatus();
  };

  auto handler = Ffi::Bind()
                     .Attr<std::string_view>("str")
                     .Attr<PairOfI32AndU32>("i32_and_u32")
                     .To(fn);

  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, AttrsAsStruct) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32", 42);
  attrs.Insert("u32", static_cast<uint32_t>(42));

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](PairOfI32AndU32 i32_and_u32) {
    EXPECT_EQ(i32_and_u32.i32, 42);
    EXPECT_EQ(i32_and_u32.u32, 42u);
    return absl::OkStatus();
  };

  auto handler = Ffi::Bind().Attrs<PairOfI32AndU32>().To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, DecodingErrors) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32", 42);
  attrs.Insert("i64", 42);
  attrs.Insert("u32", static_cast<uint32_t>(42));
  attrs.Insert("str", "foo");

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [](int32_t, int64_t, uint32_t, std::string_view) {
    return absl::OkStatus();
  };

  auto handler = Ffi::Bind()
                     .Attr<int32_t>("not_i32_should_fail")
                     .Attr<int64_t>("not_i64_should_fail")
                     .Attr<uint32_t>("u32")
                     .Attr<std::string_view>("not_str_should_fail")
                     .To(fn);

  auto status = Call(*handler, call_frame);

  EXPECT_TRUE(absl::StrContains(
      status.message(),
      "Failed to decode all FFI handler operands (bad operands at: 0, 1, 3)"))
      << "status.message():\n"
      << status.message() << "\n";

  EXPECT_TRUE(absl::StrContains(
      status.message(), "Attribute name mismatch: i32 vs not_i32_should_fail"))
      << "status.message():\n"
      << status.message() << "\n";

  EXPECT_TRUE(absl::StrContains(
      status.message(), "Attribute name mismatch: i64 vs not_i64_should_fail"))
      << "status.message():\n"
      << status.message() << "\n";

  EXPECT_TRUE(absl::StrContains(
      status.message(), "Attribute name mismatch: str vs not_str_should_fail"))
      << "status.message():\n"
      << status.message() << "\n";
}

TEST(FfiTest, AnyBufferArgument) {
  std::vector<uint32_t> storage(4, 0);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(uint32_t));

  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(memory, PrimitiveType::U32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto fn = [&](AnyBuffer buffer) {
    EXPECT_EQ(buffer.element_type(), PrimitiveType::U32);
    EXPECT_EQ(buffer.untyped_data(), storage.data());
    EXPECT_EQ(buffer.typed_data<uint32_t>(),
              reinterpret_cast<uint32_t*>(storage.data()));
    EXPECT_EQ(buffer.reinterpret_data<int32_t>(),
              reinterpret_cast<int32_t*>(storage.data()));
    AnyBuffer::Dimensions dimensions = buffer.dimensions();
    EXPECT_EQ(dimensions.size(), 2);
    EXPECT_EQ(dimensions[0], 2);
    EXPECT_EQ(dimensions[1], 2);
    return absl::OkStatus();
  };

  {  // Test explicit binding signature declaration.
    auto handler = Ffi::Bind().Arg<AnyBuffer>().To(fn);
    auto status = Call(*handler, call_frame);
    TF_ASSERT_OK(status);
  }

  {  // Test inferring binding signature from a handler type.
    auto handler = Ffi::BindTo(fn);
    auto status = Call(*handler, call_frame);
    TF_ASSERT_OK(status);
  }
}

TEST(FfiTest, TypedAndRankedBufferArgument) {
  std::vector<uint32_t> storage(4, 0);
  se::DeviceMemoryBase memory(storage.data(),
                              storage.size() * sizeof(uint32_t));

  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(memory, PrimitiveType::U32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto fn = [&](BufferR2<PrimitiveType::U32> buffer) {
    EXPECT_EQ(buffer.untyped_data(), storage.data());
    EXPECT_EQ(buffer.element_count(), storage.size());
    EXPECT_EQ(buffer.dimensions().size(), 2);
    return absl::OkStatus();
  };

  {  // Test explicit binding signature declaration.
    auto handler = Ffi::Bind().Arg<BufferR2<PrimitiveType::U32>>().To(fn);
    auto status = Call(*handler, call_frame);
    TF_ASSERT_OK(status);
  }

  {  // Test inferring binding signature from a handler type.
    auto handler = Ffi::BindTo(fn);
    auto status = Call(*handler, call_frame);
    TF_ASSERT_OK(status);
  }
}

TEST(FfiTest, TokenArgument) {
  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(se::DeviceMemoryBase(), PrimitiveType::TOKEN,
                       /*dims=*/{});
  auto call_frame = builder.Build();

  auto fn = [&](Token tok) {
    EXPECT_EQ(tok.untyped_data(), nullptr);
    EXPECT_EQ(tok.dimensions().size(), 0);
    return absl::OkStatus();
  };

  auto handler = Ffi::Bind().Arg<Token>().To(fn);
  auto status = Call(*handler, call_frame);
  TF_ASSERT_OK(status);
}

TEST(FfiTest, WrongRankBufferArgument) {
  std::vector<int32_t> storage(4, 0);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(int32_t));

  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(memory, PrimitiveType::U32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Arg<BufferR1<PrimitiveType::U32>>().To(
      [](auto) { return absl::OkStatus(); });
  auto status = Call(*handler, call_frame);

  EXPECT_THAT(status,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Wrong buffer rank: expected 1 but got 2")));
}

TEST(FfiTest, WrongTypeBufferArgument) {
  std::vector<int32_t> storage(4, 0);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(int32_t));

  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(memory, PrimitiveType::S32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Arg<BufferR2<PrimitiveType::U32>>().To(
      [](auto) { return absl::OkStatus(); });
  auto status = Call(*handler, call_frame);

  EXPECT_THAT(
      status,
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Wrong buffer dtype: expected u32 but got s32")));
}

TEST(FfiTest, RemainingArgs) {
  std::vector<uint32_t> storage(4, 0);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(uint32_t));

  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(memory, PrimitiveType::U32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto fn = [&](RemainingArgs args) {
    EXPECT_EQ(args.size(), 1);

    absl::StatusOr<AnyBuffer> arg0 = args.get<AnyBuffer>(0);
    absl::StatusOr<AnyBuffer> arg1 = args.get<AnyBuffer>(1);

    EXPECT_TRUE(arg0.ok());
    EXPECT_THAT(arg1.status(), StatusIs(absl::StatusCode::kInvalidArgument,
                                        HasSubstr("Index out of range")));

    return absl::OkStatus();
  };

  auto handler = Ffi::Bind().RemainingArgs().To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, RemainingRets) {
  std::vector<uint32_t> storage(4, 0);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(uint32_t));

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/2);
  builder.AddBufferRet(memory, PrimitiveType::U32, /*dims=*/{2, 2});
  builder.AddBufferRet(memory, PrimitiveType::U32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto fn = [&](Result<AnyBuffer> ret, RemainingRets rets) {
    EXPECT_EQ(rets.size(), 1);

    absl::StatusOr<Result<AnyBuffer>> ret0 = rets.get<AnyBuffer>(0);
    absl::StatusOr<Result<AnyBuffer>> ret1 = rets.get<AnyBuffer>(1);

    EXPECT_TRUE(ret0.ok());
    EXPECT_THAT(ret1.status(), StatusIs(absl::StatusCode::kInvalidArgument,
                                        HasSubstr("Index out of range")));

    return absl::OkStatus();
  };

  auto handler = Ffi::Bind().Ret<AnyBuffer>().RemainingRets().To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, OptionalArgs) {
  std::vector<uint32_t> storage(4, 0);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(uint32_t));

  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(memory, PrimitiveType::U32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  {  // Single optional argument.
    auto fn = [&](std::optional<AnyBuffer> arg0) {
      EXPECT_TRUE(arg0.has_value());
      return absl::OkStatus();
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
      return absl::OkStatus();
    };

    auto handler =
        Ffi::Bind().OptionalArg<AnyBuffer>().OptionalArg<AnyBuffer>().To(fn);
    auto status = Call(*handler, call_frame);

    TF_ASSERT_OK(status);
  }

  {  // Optional argument after a regular one.
    auto fn = [&](AnyBuffer arg0, std::optional<AnyBuffer> arg1) {
      EXPECT_FALSE(arg1.has_value());
      return absl::OkStatus();
    };

    auto handler = Ffi::Bind().Arg<AnyBuffer>().OptionalArg<AnyBuffer>().To(fn);
    auto status = Call(*handler, call_frame);

    TF_ASSERT_OK(status);
  }

  {  // Remaining arguments after optional one.
    auto fn = [&](std::optional<AnyBuffer> arg0, RemainingArgs args) {
      EXPECT_TRUE(arg0.has_value());
      EXPECT_EQ(args.size(), 0);
      return absl::OkStatus();
    };

    auto handler = Ffi::Bind().OptionalArg<AnyBuffer>().RemainingArgs().To(fn);
    auto status = Call(*handler, call_frame);

    TF_ASSERT_OK(status);
  }
}

TEST(FfiTest, OptionalRets) {
  std::vector<uint32_t> storage(4, 0);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(uint32_t));

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/1);
  builder.AddBufferRet(memory, PrimitiveType::U32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  {  // Single optional result.
    auto fn = [&](std::optional<Result<AnyBuffer>> ret0) {
      EXPECT_TRUE(ret0.has_value());
      return absl::OkStatus();
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
      return absl::OkStatus();
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
      return absl::OkStatus();
    };

    auto handler = Ffi::Bind().Ret<AnyBuffer>().OptionalRet<AnyBuffer>().To(fn);
    auto status = Call(*handler, call_frame);

    TF_ASSERT_OK(status);
  }

  {  // Remaining results after optional one.
    auto fn = [&](std::optional<Result<AnyBuffer>> ret0, RemainingRets rets) {
      EXPECT_TRUE(ret0.has_value());
      EXPECT_EQ(rets.size(), 0);
      return absl::OkStatus();
    };

    auto handler = Ffi::Bind().OptionalRet<AnyBuffer>().RemainingRets().To(fn);
    auto status = Call(*handler, call_frame);

    TF_ASSERT_OK(status);
  }
}

TEST(FfiTest, RunOptionsCtx) {
  auto call_frame = CallFrameBuilder(/*num_args=*/0, /*num_rets=*/0).Build();
  auto* expected = reinterpret_cast<se::Stream*>(0x01234567);

  auto fn = [&](const se::Stream* run_options) {
    EXPECT_EQ(run_options, expected);
    return absl::OkStatus();
  };

  CallOptions options;
  options.backend_options = CallOptions::GpuOptions{expected};

  auto handler = Ffi::Bind().Ctx<Stream>().To(fn);
  auto status = Call(*handler, call_frame, options);

  TF_ASSERT_OK(status);
}

struct StrUserData {
  explicit StrUserData(std::string str) : str(std::move(str)) {}
  std::string str;
};

TEST(FfiTest, UserData) {
  ExecutionContext execution_context;
  TF_ASSERT_OK(execution_context.Emplace<StrUserData>("foo"));

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  auto call_frame = builder.Build();

  auto fn = [&](StrUserData* data) {
    EXPECT_EQ(data->str, "foo");
    return absl::OkStatus();
  };

  CallOptions options;
  options.execution_context = &execution_context;

  auto handler = Ffi::Bind().Ctx<UserData<StrUserData>>().To(fn);
  auto status = Call(*handler, call_frame, options);

  TF_ASSERT_OK(status);
}

struct StrState {
  explicit StrState(std::string str) : str(std::move(str)) {}
  std::string str;
};

TEST(FfiTest, StatefulHandler) {
  ExecutionState execution_state;

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  auto call_frame = builder.Build();

  CallOptions options;
  options.execution_state = &execution_state;

  // FFI instantiation handler that creates a state for FFI handler.
  auto instantiate = Ffi::BindInstantiate().To(
      []() -> absl::StatusOr<std::unique_ptr<StrState>> {
        return std::make_unique<StrState>("foo");
      });

  // FFI execute handler that uses state created by the instantiation handler.
  auto execute = Ffi::Bind().Ctx<State<StrState>>().To([](StrState* state) {
    EXPECT_EQ(state->str, "foo");
    return absl::OkStatus();
  });

  // Create `State` and store it in the execution state.
  TF_ASSERT_OK(
      Call(*instantiate, call_frame, options, ExecutionStage::kInstantiate));

  // Check that state was created and forwarded to the execute handler.
  TF_ASSERT_OK(Call(*execute, call_frame, options));
}

TEST(FfiTest, UpdateBufferArgumentsAndResults) {
  std::vector<uint32_t> storage0(4, 0);
  std::vector<uint32_t> storage1(4, 0);

  se::DeviceMemoryBase memory0(storage0.data(), 4 * sizeof(uint32_t));
  se::DeviceMemoryBase memory1(storage1.data(), 4 * sizeof(uint32_t));

  std::vector<int64_t> dims = {2, 2};

  auto bind = Ffi::Bind()
                  .Arg<BufferR2<PrimitiveType::U32>>()
                  .Ret<BufferR2<PrimitiveType::U32>>()
                  .Attr<int32_t>("n");

  // `fn0` expects argument to be `memory0` and result to be `memory1`.
  auto fn0 = [&](BufferR2<PrimitiveType::U32> arg,
                 Result<BufferR2<PrimitiveType::U32>> ret, int32_t n) {
    EXPECT_EQ(arg.untyped_data(), storage0.data());
    EXPECT_EQ(ret->untyped_data(), storage1.data());
    EXPECT_EQ(arg.dimensions(), dims);
    EXPECT_EQ(ret->dimensions(), dims);
    EXPECT_EQ(n, 42);
    return absl::OkStatus();
  };

  // `fn1` expects argument to be `memory1` and result to be `memory0`.
  auto fn1 = [&](BufferR2<PrimitiveType::U32> arg,
                 Result<BufferR2<PrimitiveType::U32>> ret, int32_t n) {
    EXPECT_EQ(arg.untyped_data(), storage1.data());
    EXPECT_EQ(ret->untyped_data(), storage0.data());
    EXPECT_EQ(arg.dimensions(), dims);
    EXPECT_EQ(ret->dimensions(), dims);
    EXPECT_EQ(n, 42);
    return absl::OkStatus();
  };

  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("n", 42);

  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/1);
  builder.AddBufferArg(memory0, PrimitiveType::U32, dims);
  builder.AddBufferRet(memory1, PrimitiveType::U32, dims);
  builder.AddAttributes(attrs.Build());

  // Keep call frame wrapped in optional to be able to destroy it and test that
  // updated call frame does not reference any destroyed memory.
  std::optional<CallFrame> call_frame(builder.Build());

  {  // Call `fn0` with an original call frame.
    auto handler = bind.To(fn0);
    auto status = Call(*handler, *call_frame);
    TF_ASSERT_OK(status);
  }

  {  // Call `fn1` with swapped buffers for argument and result.
    auto handler = bind.To(fn1);
    TF_ASSERT_OK_AND_ASSIGN(
        CallFrame updated_call_frame,
        std::move(call_frame)->CopyWithBuffers({memory1}, {memory0}));
    auto status = Call(*handler, updated_call_frame);
    TF_ASSERT_OK(status);
  }
}

TEST(FfiTest, DuplicateHandlerTraits) {
  static constexpr auto* noop = +[] { return absl::OkStatus(); };
  ZKX_FFI_DEFINE_HANDLER(NoOp1, noop, Ffi::Bind());
  ZKX_FFI_DEFINE_HANDLER(NoOp2, noop, Ffi::Bind(),
                         {Traits::kCmdBufferCompatible});
  ZKX_FFI_REGISTER_HANDLER(GetZkxFfiApi(), "duplicate-traits", "Host", NoOp1);
  auto status = TakeStatus(Ffi::RegisterStaticHandler(
      GetZkxFfiApi(), "duplicate-traits", "Host", NoOp2));
  EXPECT_TRUE(
      absl::StrContains(status.message(), "Duplicate FFI handler registration"))
      << "status.message():\n"
      << status.message() << "\n";
}

TEST(FfiTest, DuplicateHandlerAddress) {
  static constexpr auto* noop1 = +[] { return absl::OkStatus(); };
  static constexpr auto* noop2 = +[] { return absl::OkStatus(); };
  ZKX_FFI_DEFINE_HANDLER(NoOp1, noop1, Ffi::Bind());
  ZKX_FFI_DEFINE_HANDLER(NoOp2, noop2, Ffi::Bind());
  ZKX_FFI_REGISTER_HANDLER(GetZkxFfiApi(), "duplicate-address", "Host", NoOp1);
  auto status = TakeStatus(Ffi::RegisterStaticHandler(
      GetZkxFfiApi(), "duplicate-address", "Host", NoOp2));
  EXPECT_TRUE(
      absl::StrContains(status.message(), "Duplicate FFI handler registration"))
      << "status.message():\n"
      << status.message() << "\n";
}

TEST(FfiTest, AllowRegisterDuplicateWhenEqual) {
  static constexpr auto* noop = +[] { return absl::OkStatus(); };
  ZKX_FFI_DEFINE_HANDLER(NoOp, noop, Ffi::Bind());
  ZKX_FFI_REGISTER_HANDLER(GetZkxFfiApi(), "duplicate-when-equal", "Host",
                           NoOp);
  auto status = TakeStatus(Ffi::RegisterStaticHandler(
      GetZkxFfiApi(), "duplicate-when-equal", "Host", NoOp));
  TF_ASSERT_OK(status);
}

TEST(FfiTest, AsyncHandler) {
  Eigen::ThreadPool pool(2);
  Eigen::ThreadPoolDevice device(&pool, pool.NumThreads());

  int32_t value = 0;

  // Handler completes execution asynchronously on a given thread pool.
  auto fn = [&](const Eigen::ThreadPoolDevice* device) {
    auto async_value = tsl::MakeConstructedAsyncValueRef<tsl::Chain>();

    device->enqueueNoNotification([&, async_value]() mutable {
      value = 42;
      async_value.SetStateConcrete();
    });

    return async_value;
  };

  auto handler = Ffi::Bind().Ctx<IntraOpThreadPool>().To(fn);
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

TEST(FfiTest, Metadata) {
  static constexpr auto* noop = +[] { return absl::OkStatus(); };
  ZKX_FFI_DEFINE_HANDLER(handler, noop, Ffi::Bind());
  auto maybe_metadata = GetMetadata(handler);
  EXPECT_TRUE(maybe_metadata.ok());
  auto metadata = maybe_metadata.value();
  EXPECT_EQ(metadata.traits, 0);
  EXPECT_EQ(metadata.api_version.major_version, ZKX_FFI_API_MAJOR);
  EXPECT_EQ(metadata.api_version.minor_version, ZKX_FFI_API_MINOR);
}

TEST(FfiTest, MetadataTraits) {
  static constexpr auto* noop = +[] { return absl::OkStatus(); };
  ZKX_FFI_DEFINE_HANDLER(handler, noop, Ffi::Bind(),
                         {Traits::kCmdBufferCompatible});
  auto maybe_metadata = GetMetadata(handler);
  EXPECT_TRUE(maybe_metadata.ok());
  auto metadata = maybe_metadata.value();
  EXPECT_EQ(metadata.traits, ZKX_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE);
  EXPECT_EQ(metadata.api_version.major_version, ZKX_FFI_API_MAJOR);
  EXPECT_EQ(metadata.api_version.minor_version, ZKX_FFI_API_MINOR);
}

// Use opaque struct to define a platform stream type just like platform
// stream for GPU backend (e.g. `CUstream_st`  and `cudaStream_t`).
struct TestStreamSt;
using TestStream = TestStreamSt*;

template <>
struct CtxBinding<TestStream> {
  using Ctx = PlatformStream<TestStream>;
};

TEST(FfiTest, PlatformStream) {
  // We only check that it compiles.
  (void)Ffi::BindTo(+[](TestStream stream) { return absl::OkStatus(); });
}

}  // namespace zkx::ffi
