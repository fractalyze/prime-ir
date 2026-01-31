/* Copyright 2024 The OpenXLA Authors.
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

#include "zkx/ffi/call_frame.h"

#include <optional>

#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"

namespace zkx::ffi {

TEST(CallFrameTest, UpdateCallFrame) {
  se::DeviceMemoryBase mem0(reinterpret_cast<void*>(0x12345678), 1024);
  se::DeviceMemoryBase mem1(reinterpret_cast<void*>(0x87654321), 1024);

  std::vector<int64_t> dims = {1, 2, 3, 4};

  CallFrameBuilder::AttributesBuilder attrs_builder;
  attrs_builder.Insert("attr1", "value1");
  attrs_builder.Insert("attr2", "value2");

  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/1);
  // Use U32 since ZKX doesn't have F32 PrimitiveType (no floating-point types)
  builder.AddBufferArg(mem0, PrimitiveType::U32, dims);
  builder.AddBufferRet(mem1, PrimitiveType::U32, dims);
  builder.AddAttributes(attrs_builder.Build());

  // Keep call frame wrapped in optional to be able to destroy it and test that
  // updated call frame does not reference any destroyed memory.
  std::optional<CallFrame> call_frame = builder.Build();

  {  // Construct ZKX_FFI_CallFrame from the original call frame.
    ZKX_FFI_CallFrame ffi_call_frame = call_frame->Build(
        /*api=*/nullptr, /*ctx=*/nullptr, ZKX_FFI_ExecutionStage_EXECUTE);

    EXPECT_EQ(ffi_call_frame.args.size, 1);
    EXPECT_EQ(ffi_call_frame.args.types[0], ZKX_FFI_ArgType_BUFFER);
    EXPECT_EQ(static_cast<ZKX_FFI_Buffer*>(ffi_call_frame.args.args[0])->data,
              mem0.opaque());

    EXPECT_EQ(ffi_call_frame.rets.size, 1);
    EXPECT_EQ(ffi_call_frame.rets.types[0], ZKX_FFI_ArgType_BUFFER);
    EXPECT_EQ(static_cast<ZKX_FFI_Buffer*>(ffi_call_frame.rets.rets[0])->data,
              mem1.opaque());

    EXPECT_EQ(ffi_call_frame.attrs.size, 2);
  }

  CallFrame updated_call_frame =
      std::move(call_frame)->CopyWithBuffers({mem1}, {mem0}).value();

  {  // Construct ZKX_FFI_CallFrame from the updated call frame.
    ZKX_FFI_CallFrame ffi_call_frame = updated_call_frame.Build(
        /*api=*/nullptr, /*ctx=*/nullptr, ZKX_FFI_ExecutionStage_EXECUTE);

    EXPECT_EQ(ffi_call_frame.args.size, 1);
    EXPECT_EQ(ffi_call_frame.args.types[0], ZKX_FFI_ArgType_BUFFER);
    EXPECT_EQ(static_cast<ZKX_FFI_Buffer*>(ffi_call_frame.args.args[0])->data,
              mem1.opaque());

    EXPECT_EQ(ffi_call_frame.rets.size, 1);
    EXPECT_EQ(ffi_call_frame.rets.types[0], ZKX_FFI_ArgType_BUFFER);
    EXPECT_EQ(static_cast<ZKX_FFI_Buffer*>(ffi_call_frame.rets.rets[0])->data,
              mem0.opaque());

    EXPECT_EQ(ffi_call_frame.attrs.size, 2);
  }

  TF_ASSERT_OK(updated_call_frame.UpdateWithBuffers({mem0}, {mem1}));

  {  // Construct ZKX_FFI_CallFrame from the call frame updated in place.
    ZKX_FFI_CallFrame ffi_call_frame = updated_call_frame.Build(
        /*api=*/nullptr, /*ctx=*/nullptr, ZKX_FFI_ExecutionStage_EXECUTE);

    EXPECT_EQ(ffi_call_frame.args.size, 1);
    EXPECT_EQ(ffi_call_frame.args.types[0], ZKX_FFI_ArgType_BUFFER);
    EXPECT_EQ(static_cast<ZKX_FFI_Buffer*>(ffi_call_frame.args.args[0])->data,
              mem0.opaque());

    EXPECT_EQ(ffi_call_frame.rets.size, 1);
    EXPECT_EQ(ffi_call_frame.rets.types[0], ZKX_FFI_ArgType_BUFFER);
    EXPECT_EQ(static_cast<ZKX_FFI_Buffer*>(ffi_call_frame.rets.rets[0])->data,
              mem1.opaque());

    EXPECT_EQ(ffi_call_frame.attrs.size, 2);
  }
}

}  // namespace zkx::ffi
