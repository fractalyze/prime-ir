/* Copyright 2025 The ZKX Authors.

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

#include "zkx/math/base/big_int_serde.h"

#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "zk_dtypes/include/big_int.h"
#include "zkx/base/buffer/vector_buffer.h"

namespace zkx {
namespace {

using namespace zk_dtypes;

TEST(BigIntTest, Serde) {
  BigInt<2> expected = BigInt<2>::Random();

  base::Uint8VectorBuffer write_buf;
  TF_ASSERT_OK(write_buf.Grow(base::EstimateSize(expected)));
  TF_ASSERT_OK(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  BigInt<2> value;
  TF_ASSERT_OK(write_buf.Read(&value));
  EXPECT_EQ(value, expected);
}

TEST(BigIntTest, JsonSerde) {
  rapidjson::Document doc;

  // Test with uint64_t value
  BigInt<2> expected(12345);
  rapidjson::Value json_value =
      base::JsonSerde<BigInt<2>>::From(expected, doc.GetAllocator());
  EXPECT_TRUE(json_value.IsUint64());
  EXPECT_EQ(json_value.GetUint64(), 12345);

  TF_ASSERT_OK_AND_ASSIGN(BigInt<2> actual,
                          base::JsonSerde<BigInt<2>>::To(json_value, ""));
  EXPECT_EQ(actual, expected);

  // Test with large value that needs string representation
  expected = BigInt<2>::FromDecString("36893488147419103232").value();
  json_value = base::JsonSerde<BigInt<2>>::From(expected, doc.GetAllocator());
  EXPECT_TRUE(json_value.IsString());
  EXPECT_STREQ(json_value.GetString(), "36893488147419103232");

  TF_ASSERT_OK_AND_ASSIGN(actual,
                          base::JsonSerde<BigInt<2>>::To(json_value, ""));
  EXPECT_EQ(actual, expected);
}

}  // namespace
}  // namespace zkx
