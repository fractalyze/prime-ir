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

#include "zkx/math/field/extension_field_serde.h"

#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "zk_dtypes/include/all_types.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/test/sw_curve_config.h"
#include "zkx/base/buffer/vector_buffer.h"

namespace zkx {
namespace {

template <typename T>
class ExtensionFieldTypedTest : public testing::Test {};

using ExtensionFieldTypes = testing::Types<
#define EXTENSION_FIELD_TYPE(ActualType, ...) ActualType,
    ZK_DTYPES_ALL_EXT_FIELD_TYPE_LIST(EXTENSION_FIELD_TYPE)
#undef EXTENSION_FIELD_TYPE
        zk_dtypes::test::FqX2,
    zk_dtypes::test::FqX2Std>;
TYPED_TEST_SUITE(ExtensionFieldTypedTest, ExtensionFieldTypes);

TYPED_TEST(ExtensionFieldTypedTest, Serde) {
  using ExtF = TypeParam;

  ExtF expected = ExtF::Random();

  base::Uint8VectorBuffer write_buf;
  TF_ASSERT_OK(write_buf.Grow(base::EstimateSize(expected)));
  TF_ASSERT_OK(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  ExtF value;
  TF_ASSERT_OK(write_buf.Read(&value));
  EXPECT_EQ(expected, value);
}

TYPED_TEST(ExtensionFieldTypedTest, JsonSerde) {
  using ExtF = TypeParam;

  rapidjson::Document doc;

  ExtF expected = ExtF::Random();
  rapidjson::Value json_value =
      base::JsonSerde<ExtF>::From(expected, doc.GetAllocator());
  TF_ASSERT_OK_AND_ASSIGN(ExtF value,
                          base::JsonSerde<ExtF>::To(json_value, ""));
  EXPECT_EQ(expected, value);
}

}  // namespace
}  // namespace zkx
