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

#include "zkx/math/elliptic_curve/short_weierstrass/affine_point_serde.h"

#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/test/sw_curve_config.h"
#include "zkx/base/auto_reset.h"
#include "zkx/base/buffer/vector_buffer.h"

namespace zkx::math {
namespace {

using namespace zk_dtypes::test;

TEST(AffinePointTest, Serde) {
  AffinePoint expected = AffinePoint::Random();

  base::Uint8VectorBuffer write_buf;
  TF_ASSERT_OK(write_buf.Grow(base::EstimateSize(expected)));
  TF_ASSERT_OK(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  AffinePoint value;
  TF_ASSERT_OK(write_buf.Read(&value));
  EXPECT_EQ(expected, value);
}

TEST(AffinePointTest, SerdeWithGnark) {
  for (size_t i = 0; i < 2; ++i) {
    auto mode = static_cast<AffinePointSerdeMode>(i);
    SCOPED_TRACE(absl::Substitute("mode: $0", i));
    base::AutoReset<AffinePointSerdeMode> auto_reset(
        &base::Serde<AffinePoint>::s_mode, mode);
    base::AutoReset<bool> auto_reset2(&base::Serde<Fq>::s_is_in_montgomery,
                                      false);

    AffinePoint expected = AffinePoint::Random();

    base::Uint8VectorBuffer write_buf;
    write_buf.set_endian(base::Endian::kBig);
    TF_ASSERT_OK(write_buf.Grow(base::EstimateSize(expected)));
    TF_ASSERT_OK(write_buf.Write(expected));
    ASSERT_TRUE(write_buf.Done());

    write_buf.set_buffer_offset(0);

    AffinePoint value;
    TF_ASSERT_OK(write_buf.Read(&value));
    EXPECT_EQ(expected, value);
  }
}

TEST(AffinePointTest, JsonSerde) {
  rapidjson::Document doc;

  AffinePoint expected = AffinePoint::Random();
  rapidjson::Value json_value =
      base::JsonSerde<AffinePoint>::From(expected, doc.GetAllocator());
  EXPECT_TRUE(json_value.IsObject());
  EXPECT_EQ(Fq::FromUnchecked(json_value["x"].GetInt()), expected.x());
  EXPECT_EQ(Fq::FromUnchecked(json_value["y"].GetInt()), expected.y());

  TF_ASSERT_OK_AND_ASSIGN(AffinePoint value,
                          base::JsonSerde<AffinePoint>::To(json_value, ""));
  EXPECT_EQ(expected, value);
}

}  // namespace
}  // namespace zkx::math
