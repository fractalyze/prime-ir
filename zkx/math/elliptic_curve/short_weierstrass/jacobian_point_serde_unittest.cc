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

#include "zkx/math/elliptic_curve/short_weierstrass/jacobian_point_serde.h"

#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/test/sw_curve_config.h"
#include "zkx/base/buffer/vector_buffer.h"

namespace zkx {
namespace {

using namespace zk_dtypes::test;

TEST(JacobianPointTest, Serde) {
  JacobianPoint expected = JacobianPoint::Random();

  base::Uint8VectorBuffer write_buf;
  TF_ASSERT_OK(write_buf.Grow(base::EstimateSize(expected)));
  TF_ASSERT_OK(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  JacobianPoint value;
  TF_ASSERT_OK(write_buf.Read(&value));
  EXPECT_EQ(expected, value);
}

TEST(JacobianPointTest, JsonSerde) {
  rapidjson::Document doc;

  JacobianPoint expected = JacobianPoint::Random();
  rapidjson::Value json_value =
      base::JsonSerde<JacobianPoint>::From(expected, doc.GetAllocator());
  EXPECT_TRUE(json_value.IsObject());
  EXPECT_EQ(Fq::FromUnchecked(json_value["x"].GetInt()), expected.x());
  EXPECT_EQ(Fq::FromUnchecked(json_value["y"].GetInt()), expected.y());
  EXPECT_EQ(Fq::FromUnchecked(json_value["z"].GetInt()), expected.z());

  TF_ASSERT_OK_AND_ASSIGN(JacobianPoint value,
                          base::JsonSerde<JacobianPoint>::To(json_value, ""));
  EXPECT_EQ(expected, value);
}

}  // namespace
}  // namespace zkx
