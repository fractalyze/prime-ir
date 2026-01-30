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

#ifndef ZKX_MATH_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_JACOBIAN_POINT_SERDE_H_
#define ZKX_MATH_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_JACOBIAN_POINT_SERDE_H_

#include <string>
#include <type_traits>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/point_xyzz.h"
#include "zk_dtypes/include/geometry/curve_type.h"
#include "zkx/base/buffer/serde.h"
#include "zkx/base/json/json_serde.h"
#include "zkx/math/field/extension_field_serde.h"
#include "zkx/math/field/prime_field_serde.h"

namespace zkx::base {

template <typename Curve>
class Serde<zk_dtypes::JacobianPoint<
    Curve, std::enable_if_t<Curve::kType ==
                            zk_dtypes::CurveType::kShortWeierstrass>>> {
 public:
  static absl::Status WriteTo(const zk_dtypes::JacobianPoint<Curve>& point,
                              Buffer* buffer, Endian) {
    return buffer->WriteMany(point.x(), point.y(), point.z());
  }

  static absl::Status ReadFrom(const ReadOnlyBuffer& buffer,
                               zk_dtypes::JacobianPoint<Curve>* point, Endian) {
    using BaseField = typename zk_dtypes::JacobianPoint<Curve>::BaseField;
    BaseField x, y, z;
    TF_RETURN_IF_ERROR(buffer.ReadMany(&x, &y, &z));
    *point = zk_dtypes::JacobianPoint<Curve>(x, y, z);
    return absl::OkStatus();
  }

  static size_t EstimateSize(const zk_dtypes::JacobianPoint<Curve>& point) {
    return base::EstimateSize(point.x(), point.y(), point.z());
  }
};

template <typename Curve>
class JsonSerde<zk_dtypes::JacobianPoint<
    Curve, std::enable_if_t<Curve::kType ==
                            zk_dtypes::CurveType::kShortWeierstrass>>> {
 public:
  using Field = typename zk_dtypes::JacobianPoint<Curve>::BaseField;

  template <typename Allocator>
  static rapidjson::Value From(const zk_dtypes::JacobianPoint<Curve>& value,
                               Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    AddJsonElement(object, "x", value.x(), allocator);
    AddJsonElement(object, "y", value.y(), allocator);
    AddJsonElement(object, "z", value.z(), allocator);
    return object;
  }

  static absl::StatusOr<zk_dtypes::JacobianPoint<Curve>> To(
      const rapidjson::Value& json_value, std::string_view key) {
    TF_ASSIGN_OR_RETURN(Field x, ParseJsonElement<Field>(json_value, "x"));
    TF_ASSIGN_OR_RETURN(Field y, ParseJsonElement<Field>(json_value, "y"));
    TF_ASSIGN_OR_RETURN(Field z, ParseJsonElement<Field>(json_value, "z"));
    return zk_dtypes::JacobianPoint<Curve>(x, y, z);
  }
};

}  // namespace zkx::base

#endif  // ZKX_MATH_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_JACOBIAN_POINT_SERDE_H_
