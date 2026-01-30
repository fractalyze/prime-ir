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

#ifndef ZKX_MATH_FIELD_EXTENSION_FIELD_SERDE_H_
#define ZKX_MATH_FIELD_EXTENSION_FIELD_SERDE_H_

#include <stddef.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zk_dtypes/include/field/extension_field.h"
#include "zk_dtypes/include/field/finite_field.h"
#include "zkx/base/buffer/serde.h"
#include "zkx/base/json/json_serde.h"
#include "zkx/math/field/prime_field_serde.h"

namespace zkx::base {

template <typename Config>
class Serde<zk_dtypes::ExtensionField<Config>> {
 public:
  constexpr static size_t N = zk_dtypes::ExtensionField<Config>::N;

  static absl::Status WriteTo(
      const zk_dtypes::ExtensionField<Config>& ext_field, Buffer* buffer,
      Endian) {
    return buffer->Write(ext_field.values());
  }

  static absl::Status ReadFrom(const ReadOnlyBuffer& buffer,
                               zk_dtypes::ExtensionField<Config>* ext_field,
                               Endian) {
    for (size_t i = 0; i < N; ++i) {
      TF_RETURN_IF_ERROR(buffer.Read(&ext_field->values()[i]));
    }
    return absl::OkStatus();
  }

  static size_t EstimateSize(
      const zk_dtypes::ExtensionField<Config>& ext_field) {
    return base::EstimateSize(ext_field.values());
  }
};

template <typename Config>
class JsonSerde<zk_dtypes::ExtensionField<Config>> {
 public:
  using BaseField = typename zk_dtypes::ExtensionField<Config>::BaseField;
  constexpr static size_t N = zk_dtypes::ExtensionField<Config>::N;

  template <typename Allocator>
  static rapidjson::Value From(const zk_dtypes::ExtensionField<Config>& value,
                               Allocator& allocator) {
    return JsonSerde<std::array<BaseField, N>>::From(value.values(), allocator);
  }

  static absl::StatusOr<zk_dtypes::ExtensionField<Config>> To(
      const rapidjson::Value& json_value, std::string_view key) {
    absl::StatusOr<std::array<BaseField, N>> values =
        JsonSerde<std::array<BaseField, N>>::To(json_value, key);
    if (values.ok()) {
      return zk_dtypes::ExtensionField<Config>(values.value());
    }
    return values.status();
  }
};

}  // namespace zkx::base

#endif  // ZKX_MATH_FIELD_EXTENSION_FIELD_SERDE_H_
