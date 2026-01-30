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

#ifndef ZKX_MATH_FIELD_PRIME_FIELD_SERDE_H_
#define ZKX_MATH_FIELD_PRIME_FIELD_SERDE_H_

#include <stddef.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zk_dtypes/include/field/prime_field.h"
#include "zkx/base/buffer/serde.h"
#include "zkx/base/json/json_serde.h"
#include "zkx/math/base/big_int_serde.h"

namespace zkx::base {

template <typename Config>
class Serde<zk_dtypes::PrimeField<Config>> {
 public:
  using UnderlyingType = typename zk_dtypes::PrimeField<Config>::UnderlyingType;

  static bool s_is_in_montgomery;

  static absl::Status WriteTo(const zk_dtypes::PrimeField<Config>& prime_field,
                              Buffer* buffer, Endian) {
    if constexpr (Config::kUseMontgomery) {
      if (s_is_in_montgomery) {
        return buffer->Write(prime_field.value());
      } else {
        return buffer->Write(prime_field.MontReduce().value());
      }
    } else {
      return buffer->Write(prime_field.value());
    }
  }

  static absl::Status ReadFrom(const ReadOnlyBuffer& buffer,
                               zk_dtypes::PrimeField<Config>* prime_field,
                               Endian) {
    UnderlyingType v;
    TF_RETURN_IF_ERROR(buffer.Read(&v));
    if constexpr (Config::kUseMontgomery) {
      if (s_is_in_montgomery) {
        *prime_field = zk_dtypes::PrimeField<Config>::FromUnchecked(v);
      } else {
        *prime_field = zk_dtypes::PrimeField<Config>(v);
      }
    } else {
      *prime_field = zk_dtypes::PrimeField<Config>(v);
    }
    return absl::OkStatus();
  }

  static size_t EstimateSize(const zk_dtypes::PrimeField<Config>& prime_field) {
    return zk_dtypes::PrimeField<Config>::kByteWidth;
  }
};

// static
template <typename Config>
bool Serde<zk_dtypes::PrimeField<Config>>::s_is_in_montgomery = true;

template <typename Config>
class JsonSerde<zk_dtypes::PrimeField<Config>> {
 public:
  using UnderlyingType = typename zk_dtypes::PrimeField<Config>::UnderlyingType;

  static bool s_is_in_montgomery;

  template <typename Allocator>
  static rapidjson::Value From(const zk_dtypes::PrimeField<Config>& value,
                               Allocator& allocator) {
    if constexpr (Config::kUseMontgomery) {
      if (s_is_in_montgomery) {
        return JsonSerde<UnderlyingType>::From(value.value(), allocator);
      } else {
        return JsonSerde<UnderlyingType>::From(value.MontReduce().value(),
                                               allocator);
      }
    } else {
      return JsonSerde<UnderlyingType>::From(value.value(), allocator);
    }
  }

  static absl::StatusOr<zk_dtypes::PrimeField<Config>> To(
      const rapidjson::Value& json_value, std::string_view key) {
    TF_ASSIGN_OR_RETURN(UnderlyingType v,
                        JsonSerde<UnderlyingType>::To(json_value, key));

    if constexpr (Config::kUseMontgomery) {
      if (s_is_in_montgomery) {
        return zk_dtypes::PrimeField<Config>::FromUnchecked(v);
      } else {
        return zk_dtypes::PrimeField<Config>(v);
      }
    } else {
      return zk_dtypes::PrimeField<Config>(v);
    }
  }
};

// static
template <typename Config>
bool JsonSerde<zk_dtypes::PrimeField<Config>>::s_is_in_montgomery = true;

}  // namespace zkx::base

#endif  // ZKX_MATH_FIELD_PRIME_FIELD_SERDE_H_
