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

#ifndef ZKX_MATH_BASE_BIG_INT_SERDE_H_
#define ZKX_MATH_BASE_BIG_INT_SERDE_H_

#include <stddef.h>
#include <stdint.h>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zk_dtypes/include/big_int.h"
#include "zkx/base/buffer/serde.h"
#include "zkx/base/json/json_serde.h"

namespace zkx::base {

template <size_t N>
class Serde<zk_dtypes::BigInt<N>> {
 public:
  static absl::Status WriteTo(const zk_dtypes::BigInt<N>& bigint,
                              Buffer* buffer, Endian endian) {
    switch (endian) {
      case Endian::kNative:
        return buffer->Write(reinterpret_cast<const uint8_t*>(bigint.limbs()),
                             sizeof(uint64_t) * N);
      case Endian::kBig: {
        return buffer->Write(bigint.ToBytesBE());
      }
      case Endian::kLittle: {
        return buffer->Write(bigint.ToBytesLE());
      }
    }
    ABSL_UNREACHABLE();
    return absl::InternalError("Corrupted endian");
  }

  static absl::Status ReadFrom(const ReadOnlyBuffer& buffer,
                               zk_dtypes::BigInt<N>* bigint, Endian endian) {
    switch (endian) {
      case Endian::kNative:
        TF_RETURN_IF_ERROR(buffer.Read(
            reinterpret_cast<uint8_t*>(bigint->limbs()), sizeof(uint64_t) * N));
        return absl::OkStatus();
      case Endian::kBig: {
        uint8_t bytes[N * sizeof(uint64_t)];
        TF_RETURN_IF_ERROR(buffer.Read(bytes));
        *bigint = zk_dtypes::BigInt<N>::FromBytesBE(bytes);
        return absl::OkStatus();
      }
      case Endian::kLittle: {
        uint8_t bytes[N * sizeof(uint64_t)];
        TF_RETURN_IF_ERROR(buffer.Read(bytes));
        *bigint = zk_dtypes::BigInt<N>::FromBytesLE(bytes);
        return absl::OkStatus();
      }
    }
    ABSL_UNREACHABLE();
    return absl::InternalError("Corrupted endian");
  }

  static size_t EstimateSize(const zk_dtypes::BigInt<N>& bigint) {
    return sizeof(uint64_t) * N;
  }
};

template <size_t N>
class JsonSerde<zk_dtypes::BigInt<N>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(const zk_dtypes::BigInt<N>& value,
                               Allocator& allocator) {
    if (value < zk_dtypes::BigInt<N>(std::numeric_limits<uint64_t>::max())) {
      return rapidjson::Value(value.limbs()[0]);
    } else {
      return rapidjson::Value(value.ToString(), allocator);
    }
  }

  static absl::StatusOr<zk_dtypes::BigInt<N>> To(
      const rapidjson::Value& json_value, std::string_view key) {
    if (json_value.IsUint64()) {
      return zk_dtypes::BigInt<N>(json_value.GetUint64());
    } else if (json_value.IsString()) {
      TF_ASSIGN_OR_RETURN(std::string value,
                          JsonSerde<std::string>::To(json_value, key));
      return zk_dtypes::BigInt<N>::FromDecString(value);
    } else {
      return absl::InvalidArgumentError(
          RapidJsonMismatchedTypeError(key, "string", json_value));
    }
  }
};

}  // namespace zkx::base

#endif  // ZKX_MATH_BASE_BIG_INT_SERDE_H_
