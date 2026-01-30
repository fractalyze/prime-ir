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

#ifndef ZKX_MATH_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_AFFINE_POINT_SERDE_H_
#define ZKX_MATH_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_AFFINE_POINT_SERDE_H_

#include <string>
#include <type_traits>

#include "absl/base/internal/endian.h"
#include "absl/base/optimization.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/affine_point.h"
#include "zk_dtypes/include/field/finite_field_traits.h"
#include "zk_dtypes/include/geometry/curve_type.h"
#include "zkx/base/buffer/serde.h"
#include "zkx/base/json/json_serde.h"
#include "zkx/math/field/extension_field_serde.h"
#include "zkx/math/field/prime_field_serde.h"

namespace zkx {
namespace math {

enum class AffinePointSerdeMode {
  kNone,
  // For Gnark's default format (i.e., data saved using Gnark's WriteTo())
  kGnarkDefault,
  // For Gnark's raw format (i.e., data saved using Gnark's WriteRawTo())
  kGnarkRaw,
};

// See
// https://github.com/Consensys/gnark-crypto/blob/43897fd/ecc/bn254/marshal.go#L21-L31
enum class GnarkCompressedFlag {
  kMask = 0b11 << 6,
  kUncompressed = 0b00 << 6,
  kCompressedSmallest = 0b10 << 6,
  kCompressedLargest = 0b11 << 6,
  kCompressedInfinity = 0b01 << 6,
};

}  // namespace math

namespace base {

template <typename Curve>
class Serde<zk_dtypes::AffinePoint<
    Curve, std::enable_if_t<Curve::kType ==
                            zk_dtypes::CurveType::kShortWeierstrass>>> {
 public:
  using BaseField = typename zk_dtypes::AffinePoint<Curve>::BaseField;
  using BasePrimeField =
      typename zk_dtypes::FiniteFieldTraits<BaseField>::BasePrimeField;
  using UnderlyingType = typename BasePrimeField::UnderlyingType;

  static math::AffinePointSerdeMode s_mode;

  static absl::Status WriteTo(const zk_dtypes::AffinePoint<Curve>& point,
                              Buffer* buffer, Endian endian) {
    switch (s_mode) {
      case math::AffinePointSerdeMode::kNone:
        return buffer->WriteMany(point.x(), point.y());
      case math::AffinePointSerdeMode::kGnarkDefault:
      case math::AffinePointSerdeMode::kGnarkRaw: {
        if (endian != Endian::kBig) {
          return absl::InvalidArgumentError(
              "Invalid endian: GnarkDefault and GnarkRaw modes require "
              "BigEndian input");
        }
        if (Serde<BasePrimeField>::s_is_in_montgomery) {
          return absl::InvalidArgumentError(
              "Invalid format: Input must be in non-Montgomery form");
        }
        if (BasePrimeField::kBitWidth - BasePrimeField::Config::kModulusBits <
            2) {
          return absl::InvalidArgumentError(
              "Invalid format: BasePrimeField::kBitWidth - "
              "BasePrimeField::Config::kModulusBits must be at least 2");
        }

        // See
        // https://github.com/Consensys/gnark-crypto/blob/43897fd/ecc/bn254/marshal.go#L790-L822.
        math::GnarkCompressedFlag flag;
        size_t old_buffer_offset = buffer->buffer_offset();
        if (point.IsZero()) {
          if (s_mode == math::AffinePointSerdeMode::kGnarkDefault) {
            uint8_t zeros[BaseField::kByteWidth] = {0};
            TF_RETURN_IF_ERROR(buffer->Write(zeros));
            flag = math::GnarkCompressedFlag::kCompressedInfinity;
          } else {
            uint8_t zeros[BaseField::kByteWidth * 2] = {0};
            TF_RETURN_IF_ERROR(buffer->Write(zeros));
            flag = math::GnarkCompressedFlag::kUncompressed;
          }
        } else {
          if (s_mode == math::AffinePointSerdeMode::kGnarkDefault) {
            if (point.y().LexicographicallyLargest()) {
              flag = math::GnarkCompressedFlag::kCompressedLargest;
            } else {
              flag = math::GnarkCompressedFlag::kCompressedSmallest;
            }
          } else {
            flag = math::GnarkCompressedFlag::kUncompressed;
          }
          if constexpr (BaseField::ExtensionDegree() == 1) {
            TF_RETURN_IF_ERROR(buffer->Write(point.x()));
            if (s_mode != math::AffinePointSerdeMode::kGnarkDefault) {
              TF_RETURN_IF_ERROR(buffer->Write(point.y()));
            }
          } else if constexpr (BaseField::ExtensionDegree() == 2) {
            TF_RETURN_IF_ERROR(buffer->Write(point.x()[1]));
            TF_RETURN_IF_ERROR(buffer->Write(point.x()[0]));
            if (s_mode != math::AffinePointSerdeMode::kGnarkDefault) {
              TF_RETURN_IF_ERROR(buffer->Write(point.y()[1]));
              TF_RETURN_IF_ERROR(buffer->Write(point.y()[0]));
            }
          } else {
            return absl::InvalidArgumentError(
                "Invalid extension degree: GnarkDefault and GnarkRaw modes "
                "only support extension degree 1 or 2");
          }
        }
        reinterpret_cast<uint8_t*>(buffer->buffer())[old_buffer_offset] |=
            static_cast<uint8_t>(flag);
        return absl::OkStatus();
      }
    }
    return absl::InvalidArgumentError("Unsupported AffinePointSerdeMode value");
  }

  static absl::Status ReadFrom(const ReadOnlyBuffer& buffer,
                               zk_dtypes::AffinePoint<Curve>* point,
                               Endian endian) {
    switch (s_mode) {
      case math::AffinePointSerdeMode::kNone: {
        BaseField x, y;
        TF_RETURN_IF_ERROR(buffer.ReadMany(&x, &y));
        *point = zk_dtypes::AffinePoint<Curve>(x, y);
        return absl::OkStatus();
      }
      case math::AffinePointSerdeMode::kGnarkDefault:
      case math::AffinePointSerdeMode::kGnarkRaw: {
        if (endian != Endian::kBig) {
          return absl::InvalidArgumentError(
              "Invalid endian: GnarkDefault and GnarkRaw modes require "
              "BigEndian input");
        }
        if (Serde<BasePrimeField>::s_is_in_montgomery) {
          return absl::InvalidArgumentError(
              "Invalid format: Input must be in non-Montgomery form");
        }
        if (BasePrimeField::kBitWidth - BasePrimeField::Config::kModulusBits <
            2) {
          return absl::InvalidArgumentError(
              "Invalid format: BasePrimeField::kBitWidth - "
              "BasePrimeField::Config::kModulusBits must be at least 2");
        }
        // See
        // https://github.com/Consensys/gnark-crypto/blob/43897fd/ecc/bn254/marshal.go#L790-L822.
        std::array<uint8_t, BaseField::kByteWidth> bytes;
        TF_RETURN_IF_ERROR(buffer.Read(&bytes));
        auto flag = static_cast<math::GnarkCompressedFlag>(
            bytes[0] & static_cast<uint64_t>(math::GnarkCompressedFlag::kMask));
        bytes[0] &= 0x3F;
        switch (flag) {
          case math::GnarkCompressedFlag::kUncompressed: {
            std::array<uint8_t, BaseField::kByteWidth> bytes2;
            TF_RETURN_IF_ERROR(buffer.Read(&bytes2));
            BaseField x, y;
            if constexpr (BaseField::ExtensionDegree() == 1) {
              x = BasePrimeField(ReadFromBytesBE<UnderlyingType>(bytes));
              y = BasePrimeField(ReadFromBytesBE<UnderlyingType>(bytes2));
            } else if constexpr (BaseField::ExtensionDegree() == 2) {
              x = {
                  BasePrimeField(
                      UnderlyingType::FromBytesBE(absl::Span<const uint8_t>(
                          &bytes[BasePrimeField::kByteWidth],
                          BasePrimeField::kByteWidth))),
                  BasePrimeField(
                      UnderlyingType::FromBytesBE(absl::Span<const uint8_t>(
                          &bytes[0], BasePrimeField::kByteWidth))),
              };
              y = {
                  BasePrimeField(
                      UnderlyingType::FromBytesBE(absl::Span<const uint8_t>(
                          &bytes2[BasePrimeField::kByteWidth],
                          BasePrimeField::kByteWidth))),
                  BasePrimeField(
                      UnderlyingType::FromBytesBE(absl::Span<const uint8_t>(
                          &bytes2[0], BasePrimeField::kByteWidth))),
              };
            } else {
              return absl::InvalidArgumentError(
                  "Invalid extension degree: GnarkDefault and GnarkRaw modes "
                  "only support extension degree 1 or 2");
            }
            *point = zk_dtypes::AffinePoint<Curve>(x, y);
            return absl::OkStatus();
          }
          case math::GnarkCompressedFlag::kCompressedInfinity: {
            *point = zk_dtypes::AffinePoint<Curve>::Zero();
            return absl::OkStatus();
          }
          case math::GnarkCompressedFlag::kCompressedSmallest:
          case math::GnarkCompressedFlag::kCompressedLargest: {
            BaseField x;
            if constexpr (BaseField::ExtensionDegree() == 1) {
              x = BasePrimeField(ReadFromBytesBE<UnderlyingType>(bytes));
            } else if constexpr (BaseField::ExtensionDegree() == 2) {
              x = {
                  BasePrimeField(
                      UnderlyingType::FromBytesBE(absl::Span<const uint8_t>(
                          &bytes[BasePrimeField::kByteWidth],
                          BasePrimeField::kByteWidth))),
                  BasePrimeField(
                      UnderlyingType::FromBytesBE(absl::Span<const uint8_t>(
                          &bytes[0], BasePrimeField::kByteWidth))),
              };
            } else {
              return absl::InvalidArgumentError(
                  "Invalid extension degree: GnarkDefault and GnarkRaw modes "
                  "only support extension degree 1 or 2");
            }
            TF_ASSIGN_OR_RETURN(*point,
                                zk_dtypes::AffinePoint<Curve>::CreateFromX(x));
            if (point->y().LexicographicallyLargest()) {
              if (flag == math::GnarkCompressedFlag::kCompressedSmallest) {
                *point = -*point;
              }
            } else {
              if (flag == math::GnarkCompressedFlag::kCompressedLargest) {
                *point = -*point;
              }
            }
            return absl::OkStatus();
          }
        }
        return absl::InvalidArgumentError(
            "Unsupported GnarkCompressedFlag value");
      }
    }
    return absl::InvalidArgumentError("Unsupported AffinePointSerdeMode value");
  }

  static size_t EstimateSize(const zk_dtypes::AffinePoint<Curve>& point) {
    switch (s_mode) {
      case math::AffinePointSerdeMode::kNone:
        return base::EstimateSize(point.x(), point.y());
      case math::AffinePointSerdeMode::kGnarkDefault:
        return base::EstimateSize(point.x());
      case math::AffinePointSerdeMode::kGnarkRaw: {
        if (point.IsZero()) {
          return base::EstimateSize(point.x());
        } else {
          return base::EstimateSize(point.x(), point.y());
        }
      }
    }
    ABSL_UNREACHABLE();
    return 0;
  }

  template <typename T, typename BytesContainer>
  static T ReadFromBytesBE(const BytesContainer& bytes) {
    using UnderlyingType = typename BasePrimeField::UnderlyingType;

    if constexpr (std::is_same_v<UnderlyingType, uint64_t>) {
      return BasePrimeField(absl::big_endian::Load64(bytes.data()));
    } else if constexpr (std::is_same_v<UnderlyingType, uint32_t>) {
      return BasePrimeField(absl::big_endian::Load32(bytes.data()));
    } else if constexpr (std::is_same_v<UnderlyingType, uint16_t>) {
      return BasePrimeField(absl::big_endian::Load16(bytes.data()));
    } else if constexpr (std::is_same_v<UnderlyingType, uint8_t>) {
      return bytes[0];
    } else {
      return UnderlyingType::FromBytesBE(bytes);
    }
  }
};

// static
template <typename Curve>
math::AffinePointSerdeMode Serde<zk_dtypes::AffinePoint<
    Curve, std::enable_if_t<Curve::kType ==
                            zk_dtypes::CurveType::kShortWeierstrass>>>::s_mode =
    math::AffinePointSerdeMode::kNone;

template <typename Curve>
class JsonSerde<zk_dtypes::AffinePoint<
    Curve, std::enable_if_t<Curve::kType ==
                            zk_dtypes::CurveType::kShortWeierstrass>>> {
 public:
  using Field = typename zk_dtypes::AffinePoint<Curve>::BaseField;

  template <typename Allocator>
  static rapidjson::Value From(const zk_dtypes::AffinePoint<Curve>& value,
                               Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    AddJsonElement(object, "x", value.x(), allocator);
    AddJsonElement(object, "y", value.y(), allocator);
    return object;
  }

  static absl::StatusOr<zk_dtypes::AffinePoint<Curve>> To(
      const rapidjson::Value& json_value, std::string_view key) {
    TF_ASSIGN_OR_RETURN(Field x, ParseJsonElement<Field>(json_value, "x"));
    TF_ASSIGN_OR_RETURN(Field y, ParseJsonElement<Field>(json_value, "y"));
    return zk_dtypes::AffinePoint<Curve>(x, y);
  }
};

}  // namespace base
}  // namespace zkx

#endif  // ZKX_MATH_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_AFFINE_POINT_SERDE_H_
