/* Copyright 2017 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

// Utilities for dealing with ZKX primitive types.

#ifndef ZKX_PRIMITIVE_UTIL_H_
#define ZKX_PRIMITIVE_UTIL_H_

#include <stdint.h>

#include <array>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"

#include "xla/tsl/lib/math/math_util.h"
#include "zk_dtypes/include/all_types.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/fr.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g1.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g2.h"
#include "zk_dtypes/include/field/babybear/babybear.h"
#include "zk_dtypes/include/field/extension_field.h"
#include "zk_dtypes/include/field/field.h"
#include "zk_dtypes/include/field/goldilocks/goldilocks.h"
#include "zk_dtypes/include/field/koalabear/koalabear.h"
#include "zk_dtypes/include/field/mersenne31/mersenne31.h"
#include "zkx/types.h"
#include "zkx/util.h"
#include "zkx/zkx_data.pb.h"

namespace zkx::primitive_util {

// Returns the ZKX primitive type (eg, U8) corresponding to the given
// template parameter native type (eg, uint8_t).
template <typename NativeT>
constexpr PrimitiveType NativeToPrimitiveType() {
  // Make the expression depend on the template parameter NativeT so
  // that this compile-time error only appears if this function is
  // instantiated with some concrete type that is not specialized
  // below.
  static_assert(!std::is_same<NativeT, NativeT>::value,
                "Cannot map native type to primitive type.");
  return PRIMITIVE_TYPE_INVALID;
}

// Declarations of specializations for each native type which correspond to a
// ZKX primitive type.
template <>
constexpr PrimitiveType NativeToPrimitiveType<bool>() {
  return PRED;
}

// Unsigned integer
template <>
constexpr PrimitiveType NativeToPrimitiveType<u1>() {
  return U1;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<u2>() {
  return U2;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<u4>() {
  return U4;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<uint8_t>() {
  return U8;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<uint16_t>() {
  return U16;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<uint32_t>() {
  return U32;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<uint64_t>() {
  return U64;
}

// Signed integer
template <>
constexpr PrimitiveType NativeToPrimitiveType<s1>() {
  return S1;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<s2>() {
  return S2;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<s4>() {
  return S4;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<int8_t>() {
  return S8;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<int16_t>() {
  return S16;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<int32_t>() {
  return S32;
}

template <>
constexpr PrimitiveType NativeToPrimitiveType<int64_t>() {
  return S64;
}

#define ZK_DTYPES_CONVERSION(cpp_type, unused, enum, unused2) \
  template <>                                                 \
  constexpr PrimitiveType NativeToPrimitiveType<cpp_type>() { \
    return enum;                                              \
  }
ZK_DTYPES_PUBLIC_TYPE_LIST(ZK_DTYPES_CONVERSION)
#undef ZK_DTYPES_CONVERSION

// Returns the native type (eg, uint32_t) corresponding to the given template
// parameter ZKX primitive type (eg, U32).
template <PrimitiveType>
struct PrimitiveTypeToNative;

// Declarations of specializations for each native type which correspond to a
// ZKX primitive type.
template <>
struct PrimitiveTypeToNative<PRED> {
  using type = bool;
};

// Unsigned integer
template <>
struct PrimitiveTypeToNative<U1> {
  using type = u1;
};

template <>
struct PrimitiveTypeToNative<U2> {
  using type = u2;
};

template <>
struct PrimitiveTypeToNative<U4> {
  using type = u4;
};

template <>
struct PrimitiveTypeToNative<U8> {
  using type = uint8_t;
};

template <>
struct PrimitiveTypeToNative<U16> {
  using type = uint16_t;
};

template <>
struct PrimitiveTypeToNative<U32> {
  using type = uint32_t;
};

template <>
struct PrimitiveTypeToNative<U64> {
  using type = uint64_t;
};

// Signed integer
template <>
struct PrimitiveTypeToNative<S1> {
  using type = s1;
};

template <>
struct PrimitiveTypeToNative<S2> {
  using type = s2;
};

template <>
struct PrimitiveTypeToNative<S4> {
  using type = s4;
};

template <>
struct PrimitiveTypeToNative<S8> {
  using type = int8_t;
};

template <>
struct PrimitiveTypeToNative<S16> {
  using type = int16_t;
};

template <>
struct PrimitiveTypeToNative<S32> {
  using type = int32_t;
};

template <>
struct PrimitiveTypeToNative<S64> {
  using type = int64_t;
};

// Token
template <>
struct PrimitiveTypeToNative<TOKEN> {
  using type = void;
};

#define ZK_DTYPES_CONVERSION(cpp_type, unused, enum, unused2) \
  template <>                                                 \
  struct PrimitiveTypeToNative<enum> {                        \
    using type = cpp_type;                                    \
  };
ZK_DTYPES_PUBLIC_TYPE_LIST(ZK_DTYPES_CONVERSION)
#undef ZK_DTYPES_CONVERSION

template <PrimitiveType kType>
using NativeTypeOf = typename PrimitiveTypeToNative<kType>::type;

template <PrimitiveType kPrimitiveType>
using PrimitiveTypeConstant =
    std::integral_constant<PrimitiveType, kPrimitiveType>;

// Returns true if values of the given primitive type are held in array shapes.
inline constexpr bool IsArrayType(PrimitiveType primitive_type) {
  return primitive_type != TUPLE && primitive_type != OPAQUE_TYPE &&
         primitive_type != TOKEN && primitive_type > PRIMITIVE_TYPE_INVALID &&
         primitive_type < PrimitiveType_ARRAYSIZE;
}

constexpr bool IsSignedIntegralType(PrimitiveType type) {
  return type == S1 || type == S2 || type == S4 || type == S8 || type == S16 ||
         type == S32 || type == S64;
}

constexpr bool IsUnsignedIntegralType(PrimitiveType type) {
  return type == U1 || type == U2 || type == U4 || type == U8 || type == U16 ||
         type == U32 || type == U64;
}

constexpr bool IsIntegralType(PrimitiveType type) {
  return IsUnsignedIntegralType(type) || IsSignedIntegralType(type);
}

constexpr bool Is8BitIntegralType(PrimitiveType type) {
  return type == S8 || type == U8;
}

constexpr bool IsPrimeFieldType(PrimitiveType type) {
#define ZK_DTYPES_CASE(unused, unused2, enum, unused3) type == enum ||
  return ZK_DTYPES_PUBLIC_PRIME_FIELD_TYPE_LIST(ZK_DTYPES_CASE) false;
#undef ZK_DTYPES_CASE
}

constexpr bool IsExtensionFieldType(PrimitiveType type) {
#define ZK_DTYPES_CASE(unused, unused2, enum, unused3) type == enum ||
  return ZK_DTYPES_PUBLIC_EXT_FIELD_TYPE_LIST(ZK_DTYPES_CASE) false;
#undef ZK_DTYPES_CASE
}

constexpr bool IsFieldType(PrimitiveType type) {
  return IsPrimeFieldType(type) || IsExtensionFieldType(type);
}

constexpr bool IsEcPointType(PrimitiveType type) {
#define ZK_DTYPES_CASE(unused, unused2, enum, unused3) type == enum ||
  return ZK_DTYPES_PUBLIC_EC_POINT_TYPE_LIST(ZK_DTYPES_CASE) false;
#undef ZK_DTYPES_CASE
}

// Returns true if the type supports comparison operators (<, >, <=, >=).
// Extension fields and EC points don't have natural ordering.
constexpr bool IsComparableType(PrimitiveType type) {
  return !IsExtensionFieldType(type) && !IsEcPointType(type);
}

constexpr bool IsZkDtypesType(PrimitiveType type) {
  if (type == U1 || type == S1 || type == U2 || type == S2 || type == U4 ||
      type == S4) {
    return true;
  }
#define ZK_DTYPES_CASE(unused, unused2, enum, unused3) type == enum ||
  return ZK_DTYPES_PUBLIC_TYPE_LIST(ZK_DTYPES_CASE) false;
#undef ZK_DTYPES_CASE
}

template <typename R, typename F>
constexpr R PrimitiveTypeSwitch(F&& f, PrimitiveType type);

constexpr bool IsMontgomeryForm(PrimitiveType type) {
  return PrimitiveTypeSwitch<bool>(
      [&](auto primitive_type_constant) -> bool {
        if constexpr (IsFieldType(primitive_type_constant) ||
                      IsEcPointType(primitive_type_constant)) {
          using NativeT = NativeTypeOf<primitive_type_constant>;
          return NativeT::kUseMontgomery;
        } else {
          return false;
        }
      },
      type);
}

constexpr bool IsStandardForm(PrimitiveType type) {
  return PrimitiveTypeSwitch<bool>(
      [&](auto primitive_type_constant) -> bool {
        if constexpr (IsFieldType(primitive_type_constant) ||
                      IsEcPointType(primitive_type_constant)) {
          using NativeT = NativeTypeOf<primitive_type_constant>;
          return !NativeT::kUseMontgomery;
        } else {
          return false;
        }
      },
      type);
}

template <typename R, typename F>
constexpr R IntegralTypeSwitch(F&& f, PrimitiveType type) {
  if (ABSL_PREDICT_TRUE(IsIntegralType(type))) {
    switch (type) {
      case S1:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::S1>());
      case S2:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::S2>());
      case S4:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::S4>());
      case S8:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::S8>());
      case S16:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::S16>());
      case S32:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::S32>());
      case S64:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::S64>());
      case U1:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::U1>());
      case U2:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::U2>());
      case U4:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::U4>());
      case U8:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::U8>());
      case U16:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::U16>());
      case U32:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::U32>());
      case U64:
        return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::U64>());
      default:
        ABSL_UNREACHABLE();
    }
  }
  LOG(FATAL) << "Not an integral data type " << type;
}

template <typename R, typename F>
constexpr R FieldTypeSwitch(F&& f, PrimitiveType type) {
  if (ABSL_PREDICT_TRUE(IsFieldType(type))) {
    switch (type) {
#define ZK_DTYPES_CASE(unused, unused2, enum, unused3) \
  case enum:                                           \
    return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::enum>());
      ZK_DTYPES_PUBLIC_FIELD_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE
      default:
        ABSL_UNREACHABLE();
    }
  }
  LOG(FATAL) << "Not a prime field data type " << type;
}

template <typename R, typename F>
constexpr R EcPointTypeSwitch(F&& f, PrimitiveType type) {
  if (ABSL_PREDICT_TRUE(IsEcPointType(type))) {
    switch (type) {
#define ZK_DTYPES_CASE(unused, unused2, enum, unused3) \
  case enum:                                           \
    return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::enum>());
      ZK_DTYPES_PUBLIC_EC_POINT_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE
      default:
        ABSL_UNREACHABLE();
    }
  }
  LOG(FATAL) << "Not an elliptic curve point data type " << type;
}

template <typename R, typename F>
constexpr R ArrayTypeSwitch(F&& f, PrimitiveType type) {
  if (ABSL_PREDICT_TRUE(IsArrayType(type))) {
    if (IsIntegralType(type)) {
      return IntegralTypeSwitch<R>(std::forward<F>(f), type);
    }
    if (IsFieldType(type)) {
      return FieldTypeSwitch<R>(std::forward<F>(f), type);
    }
    if (IsEcPointType(type)) {
      return EcPointTypeSwitch<R>(std::forward<F>(f), type);
    }
    if (type == PRED) {
      return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::PRED>());
    }
  }
  LOG(FATAL) << "Not an array data type " << type;
}

template <typename R, typename F>
constexpr R PrimitiveTypeSwitch(F&& f, PrimitiveType type) {
  if (ABSL_PREDICT_TRUE(IsArrayType(type))) {
    return ArrayTypeSwitch<R>(std::forward<F>(f), type);
  }
  if (type == TUPLE) {
    return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::TUPLE>());
  }
  if (type == TOKEN) {
    return std::forward<F>(f)(PrimitiveTypeConstant<PrimitiveType::TOKEN>());
  }
  if (type == OPAQUE_TYPE) {
    return std::forward<F>(f)(
        PrimitiveTypeConstant<PrimitiveType::OPAQUE_TYPE>());
  }
  LOG(FATAL) << "unhandled type " << type;
}

namespace internal {

template <PrimitiveType primitive_type>
inline constexpr int PrimitiveTypeBitWidth() {
  if constexpr (IsArrayType(primitive_type)) {
    using NativeT = NativeTypeOf<primitive_type>;
    if constexpr (IsIntegralType(primitive_type)) {
      static_assert(is_specialized_integral_v<NativeT>);
      static_assert(std::numeric_limits<NativeT>::is_signed ==
                    IsSignedIntegralType(primitive_type));
      static_assert(std::numeric_limits<NativeT>::radix == 2);
      return std::numeric_limits<NativeT>::digits +
             (IsSignedIntegralType(primitive_type) ? 1 : 0);
    }
    if constexpr (IsFieldType(primitive_type) ||
                  IsEcPointType(primitive_type)) {
      return NativeT::kBitWidth;
    }
    if constexpr (primitive_type == PRED) {
      return std::numeric_limits<NativeT>::digits;
    }
  }
  return 0;
}

template <int... Types>
inline constexpr auto BitWidthArrayHelper(
    std::integer_sequence<int, Types...>) {
  return std::array{PrimitiveTypeBitWidth<PrimitiveType{Types}>()...};
}

inline constexpr auto kBitWidths = BitWidthArrayHelper(
    std::make_integer_sequence<int, PrimitiveType_ARRAYSIZE>{});

template <int... Types>
inline constexpr auto ByteWidthArrayHelper(
    std::integer_sequence<int, Types...>) {
  return std::array{tsl::MathUtil::CeilOfRatio(
      PrimitiveTypeBitWidth<PrimitiveType{Types}>(), 8)...};
}

inline constexpr auto kByteWidths = ByteWidthArrayHelper(
    std::make_integer_sequence<int, PrimitiveType_ARRAYSIZE>{});

template <const std::array<int, PrimitiveType_ARRAYSIZE>& kWidths>
inline constexpr int WidthForType(PrimitiveType type) {
  if (ABSL_PREDICT_TRUE(IsArrayType(type))) {
    return kWidths[type];
  }
  LOG(FATAL) << "Unhandled primitive type " << type;
}

}  // namespace internal

// Returns the number of bits in the representation for a given type.
inline constexpr int BitWidth(PrimitiveType type) {
  return internal::WidthForType<internal::kBitWidths>(type);
}

// Returns the number of bytes in the representation for a given type.
inline constexpr int ByteWidth(PrimitiveType type) {
  return internal::WidthForType<internal::kByteWidths>(type);
}

PrimitiveType UnsignedIntegralTypeForBitWidth(int64_t src_bitwidth);

PrimitiveType SignedIntegralTypeForBitWidth(int64_t src_bitwidth);

// Returns the higher-precision element type if a and b are both floating
// point types; otherwise, checks that they have the same element type
// and returns it.
inline PrimitiveType HigherPrecisionType(PrimitiveType a, PrimitiveType b) {
  // Returns a tuple where the elements are lexicographically ordered in terms
  // of importance.
  auto type_properties = [](PrimitiveType type) {
    return std::make_tuple(
        // Prefer wider types over narrower types.
        BitWidth(type),
        // Prefer signed integer types over unsigned integer types.
        IsSignedIntegralType(type));
  };
  auto a_properties = type_properties(a);
  auto b_properties = type_properties(b);
  if (a_properties > b_properties) {
    return a;
  }
  if (b_properties > a_properties) {
    return b;
  }
  CHECK_EQ(a, b);
  return a;
}

// Returns the lower-case name of the given primitive type.
std::string_view LowercasePrimitiveTypeName(PrimitiveType s);

// Returns the PrimitiveType matching the given name. The given name is expected
// to be lower-case.
absl::StatusOr<PrimitiveType> StringToPrimitiveType(std::string_view name);

// Returns true if the given name is a primitive type string (lower-case).
bool IsPrimitiveTypeName(std::string_view name);

// Returns whether `type` can be expressed as an instance of T.
// For example,
//  IsCanonicalRepresentation<int32_t>(S8)         // true, 8 <= 32
//  IsCanonicalRepresentation<uint16_t>(S16)       // false, unsigned.
template <typename T>
bool IsCanonicalRepresentation(PrimitiveType type) {
  return PrimitiveTypeSwitch<bool>(
      [](auto primitive_type) -> bool {
        if constexpr (IsSignedIntegralType(primitive_type)) {
          return std::numeric_limits<T>::is_integer &&
                 std::numeric_limits<T>::is_signed &&
                 BitWidth(primitive_type) <=
                     (std::numeric_limits<T>::digits + 1);
        }
        if constexpr (IsUnsignedIntegralType(primitive_type) ||
                      primitive_type == PRED) {
          return std::numeric_limits<T>::is_integer &&
                 !std::numeric_limits<T>::is_signed &&
                 BitWidth(primitive_type) <= std::numeric_limits<T>::digits;
        }
        if constexpr (IsFieldType(primitive_type) ||
                      IsEcPointType(primitive_type)) {
          // TODO(chokobole): Maybe we need to consider binary field packing
          // here.
          using NativeT = NativeTypeOf<primitive_type>;
          return std::is_same_v<T, NativeT>;
        }
        return false;
      },
      type);
}

constexpr bool IsSubByteNonPredType(PrimitiveType type) {
  return IsArrayType(type) && type != PRED && BitWidth(type) < 8;
}

inline void PackIntN(PrimitiveType input_type, absl::Span<const char> input,
                     absl::Span<char> output) {
  zkx::PackIntN(BitWidth(input_type), input, output);
}

inline void UnpackIntN(PrimitiveType input_type, absl::Span<const char> input,
                       absl::Span<char> output) {
  zkx::UnpackIntN(BitWidth(input_type), input, output);
}

template <typename NativeT>
std::string NativeTypeToString(NativeT value) {
  constexpr PrimitiveType primitive_type = NativeToPrimitiveType<NativeT>();
  if constexpr (IsZkDtypesType(primitive_type)) {
    return value.ToString();
  } else {
    return absl::StrCat(value);
  }
}

}  // namespace zkx::primitive_util

#endif  // ZKX_PRIMITIVE_UTIL_H_
