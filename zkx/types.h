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

#ifndef ZKX_TYPES_H_
#define ZKX_TYPES_H_

#include <limits>
#include <type_traits>

#include "zk_dtypes/include/big_int.h"
#include "zk_dtypes/include/intn.h"

namespace zkx {

template <typename T>
struct is_specialized_integral
    : std::bool_constant<std::numeric_limits<T>::is_specialized &&
                         std::numeric_limits<T>::is_integer> {};

template <size_t N>
struct is_specialized_integral<::zk_dtypes::BigInt<N>> : std::true_type {};

template <typename T>
inline constexpr bool is_specialized_integral_v =
    is_specialized_integral<T>::value;

using u1 = ::zk_dtypes::uint1;
using s1 = ::zk_dtypes::int1;
using u2 = ::zk_dtypes::uint2;
using s2 = ::zk_dtypes::int2;
using u4 = ::zk_dtypes::uint4;
using s4 = ::zk_dtypes::int4;

using u128 = ::zk_dtypes::BigInt<2>;
using u256 = ::zk_dtypes::BigInt<4>;

template <class T>
struct is_big_int : std::false_type {};
template <size_t N>
struct is_big_int<::zk_dtypes::BigInt<N>> : std::true_type {};
template <typename T>
inline constexpr bool is_big_int_v = is_big_int<T>::value;

template <class T>
struct is_intN : std::false_type {};
template <int kN, typename UnderlyingType>
struct is_intN<::zk_dtypes::intN<kN, UnderlyingType>> : std::true_type {};

template <typename T>
inline constexpr bool is_intN_v = is_intN<T>::value;

// std::make_signed_t is “behavior undefined” for custom types, so provide a
// general util to make signed/unsigned for both primitive and custom types.
template <typename T, typename = void>
struct make_specialized_unsigned {
  using type = std::make_unsigned_t<T>;
};

template <typename T>
struct make_specialized_unsigned<T, typename std::enable_if_t<is_intN_v<T>>> {
  static_assert(std::is_integral_v<typename T::underlying_type>);
  using type =
      ::zk_dtypes::intN<T::bits,
                        std::make_unsigned_t<typename T::underlying_type>>;
};

template <typename T>
struct make_specialized_unsigned<T,
                                 typename std::enable_if_t<is_big_int_v<T>>> {
  // BigInt is already unsigned, return the same type.
  using type = T;
};

template <typename T>
using make_specialized_unsigned_t = typename make_specialized_unsigned<T>::type;

template <typename T, typename = void>
struct make_specialized_signed {
  using type = std::make_signed_t<T>;
};

template <typename T>
struct make_specialized_signed<T, typename std::enable_if_t<is_intN_v<T>>> {
  static_assert(std::is_integral_v<typename T::underlying_type>);
  using type =
      ::zk_dtypes::intN<T::bits,
                        std::make_signed_t<typename T::underlying_type>>;
};

template <typename T>
struct make_specialized_signed<T, typename std::enable_if_t<is_big_int_v<T>>> {
  // BigInt is unsigned only, no signed variant exists.
  using type = T;
};

template <typename T>
using make_specialized_signed_t = typename make_specialized_signed<T>::type;

}  // namespace zkx

#endif  // ZKX_TYPES_H_
