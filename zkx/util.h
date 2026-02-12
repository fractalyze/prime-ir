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

// Generally useful utility functions that are common to (not specific to any
// given part of) the ZKX code base.

#ifndef ZKX_UTIL_H_
#define ZKX_UTIL_H_

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <limits>
#include <type_traits>
#include <vector>

#include "Eigen/Core"
#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/base/macros.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"

#include "xla/tsl/lib/math/math_util.h"
#include "zkx/status_macros.h"
#include "zkx/zkx_data.pb.h"

namespace zkx {

// Ranks greater than 6 are very rare, so use InlinedVector<int64_t, 6> to store
// the bounds and indices. And for the rare cases of ranks greater than 6,
// the InlinedVector will just behave like an std::vector<> and allocate the
// memory to store its values.
inline constexpr int InlineRank() { return 6; }
using DimensionVector = absl::InlinedVector<int64_t, InlineRank()>;
using DimLevelTypeVector = absl::InlinedVector<DimLevelType, InlineRank()>;

// Performs a copy of count values from src to dest, using different strides for
// source and destination. The source starting index is src_base, while the
// destination one is dest_base.
template <typename D, typename S>
void StridedCopy(D* dest, int64_t dest_stride, const S* src, int64_t src_stride,
                 int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    dest[i * dest_stride] = static_cast<D>(src[i * src_stride]);
  }
}

// Adds some context information to the error message in a
// absl::Status. This is useful as absl::Statuses are
// propagated upwards.
absl::Status AddStatus(absl::Status prior, std::string_view context);
absl::Status AppendStatus(absl::Status prior, std::string_view context);

// Returns a PaddingConfig object that represents no padding for the given rank.
PaddingConfig MakeNoPaddingConfig(int64_t rank);

// Imports the templated FloorOfRatio math function from the TensorFlow
// namespace, as it is very commonly used.
template <typename T>
constexpr T FloorOfRatio(T dividend, T divisor) {
  return tsl::MathUtil::FloorOfRatio<T>(dividend, divisor);
}

// Imports the templated CeilOfRatio math function from the TensorFlow
// namespace, as it is very commonly used.
template <typename T>
constexpr T CeilOfRatio(T dividend, T divisor) {
  return tsl::MathUtil::CeilOfRatio<T>(dividend, divisor);
}

// Rounds the value up to a multiple of the divisor by first calling CeilOfRatio
// then multiplying by the divisor. For example: RoundUpTo(13, 8) => 16
template <typename T>
constexpr T RoundUpTo(T value, T divisor) {
  return CeilOfRatio(value, divisor) * divisor;
}

// Rounds the value down to a multiple of the divisor by first calling
// FloorOfRatio then multiplying by the divisor. For example:
// RoundDownTo(13, 8) => 8
template <typename T>
constexpr T RoundDownTo(T value, T divisor) {
  return FloorOfRatio(value, divisor) * divisor;
}

// Given a number of flops executed in an amount of time, produces a string that
// represents the throughput;
// e.g. HumanReadableNumFlops(1e9, 1e9) => 1.00GFLOP/s.
std::string HumanReadableNumFlops(double flops, double nanoseconds);

// Returns a mask with "width" number of least significant bits set.
template <typename T>
constexpr inline T LsbMask(int width) {
  static_assert(std::is_unsigned<T>::value,
                "T should be an unsigned integer type");
  ABSL_ASSERT(width >= 0);
  ABSL_ASSERT(width <= std::numeric_limits<T>::digits);
  return width == 0
             ? 0
             : static_cast<T>(-1) >> (std::numeric_limits<T>::digits - width);
}

// Returns `base` multiplied by itself `exponent` number of times.
//
// Note: returns 1 when `exponent` is zero.
// Precondition: `exponent` is non-negative for integral `T`.
template <typename T, typename ExpType>
constexpr T IPow(T base, ExpType exponent) {
  static_assert(std::numeric_limits<ExpType>::is_integer);
  if constexpr (std::numeric_limits<T>::is_integer) {
    // A negative `exponent` is indicative of a logic bug for integral `base`.
    // We disallow it for floating-point types for symmetry.
    ABSL_ASSERT(exponent >= 0);
  }
  const bool take_reciprocal = exponent < 0;
  // We use the right-to-left binary exponentiation algorithm.
  T result(1);
  for (;;) {
    if ((exponent & 1) != 0) {
      result *= base;
    }
    exponent /= 2;
    if (exponent == 0) {
      break;
    }
    base *= base;
  }
  if constexpr (std::numeric_limits<ExpType>::is_signed) {
    if (take_reciprocal) {
      return T(1) / result;
    }
  }
  return result;
}

// UnsignedIntegerTypeForSize<N> gets an unsigned integer with the given size in
// bytes.
template <size_t>
struct UnsignedIntegerTypeForSize;

template <>
struct UnsignedIntegerTypeForSize<1> {
  using type = uint8_t;
};

template <>
struct UnsignedIntegerTypeForSize<2> {
  using type = uint16_t;
};

template <>
struct UnsignedIntegerTypeForSize<4> {
  using type = uint32_t;
};

template <>
struct UnsignedIntegerTypeForSize<8> {
  using type = uint64_t;
};

template <size_t kBytes>
using UnsignedIntegerTypeForSizeType =
    typename UnsignedIntegerTypeForSize<kBytes>::type;

template <size_t kBytes>
using SignedIntegerTypeForSizeType =
    std::make_signed_t<UnsignedIntegerTypeForSizeType<kBytes>>;

template <typename T>
constexpr int NanPayloadBits() {
  // Floating point types with signaling NaNs have payloads.
  if constexpr (!std::numeric_limits<T>::has_signaling_NaN) {
    return 0;
  }
  return std::numeric_limits<T>::digits - 1;
}

template <typename T>
constexpr uint64_t QuietNanWithoutPayload() {
  constexpr int bits = NanPayloadBits<T>();
  if constexpr (bits > 0) {
    return uint64_t{1} << (bits - 1);
  }
  return 0;
}

template <typename T>
constexpr uint64_t NanPayloadBitMask() {
  constexpr int bits = NanPayloadBits<T>();
  if constexpr (bits > 0) {
    return LsbMask<uint64_t>(bits);
  }
  return 0;
}

template <typename T>
T NanWithSignAndPayload(bool sign, uint64_t nan_payload) {
  static_assert(NanPayloadBits<T>() > 0);
  using RepT = UnsignedIntegerTypeForSizeType<sizeof(T)>;
  // Clear the sign bit.
  T val = Eigen::numext::abs(std::numeric_limits<T>::quiet_NaN());
  // Conditionally set the sign bit.
  if (sign) {
    val = -val;
  }
  auto rep = absl::bit_cast<RepT>(val);
  rep |= uint64_t{sign} << (std::numeric_limits<RepT>::digits - 1);
  constexpr int kPayloadBits = NanPayloadBits<T>();
  if (kPayloadBits > 0) {
    // Clear rep's NaN payload.
    rep &= ~NanPayloadBitMask<T>();
    CHECK_NE(nan_payload, 0);
    rep |= nan_payload;
  }
  return absl::bit_cast<T>(rep);
}

template <typename Container>
int64_t PositionInContainer(const Container& container, int64_t value) {
  return std::distance(container.begin(), absl::c_find(container, value));
}

int64_t Product(absl::Span<const int64_t> xs);

// Returns the start indices of consecutive non-overlapping subsequences of `a`
// and `b` with the same product, i.e. `(i, j)` so
// • a = {a[0 = i_0], ..., a[i_1 - 1], a[i_1], ... , a[i_2 - 1], ...}
// • b = {b[0 = j_0], ..., b[j_1 - 1], b[j_1], ... , b[j_2 - 1], ...}
// • ∀ k . 0 <= k < CommonFactors(a, b).size - 1 =>
//         a[i_k] × a[i_k + 1] × ... × a[i_(k+1) - 1] =
//         b[j_k] × b[j_k + 1] × ... × b[j_(k+1) - 1]
// where `CommonFactors(a, b)[CommonFactors(a, b).size - 1] = (a.size, b.size)`
//
// If input and output are the same, return {(0, 0), {1, 1}, ... {a.size,
// b.size}}, otherwise if the given shapes have non-zero size, returns the
// bounds of the shortest possible such subsequences; else, returns `{(0, 0),
// (a.size, b.size)}`.
absl::InlinedVector<std::pair<int64_t, int64_t>, 8> CommonFactors(
    absl::Span<const int64_t> a, absl::Span<const int64_t> b);

// Returns non contracting dimensions for a dot operand based on rank, batch and
// contracting dimension numbers.
DimensionVector GetNonContractingDims(
    int64_t rank, absl::Span<const int64_t> contracting_dim_numbers,
    absl::Span<const int64_t> batch_dim_numbers);

// Removes illegal characters from filenames.
std::string SanitizeFileName(std::string_view file_name);

template <typename C, typename Value>
int64_t FindIndex(const C& c, Value&& value) {
  auto it = absl::c_find(c, std::forward<Value>(value));
  return std::distance(c.begin(), it);
}

template <typename T>
std::vector<T> SpanToVector(absl::Span<const T> slice) {
  return std::vector<T>(slice.begin(), slice.end());
}

// Returns true if `x` fits in 32-bits.
template <typename T>
bool IsInt32(T x) {
  // Following conversion rules: "the value is unchanged if it can be
  // represented in the destination type (and bit-field width); otherwise, the
  // value is implementation-defined."
  return static_cast<int32_t>(x) == x;
}

template <typename T>
absl::Status EraseElementFromVector(std::vector<T>* container, const T& value) {
  // absl::c_find returns a const_iterator which does not seem to work on
  // gcc 4.8.4, and this breaks the ubuntu/xla_gpu build bot.
  auto it = std::find(container->begin(), container->end(), value);
  TF_RET_CHECK(it != container->end());
  container->erase(it);
  return absl::OkStatus();
}

// Takes a sequence of unpacked n-bit values, such that every byte stores one
// value in the low-order bits, and packs them so every byte stores as many
// which will fit. `output` should have ceil((input.size()*kBitsPerElement)/8)
// bytes. The high-order bits of each byte in `input` are ignored.
template <size_t kBitsPerElement>
void PackIntN(absl::Span<const char> input, absl::Span<char> output) {
  constexpr auto kElementsPerByte = 8 / kBitsPerElement;
  const size_t aligned_inputs = input.size() / kElementsPerByte;
  for (size_t i = 0; i < aligned_inputs; ++i) {
    char byte = 0;
    for (size_t j = 0; j < kElementsPerByte; ++j) {
      byte |=
          (input[i * kElementsPerByte + j] & LsbMask<uint8_t>(kBitsPerElement))
          << (kBitsPerElement * (kElementsPerByte - j - 1));
    }
    output[i] = byte;
  }
  if (size_t remainder = input.size() % kElementsPerByte; remainder != 0) {
    char byte = 0;
    for (size_t j = 0; j < remainder; ++j) {
      byte |= (input[aligned_inputs * kElementsPerByte + j] &
               LsbMask<uint8_t>(kBitsPerElement))
              << (kBitsPerElement * (kElementsPerByte - j - 1));
    }
    output[aligned_inputs] = byte;
  }
}

inline void PackIntN(int bits_per_element, absl::Span<const char> input,
                     absl::Span<char> output) {
  if (bits_per_element == 2) {
    PackIntN<2>(input, output);
  } else if (bits_per_element == 4) {
    PackIntN<4>(input, output);
  } else {
    LOG(FATAL) << "Invalid bits_per_element: " << bits_per_element;
  }
}

// Takes a sequence of packed values, such that every byte stores multiple
// values, and unpacks them so every byte stores one value in the low-order
// bits. `input` should have
// ceil(output.size()*8/kBitsPerElement) bytes. The high-order bits in each
// output are zero.
template <size_t kBitsPerElement>
void UnpackIntN(absl::Span<const char> input, absl::Span<char> output) {
  constexpr auto kElementsPerByte = 8 / kBitsPerElement;
  const size_t aligned_outputs = output.size() / kElementsPerByte;
  for (size_t i = 0; i < aligned_outputs; ++i) {
    const char byte = input[i];
    for (int j = 0; j < kElementsPerByte; ++j) {
      output[i * kElementsPerByte + j] =
          (byte >> (kBitsPerElement * (kElementsPerByte - j - 1))) &
          LsbMask<uint8_t>(kBitsPerElement);
    }
  }
  if (size_t remainder = output.size() % kElementsPerByte; remainder != 0) {
    const char byte = input[aligned_outputs];
    for (size_t j = 0; j < remainder; ++j) {
      output[aligned_outputs * kElementsPerByte + j] =
          (byte >> (kBitsPerElement * (kElementsPerByte - j - 1))) &
          LsbMask<uint8_t>(kBitsPerElement);
    }
  }
}

inline void UnpackIntN(int bits_per_element, absl::Span<const char> input,
                       absl::Span<char> output) {
  if (bits_per_element == 2) {
    UnpackIntN<2>(input, output);
  } else if (bits_per_element == 4) {
    UnpackIntN<4>(input, output);
  } else {
    LOG(FATAL) << "Invalid bits_per_element: " << bits_per_element;
  }
}

// Returns a container with `sorted_ids_to_remove` elements removed.
template <typename T>
static T RemoveElements(absl::Span<const int64_t> sorted_ids_to_remove,
                        const T& container) {
  T result;
  auto id_to_remove = sorted_ids_to_remove.begin();
  for (size_t i = 0; i < container.size(); ++i) {
    if (id_to_remove != sorted_ids_to_remove.end() && *id_to_remove == i) {
      ++id_to_remove;
      continue;
    }
    result.push_back(container[i]);
  }
  return result;
}

class HloInstruction;
class HloModule;

// A predicate over HLO instruction.
using HloPredicate = std::function<bool(const HloInstruction*)>;
using HloModulePredicate = std::function<bool(const HloModule*)>;

inline bool HloPredicateTrue(const HloInstruction*) { return true; }
inline bool HloPredicateFalse(const HloInstruction*) { return false; }

using Vector2 = std::array<int64_t, 2>;
using Vector3 = std::array<int64_t, 3>;

}  // namespace zkx

#endif  // ZKX_UTIL_H_
