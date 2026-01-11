/* Copyright 2025 The PrimeIR Authors.

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

#ifndef PRIME_IR_UTILS_ZKDTYPES_H_
#define PRIME_IR_UTILS_ZKDTYPES_H_

#include <cstddef>
#include <type_traits>

#include "zk_dtypes/include/big_int.h"

namespace mlir::prime_ir {

constexpr bool isPowerOfTwo(size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

template <size_t N>
APInt convertToAPInt(const zk_dtypes::BigInt<N> &value,
                     unsigned bits = N * 64) {
  return {bits, static_cast<unsigned>(N), value.limbs()};
}

template <typename T, std::enable_if_t<std::is_integral_v<T>> * = nullptr>
APInt convertToAPInt(T value, unsigned bits = sizeof(T) * 8) {
  return APInt(bits, value);
}

} // namespace mlir::prime_ir

#endif // PRIME_IR_UTILS_ZKDTYPES_H_
