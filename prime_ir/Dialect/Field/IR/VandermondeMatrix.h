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

#ifndef PRIME_IR_DIALECT_FIELD_IR_VANDERMONDEMATRIX_H_
#define PRIME_IR_DIALECT_FIELD_IR_VANDERMONDEMATRIX_H_

#include <array>

#include "zk_dtypes/include/field/extension_field_operation_traits_forward.h"

namespace mlir::prime_ir::field {

// Mixin providing Vandermonde inverse matrix for Toom-Cook multiplication.
//
// Derived class must provide:
//   - CreateConstBaseField(int64_t): creates a constant in the base field
//   - CreateRationalConstBaseField(int64_t, int64_t): creates a rational
//   constant
template <typename Derived>
class VandermondeMatrix {
public:
  auto GetVandermondeInverseMatrix() const {
    using Traits = zk_dtypes::ExtensionFieldOperationTraits<Derived>;
    using BaseFieldT = typename Traits::BaseField;

    const auto &self = static_cast<const Derived &>(*this);

    auto C = [&](int64_t x) -> BaseFieldT {
      return self.createConstBaseField(x);
    };
    auto C2 = [&](int64_t x, int64_t y) -> BaseFieldT {
      return self.createRationalConstBaseField(x, y);
    };

    // clang-format off
    return std::array<std::array<BaseFieldT, 7>, 7>{{
        // NOLINTNEXTLINE(whitespace/line_length)
        {C(1),       C(0),       C(0),       C(0),       C(0),        C(0),       C(0)},
        // NOLINTNEXTLINE(whitespace/line_length)
        {C2(-1, 3),  C(1),       C2(-1, 2),  C2(-1, 4),  C2(1, 20),   C2(1, 30),  C(-12)},
        // NOLINTNEXTLINE(whitespace/line_length)
        {C2(-5, 4),  C2(2, 3),   C2(2, 3),   C2(-1, 24), C2(-1, 24),  C(0),       C(4)},
        // NOLINTNEXTLINE(whitespace/line_length)
        {C2(5, 12),  C2(-7, 12), C2(-1, 24), C2(7, 24),  C2(-1, 24),  C2(-1, 24), C(15)},
        // NOLINTNEXTLINE(whitespace/line_length)
        {C2(1, 4),   C2(-1, 6),  C2(-1, 6),  C2(1, 24),  C2(1, 24),   C(0),       C(-5)},
        // NOLINTNEXTLINE(whitespace/line_length)
        {C2(-1, 12), C2(1, 12),  C2(1, 24),  C2(-1, 24), C2(-1, 120), C2(1, 120), C(-3)},
        // NOLINTNEXTLINE(whitespace/line_length)
        {C(0),       C(0),       C(0),       C(0),       C(0),        C(0),       C(1)},
    }};
    // clang-format on
  }
};

} // namespace mlir::prime_ir::field

#endif // PRIME_IR_DIALECT_FIELD_IR_VANDERMONDEMATRIX_H_
