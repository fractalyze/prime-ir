/* Copyright 2025 The ZKIR Authors.

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

#ifndef ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_VANDERMONDEMATRIX_H_
#define ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_VANDERMONDEMATRIX_H_

#include <array>

#include "zkir/Dialect/Field/Conversions/FieldToModArith/ConversionUtils.h"
#include "zkir/Dialect/Field/Conversions/FieldToModArith/PrimeFieldCodeGen.h"
#include "zkir/Utils/BuilderContext.h"

namespace mlir::zkir::field {

// Mixin providing Vandermonde inverse matrix for Toom-Cook multiplication.
//
// Derived class must provide:
//   - getBuilder(): returns ImplicitLocOpBuilder&
//   - getType(): returns Type (extension field type)
template <typename Derived>
class VandermondeMatrix {
public:
  std::array<std::array<PrimeFieldCodeGen, 7>, 7>
  GetVandermondeInverseMatrix() const {
    const auto &self = static_cast<const Derived &>(*this);
    auto extField = cast<ExtensionFieldTypeInterface>(self.getType());
    auto baseField = cast<PrimeFieldType>(extField.getBaseFieldType());
    ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();

    auto C = [&](int64_t x) {
      return PrimeFieldCodeGen(createConst(*b, baseField, x));
    };
    auto C2 = [&](int64_t x, int64_t y) {
      return PrimeFieldCodeGen(createRationalConst(*b, baseField, x, y));
    };

    // NOLINTBEGIN(whitespace/line_length)
    // clang-format off
    return {{
        {C(1),       C(0),       C(0),       C(0),       C(0),        C(0),       C(0)},
        {C2(-1, 3),  C(1),       C2(-1, 2),  C2(-1, 4),  C2(1, 20),   C2(1, 30),  C(-12)},
        {C2(-5, 4),  C2(2, 3),   C2(2, 3),   C2(-1, 24), C2(-1, 24),  C(0),       C(4)},
        {C2(5, 12),  C2(-7, 12), C2(-1, 24), C2(7, 24),  C2(-1, 24),  C2(-1, 24), C(15)},
        {C2(1, 4),   C2(-1, 6),  C2(-1, 6),  C2(1, 24),  C2(1, 24),   C(0),       C(-5)},
        {C2(-1, 12), C2(1, 12),  C2(1, 24),  C2(-1, 24), C2(-1, 120), C2(1, 120), C(-3)},
        {C(0),       C(0),       C(0),       C(0),       C(0),        C(0),       C(1)},
    }};
    // clang-format on
    // NOLINTEND
  }
};

} // namespace mlir::zkir::field

// NOLINTNEXTLINE(whitespace/line_length)
#endif // ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_VANDERMONDEMATRIX_H_
