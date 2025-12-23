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

#ifndef ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_EXTENSION_EXTENSIONFIELDIMPL_H_
#define ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_EXTENSION_EXTENSIONFIELDIMPL_H_

#include "zkir/Dialect/Field/Conversions/FieldToModArith/Extension/ExtensionField.h"
#include "zkir/Dialect/Field/Conversions/FieldToModArith/ExtensionFieldCodeGen.h"

namespace mlir::zkir::field {

template <size_t Degree>
class ExtensionFieldImpl : public ExtensionField {
public:
  Value square(Value x) override {
    ExtensionFieldCodeGen<Degree> xGen(&b, x.getType(), x, nonResidue);
    return xGen.Square();
  }

  Value mul(Value x, Value y) override {
    ExtensionFieldCodeGen<Degree> xGen(&b, x.getType(), x, nonResidue);
    ExtensionFieldCodeGen<Degree> yGen(&b, y.getType(), y, nonResidue);
    return xGen * yGen;
  }

  Value inverse(Value x) override {
    ExtensionFieldCodeGen<Degree> xGen(&b, x.getType(), x, nonResidue);
    return *xGen.Inverse();
  }

private:
  friend class ExtensionField;

  using ExtensionField::ExtensionField;
};

extern template class ExtensionFieldImpl<2>;
extern template class ExtensionFieldImpl<3>;
extern template class ExtensionFieldImpl<4>;

using QuadraticExtensionField = ExtensionFieldImpl<2>;
using CubicExtensionField = ExtensionFieldImpl<3>;
using QuarticExtensionField = ExtensionFieldImpl<4>;

} // namespace mlir::zkir::field

// NOLINTNEXTLINE(whitespace/line_length)
#endif // ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_EXTENSION_EXTENSIONFIELDIMPL_H_
