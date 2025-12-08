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

#ifndef ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_EXTENSION_QUARTICEXTENSIONFIELD_H_
#define ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_EXTENSION_QUARTICEXTENSIONFIELD_H_

#include "zkir/Dialect/Field/Conversions/FieldToModArith/Extension/ExtensionField.h"

namespace mlir::zkir::field {

class QuarticExtensionField : public ExtensionField {
public:
  Value square(Value x) override;
  Value mul(Value x, Value y) override;
  Value inverse(Value x) override;
  Value frobeniusMap(Value x, const APInt &exponent) override;

private:
  friend class ExtensionField;

  using ExtensionField::ExtensionField;

  // Toom-Cook interpolation helper for mul and square.
  Value toomCookInterpolate(Value v0, Value v1, Value v2, Value v3, Value v4,
                            Value v5, Value v6);
};

} // namespace mlir::zkir::field

// NOLINTNEXTLINE(whitespace/line_length)
#endif // ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_EXTENSION_QUARTICEXTENSIONFIELD_H_
