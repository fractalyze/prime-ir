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

#ifndef ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_EXTENSION_EXTENSIONFIELD_H_
#define ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_EXTENSION_EXTENSIONFIELD_H_

#include <memory>

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::zkir::field {

// Helper class to perform extension field operations.
// This class encapsulates the logic for generating MLIR operations
// that implement extension field arithmetic (Fp2, Fp3, etc.).
class ExtensionField {
public:
  static constexpr unsigned kMaxDegreeOverBaseField = 5;

  static std::unique_ptr<ExtensionField>
  create(ImplicitLocOpBuilder &b, ExtensionFieldTypeInterface type,
         const TypeConverter *converter);

  virtual ~ExtensionField() = default;

  Value add(Value x, Value y);
  Value sub(Value x, Value y);
  // NOTE: double is a reserved word in C++.
  Value dbl(Value x);
  Value negate(Value x);
  virtual Value square(Value x) = 0;
  virtual Value mul(Value x, Value y) = 0;
  virtual Value inverse(Value x) = 0;
  virtual Value frobeniusMap(Value x, const APInt &exponent);

protected:
  ExtensionField(ImplicitLocOpBuilder &b, ExtensionFieldTypeInterface type,
                 const TypeConverter *converter);

  ImplicitLocOpBuilder &b;
  ExtensionFieldTypeInterface type;
  const TypeConverter *converter; // not owned
  Value nonResidue;
};

} // namespace mlir::zkir::field

// NOLINTNEXTLINE(whitespace/line_length)
#endif // ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_EXTENSION_EXTENSIONFIELD_H_
