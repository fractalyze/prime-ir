/* Copyright 2026 The PrimeIR Authors.

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

#ifndef PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FIELDCODEGEN_H_
#define PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FIELDCODEGEN_H_

#include <type_traits>
#include <utility>
#include <variant>

#include "mlir/Transforms/DialectConversion.h"
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/ExtensionFieldCodeGen.h"
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/PrimeFieldCodeGen.h"

namespace mlir::prime_ir::field {

class FieldCodeGen {
public:
  using CodeGenType =
      std::variant<PrimeFieldCodeGen, QuadraticExtensionFieldCodeGen,
                   CubicExtensionFieldCodeGen, QuarticExtensionFieldCodeGen>;

  template <typename T, typename = std::enable_if_t<
                            !std::is_same_v<std::decay_t<T>, FieldCodeGen> &&
                            std::is_constructible_v<CodeGenType, T>>>
  FieldCodeGen(T &&cg) // NOLINT(runtime/explicit)
      : codeGen(std::forward<T>(cg)) {}
  FieldCodeGen(Type type, Value value, const TypeConverter *converter);
  ~FieldCodeGen() = default;

  operator Value() const;

  FieldCodeGen operator+(const FieldCodeGen &other) const;
  FieldCodeGen operator-(const FieldCodeGen &other) const;
  FieldCodeGen operator*(const FieldCodeGen &other) const;
  FieldCodeGen dbl() const;
  FieldCodeGen square() const;
  FieldCodeGen negate() const;
  FieldCodeGen inverse() const;

private:
  CodeGenType codeGen;
};

} // namespace mlir::prime_ir::field

#endif // PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FIELDCODEGEN_H_
