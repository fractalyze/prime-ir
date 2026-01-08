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

#ifndef ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_PRIMEFIELDCODEGEN_H_
#define ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_PRIMEFIELDCODEGEN_H_

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"

namespace mlir::zkir::field {

// NOTE(chokobole): This class is not used directly. It is used to generate
// MLIR operations that implement prime field arithmetic. User should use
// FieldCodeGen instead.
class PrimeFieldCodeGen {
public:
  PrimeFieldCodeGen() = default;
  PrimeFieldCodeGen(ImplicitLocOpBuilder *b, Value value)
      : b(b), value(value) {}
  ~PrimeFieldCodeGen() = default;

  operator Value() const { return value; }

  PrimeFieldCodeGen operator+(const PrimeFieldCodeGen &other) const;
  PrimeFieldCodeGen &operator+=(const PrimeFieldCodeGen &other);
  PrimeFieldCodeGen operator-(const PrimeFieldCodeGen &other) const;
  PrimeFieldCodeGen &operator-=(const PrimeFieldCodeGen &other);
  PrimeFieldCodeGen operator*(const PrimeFieldCodeGen &other) const;
  PrimeFieldCodeGen &operator*=(const PrimeFieldCodeGen &other);
  PrimeFieldCodeGen operator-() const;
  PrimeFieldCodeGen Double() const;
  PrimeFieldCodeGen Square() const;
  PrimeFieldCodeGen Inverse() const;

private:
  ImplicitLocOpBuilder *b = nullptr; // not owned
  Value value;
};

} // namespace mlir::zkir::field

#endif // ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_PRIMEFIELDCODEGEN_H_
