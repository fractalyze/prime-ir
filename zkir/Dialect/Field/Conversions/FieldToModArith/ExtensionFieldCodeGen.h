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

#ifndef ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_EXTENSIONFIELDCODEGEN_H_
#define ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_EXTENSIONFIELDCODEGEN_H_

#include <cstddef>

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "zk_dtypes/include/field/quadratic_extension_field_operation.h"
#include "zkir/Dialect/Field/Conversions/FieldToModArith/ConversionUtils.h"
#include "zkir/Dialect/Field/Conversions/FieldToModArith/PrimeFieldCodeGen.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::zkir::field {

template <size_t N>
class ExtensionFieldCodeGen
    : public zk_dtypes::QuadraticExtensionFieldOperation<
          ExtensionFieldCodeGen<N>> {
public:
  ExtensionFieldCodeGen(ImplicitLocOpBuilder *b, Type type, Value value,
                        Value nonResidue)
      : b(b), type(type), value(value), nonResidue(nonResidue) {}
  ~ExtensionFieldCodeGen() = default;

  Value getValue() const { return value; }

  // zk_dtypes::QuadraticExtensionFieldOperation methods
  std::array<PrimeFieldCodeGen, N> ToBaseField() const {
    Operation::result_range coeffs = toCoeffs(*b, value);
    std::array<PrimeFieldCodeGen, N> ret;
    for (size_t i = 0; i < N; ++i) {
      ret[i] = PrimeFieldCodeGen(b, coeffs[i]);
    }
    return ret;
  }
  ExtensionFieldCodeGen
  FromBaseFields(const std::array<PrimeFieldCodeGen, N> &values) const {
    SmallVector<Value, N> coeffs;
    for (size_t i = 0; i < N; ++i) {
      coeffs.push_back(values[i].getValue());
    }
    return {b, type, fromCoeffs(*b, type, coeffs), nonResidue};
  }
  size_t DegreeOverBasePrimeField() const {
    assert(isa<ExtensionFieldTypeInterface>(type));
    return N * cast<ExtensionFieldTypeInterface>(type).getDegreeOverBase();
  }
  PrimeFieldCodeGen NonResidue() const {
    return PrimeFieldCodeGen(b, nonResidue);
  }

private:
  ImplicitLocOpBuilder *b = nullptr; // not owned
  Type type;
  Value value;
  Value nonResidue;
};

} // namespace mlir::zkir::field

namespace zk_dtypes {

template <size_t N>
class ExtensionFieldOperationTraits<
    mlir::zkir::field::ExtensionFieldCodeGen<N>> {
public:
  // TODO(chokobole): Support towers of extension field.
  using BaseField = mlir::zkir::field::PrimeFieldCodeGen;
  static constexpr size_t kDegree = N;

  constexpr static bool kHasHint = false;
};

} // namespace zk_dtypes

// NOLINTNEXTLINE(whitespace/line_length)
#endif // ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_EXTENSIONFIELDCODEGEN_H_
