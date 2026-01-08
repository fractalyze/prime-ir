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
#include "zk_dtypes/include/field/cubic_extension_field_operation.h"
#include "zk_dtypes/include/field/quadratic_extension_field_operation.h"
#include "zk_dtypes/include/field/quartic_extension_field_operation.h"
#include "zkir/Dialect/Field/Conversions/FieldToModArith/ConversionUtils.h"
#include "zkir/Dialect/Field/Conversions/FieldToModArith/FrobeniusCoeffs.h"
#include "zkir/Dialect/Field/Conversions/FieldToModArith/PrimeFieldCodeGen.h"
#include "zkir/Dialect/Field/Conversions/FieldToModArith/VandermondeMatrix.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::zkir::field {

// Forward declaration
template <size_t N>
class ExtensionFieldCodeGen;

} // namespace mlir::zkir::field

namespace zk_dtypes {

template <size_t N>
class ExtensionFieldOperationTraits<
    mlir::zkir::field::ExtensionFieldCodeGen<N>> {
public:
  // TODO(chokobole): Support towers of extension field.
  using BaseField = mlir::zkir::field::PrimeFieldCodeGen;
  static constexpr size_t kDegree = N;
};

} // namespace zk_dtypes

namespace mlir::zkir::field {

// Selects the appropriate extension field operation based on degree.
template <size_t N>
struct ExtensionFieldOperationSelector;

template <>
struct ExtensionFieldOperationSelector<2> {
  template <typename Derived>
  using Type = zk_dtypes::QuadraticExtensionFieldOperation<Derived>;
};

template <>
struct ExtensionFieldOperationSelector<3> {
  template <typename Derived>
  using Type = zk_dtypes::CubicExtensionFieldOperation<Derived>;
};

template <>
struct ExtensionFieldOperationSelector<4> {
  template <typename Derived>
  using Type = zk_dtypes::QuarticExtensionFieldOperation<Derived>;
};

// NOTE(chokobole): This class is not used directly. It is used to generate
// MLIR operations that implement extension field arithmetic. User should use
// FieldCodeGen instead.
template <size_t N>
class ExtensionFieldCodeGen
    : public ExtensionFieldOperationSelector<N>::template Type<
          ExtensionFieldCodeGen<N>>,
      public VandermondeMatrix<ExtensionFieldCodeGen<N>>,
      public FrobeniusCoeffs<ExtensionFieldCodeGen<N>, N> {
  using Base = typename ExtensionFieldOperationSelector<N>::template Type<
      ExtensionFieldCodeGen<N>>;

public:
  // Bring base class operator* into scope. Without this, the derived class's
  // operator*(const PrimeFieldCodeGen&) hides all operator* overloads from
  // Base.
  using Base::operator*;
  // TODO(junbeomlee): Remove these using declarations after refactoring
  // zk_dtypes to not define these methods in QuarticExtensionFieldOperation.
  using VandermondeMatrix<
      ExtensionFieldCodeGen<N>>::GetVandermondeInverseMatrix;
  using FrobeniusCoeffs<ExtensionFieldCodeGen<N>, N>::GetFrobeniusCoeffs;

  ExtensionFieldCodeGen(ImplicitLocOpBuilder *b, Value value, Value nonResidue)
      : b(b), value(value), nonResidue(nonResidue) {}
  ~ExtensionFieldCodeGen() = default;

  operator Value() const { return value; }

  // Accessors for mixin classes
  ImplicitLocOpBuilder &getBuilder() const {
    assert(b);
    return *b;
  }
  Type getType() const { return value.getType(); }

  ExtensionFieldCodeGen operator*(const PrimeFieldCodeGen &scalar) const {
    std::array<PrimeFieldCodeGen, N> coeffs = ToCoeffs();
    for (size_t i = 0; i < N; ++i) {
      coeffs[i] = coeffs[i] * scalar;
    }
    return FromCoeffs(coeffs);
  }

  // zk_dtypes ExtensionFieldOperation methods
  std::array<PrimeFieldCodeGen, N> ToCoeffs() const {
    Operation::result_range coeffs = toCoeffs(*b, value);
    std::array<PrimeFieldCodeGen, N> ret;
    for (size_t i = 0; i < N; ++i) {
      ret[i] = PrimeFieldCodeGen(b, coeffs[i]);
    }
    return ret;
  }
  ExtensionFieldCodeGen
  FromCoeffs(const std::array<PrimeFieldCodeGen, N> &values) const {
    SmallVector<Value, N> coeffs;
    for (size_t i = 0; i < N; ++i) {
      coeffs.push_back(values[i]);
    }
    return {b, fromCoeffs(*b, value.getType(), coeffs), nonResidue};
  }
  PrimeFieldCodeGen NonResidue() const {
    return PrimeFieldCodeGen(b, nonResidue);
  }
  zk_dtypes::ExtensionFieldMulAlgorithm GetMulAlgorithm() const {
    if constexpr (N == 4) {
      return zk_dtypes::ExtensionFieldMulAlgorithm::kToomCook;
    } else {
      return zk_dtypes::ExtensionFieldMulAlgorithm::kKaratsuba;
    }
  }
  // TODO(junbeomlee): Consider using kCustom algorithm based on limb count
  // and non-residue value, which can be obtained from the type.
  zk_dtypes::ExtensionFieldMulAlgorithm GetSquareAlgorithm() const {
    if constexpr (N == 4) {
      return zk_dtypes::ExtensionFieldMulAlgorithm::kToomCook;
    } else {
      return zk_dtypes::ExtensionFieldMulAlgorithm::kKaratsuba;
    }
  }
  PrimeFieldCodeGen CreateConstBaseField(int64_t x) const {
    auto extField = cast<ExtensionFieldTypeInterface>(value.getType());
    auto baseField = cast<PrimeFieldType>(extField.getBaseFieldType());
    return PrimeFieldCodeGen(b, createConst(*b, baseField, x));
  }

private:
  ImplicitLocOpBuilder *b = nullptr; // not owned
  Value value;
  Value nonResidue;
};

extern template class ExtensionFieldCodeGen<2>;
extern template class ExtensionFieldCodeGen<3>;
extern template class ExtensionFieldCodeGen<4>;

using QuadraticExtensionFieldCodeGen = ExtensionFieldCodeGen<2>;
using CubicExtensionFieldCodeGen = ExtensionFieldCodeGen<3>;
using QuarticExtensionFieldCodeGen = ExtensionFieldCodeGen<4>;

} // namespace mlir::zkir::field

// NOLINTNEXTLINE(whitespace/line_length)
#endif // ZKIR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_EXTENSIONFIELDCODEGEN_H_
