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

#ifndef PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_EXTENSIONFIELDCODEGEN_H_
#define PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_EXTENSIONFIELDCODEGEN_H_

#include <cstddef>
#include <type_traits>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/ConversionUtils.h"
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/FrobeniusCoeffs.h"
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/PrimeFieldCodeGen.h"
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/VandermondeMatrix.h"
#include "prime_ir/Dialect/Field/IR/ExtensionFieldOperationSelector.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Dialect/Field/IR/PrimeFieldOperation.h"
#include "prime_ir/Utils/BuilderContext.h"

namespace mlir::prime_ir::field {

// Forward declaration
template <size_t N, typename BaseFieldT = PrimeFieldCodeGen>
class ExtensionFieldCodeGen;

// Helper to select the appropriate base operation class.
// Uses ExtensionFieldOperationSelector which chooses the optimal algorithm
// (Karatsuba for degree 2/3, ToomCook for degree 4).
template <size_t N, typename BaseFieldT, typename Derived>
struct ExtensionFieldOperationBase {
  using Type =
      typename ExtensionFieldOperationSelector<N>::template Type<Derived>;
};

} // namespace mlir::prime_ir::field

namespace zk_dtypes {

template <size_t N, typename BaseFieldT>
class ExtensionFieldOperationTraits<
    mlir::prime_ir::field::ExtensionFieldCodeGen<N, BaseFieldT>> {
public:
  using BaseField = BaseFieldT;
  static constexpr size_t kDegree = N;
};

} // namespace zk_dtypes

namespace mlir::prime_ir::field {

// NOTE(chokobole): This class is not used directly. It is used to generate
// MLIR operations that implement extension field arithmetic. User should use
// FieldCodeGen instead.
template <size_t N, typename BaseFieldT>
class ExtensionFieldCodeGen
    : public ExtensionFieldOperationBase<
          N, BaseFieldT, ExtensionFieldCodeGen<N, BaseFieldT>>::Type,
      public VandermondeMatrix<ExtensionFieldCodeGen<N, BaseFieldT>>,
      public FrobeniusCoeffs<ExtensionFieldCodeGen<N, BaseFieldT>, N> {
  using Base = typename ExtensionFieldOperationBase<
      N, BaseFieldT, ExtensionFieldCodeGen<N, BaseFieldT>>::Type;

public:
  // Bring base class operator* into scope. Without this, the derived class's
  // operator*(const BaseFieldT&) hides all operator* overloads from Base.
  using Base::operator*;
  // TODO(junbeomlee): Remove these using declarations after refactoring
  // zk_dtypes to not define these methods in QuarticExtensionFieldOperation.
  using VandermondeMatrix<
      ExtensionFieldCodeGen<N, BaseFieldT>>::GetVandermondeInverseMatrix;
  using FrobeniusCoeffs<ExtensionFieldCodeGen<N, BaseFieldT>,
                        N>::GetFrobeniusCoeffs;

  ExtensionFieldCodeGen() = default;
  ExtensionFieldCodeGen(Value value, Value nonResidue)
      : value(value), nonResidue(nonResidue) {}
  ~ExtensionFieldCodeGen() = default;

  operator Value() const { return value; }

  // Accessors for mixin classes
  Type getType() const { return value.getType(); }

  ExtensionFieldCodeGen operator*(const BaseFieldT &scalar) const {
    std::array<BaseFieldT, N> coeffs = ToCoeffs();
    for (size_t i = 0; i < N; ++i) {
      coeffs[i] = coeffs[i] * scalar;
    }
    return FromCoeffs(coeffs);
  }

  // Compound assignment operators (needed when used as BaseField in tower)
  ExtensionFieldCodeGen &operator+=(const ExtensionFieldCodeGen &other) {
    *this = *this + other;
    return *this;
  }

  ExtensionFieldCodeGen &operator-=(const ExtensionFieldCodeGen &other) {
    *this = *this - other;
    return *this;
  }

  ExtensionFieldCodeGen &operator*=(const ExtensionFieldCodeGen &other) {
    *this = *this * other;
    return *this;
  }

  // Create [baseConst, 0, 0, ...] - embed base field element as scalar
  ExtensionFieldCodeGen fromScalarBaseField(const BaseFieldT &baseConst) const {
    std::array<BaseFieldT, N> coeffs;
    coeffs[0] = baseConst;
    for (size_t i = 1; i < N; ++i) {
      coeffs[i] = NonResidue().CreateConst(0);
    }
    return FromCoeffs(coeffs);
  }

  // Create a constant in this extension field (needed when used as BaseField)
  ExtensionFieldCodeGen CreateConst(int64_t x) const {
    return fromScalarBaseField(createConstBaseField(x));
  }

  // Create a rational constant (num/denom) in this extension field.
  // Returns [num/denom, 0, 0, ...] with the rational computed at compile-time.
  ExtensionFieldCodeGen CreateRationalConst(int64_t num, int64_t denom) const {
    return fromScalarBaseField(createRationalConstBaseField(num, denom));
  }

  // zk_dtypes ExtensionFieldOperation methods
  std::array<BaseFieldT, N> ToCoeffs() const {
    ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
    Operation::result_range coeffs = toCoeffs(*b, value);
    std::array<BaseFieldT, N> ret;
    for (size_t i = 0; i < N; ++i) {
      ret[i] = createBaseFieldCodeGen(coeffs[i]);
    }
    return ret;
  }

  ExtensionFieldCodeGen
  FromCoeffs(const std::array<BaseFieldT, N> &values) const {
    SmallVector<Value, N> coeffs;
    for (size_t i = 0; i < N; ++i) {
      coeffs.push_back(static_cast<Value>(values[i]));
    }
    ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
    return {fromCoeffs(*b, value.getType(), coeffs), nonResidue};
  }

  BaseFieldT NonResidue() const { return createBaseFieldCodeGen(nonResidue); }

  // Use ToomCook for quartic extensions (more efficient), Karatsuba otherwise.
  // TODO(junbeomlee): Consider using kCustom algorithm based on limb count
  // and non-residue value, which can be obtained from the type.
  static constexpr zk_dtypes::ExtensionFieldMulAlgorithm kDefaultAlgorithm =
      (N == 4) ? zk_dtypes::ExtensionFieldMulAlgorithm::kToomCook
               : zk_dtypes::ExtensionFieldMulAlgorithm::kKaratsuba;

  zk_dtypes::ExtensionFieldMulAlgorithm GetMulAlgorithm() const {
    return kDefaultAlgorithm;
  }

  zk_dtypes::ExtensionFieldMulAlgorithm GetSquareAlgorithm() const {
    return kDefaultAlgorithm;
  }

  // Create an integer constant in the base field.
  // Delegates to BaseFieldT::CreateConst() which handles the type-specific
  // logic.
  BaseFieldT createConstBaseField(int64_t x) const {
    return NonResidue().CreateConst(x);
  }

  // Create a rational constant (numerator/denominator) in the base field.
  // Delegates to BaseFieldT::CreateRationalConst() which computes num/denom
  // at compile-time and creates a single constant op.
  BaseFieldT createRationalConstBaseField(int64_t num, int64_t denom) const {
    return NonResidue().CreateRationalConst(num, denom);
  }

private:
  BaseFieldT createBaseFieldCodeGen(Value v) const {
    if constexpr (std::is_same_v<BaseFieldT, PrimeFieldCodeGen>) {
      return PrimeFieldCodeGen(v);
    } else {
      // For tower extensions, we need to get the non-residue for the base field
      ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
      auto extField = cast<ExtensionFieldType>(value.getType());
      auto baseEfType = cast<ExtensionFieldType>(extField.getBaseField());
      Value baseNonResidue = createNonResidueConstant(*b, baseEfType);
      return BaseFieldT(v, baseNonResidue);
    }
  }

  Value value;
  Value nonResidue;
};

// Explicit instantiations for non-tower extension fields
extern template class ExtensionFieldCodeGen<2, PrimeFieldCodeGen>;
extern template class ExtensionFieldCodeGen<3, PrimeFieldCodeGen>;
extern template class ExtensionFieldCodeGen<4, PrimeFieldCodeGen>;

// Type aliases for non-tower extension fields (backward compatible)
using QuadraticExtensionFieldCodeGen =
    ExtensionFieldCodeGen<2, PrimeFieldCodeGen>;
using CubicExtensionFieldCodeGen = ExtensionFieldCodeGen<3, PrimeFieldCodeGen>;
using QuarticExtensionFieldCodeGen =
    ExtensionFieldCodeGen<4, PrimeFieldCodeGen>;

// Type aliases for tower extension fields
// Tower over quadratic: e.g., Fp4 = (Fp2)^2, Fp6 = (Fp2)^3
using TowerQuadraticOverQuadraticCodeGen =
    ExtensionFieldCodeGen<2, QuadraticExtensionFieldCodeGen>;
using TowerCubicOverQuadraticCodeGen =
    ExtensionFieldCodeGen<3, QuadraticExtensionFieldCodeGen>;
using TowerQuarticOverQuadraticCodeGen =
    ExtensionFieldCodeGen<4, QuadraticExtensionFieldCodeGen>;

// Tower over cubic: e.g., Fp6 = (Fp3)^2
using TowerQuadraticOverCubicCodeGen =
    ExtensionFieldCodeGen<2, CubicExtensionFieldCodeGen>;

// Depth-2 towers (tower over tower)
// Fp12 = ((Fp2)^3)^2 = (Fp6)^2
using TowerQuadraticOverCubicOverQuadraticCodeGen =
    ExtensionFieldCodeGen<2, TowerCubicOverQuadraticCodeGen>;
// Fp12 = ((Fp2)^2)^3 = (Fp4)^3 (alternative construction)
using TowerCubicOverQuadraticOverQuadraticCodeGen =
    ExtensionFieldCodeGen<3, TowerQuadraticOverQuadraticCodeGen>;
// Fp8 = ((Fp2)^2)^2 = (Fp4)^2
using TowerQuadraticOverQuadraticOverQuadraticCodeGen =
    ExtensionFieldCodeGen<2, TowerQuadraticOverQuadraticCodeGen>;
// Fp12 = ((Fp3)^2)^2 = (Fp6)^2 (alternative)
using TowerQuadraticOverQuadraticOverCubicCodeGen =
    ExtensionFieldCodeGen<2, TowerQuadraticOverCubicCodeGen>;

// Depth-3 towers (for Fp24 = (Fp12)^2 = (((Fp2)^3)^2)^2)
using TowerQuadraticOverQuadraticOverCubicOverQuadraticCodeGen =
    ExtensionFieldCodeGen<2, TowerQuadraticOverCubicOverQuadraticCodeGen>;

} // namespace mlir::prime_ir::field

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_EXTENSIONFIELDCODEGEN_H_
