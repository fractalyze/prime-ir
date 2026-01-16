// Copyright 2026 The PrimeIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#ifndef PRIME_IR_DIALECT_FIELD_IR_EXTENSIONFIELDOPERATION_H_
#define PRIME_IR_DIALECT_FIELD_IR_EXTENSIONFIELDOPERATION_H_

#include <array>
#include <cassert>

#include "prime_ir/Dialect/Field/IR/ExtensionFieldOperationSelector.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Dialect/Field/IR/PrimeFieldOperation.h"
#include "prime_ir/Utils/Power.h"

namespace mlir::prime_ir::field {

template <size_t N>
class ExtensionFieldOperation;

} // namespace mlir::prime_ir::field

namespace zk_dtypes {
template <size_t N>
class ExtensionFieldOperationTraits<
    mlir::prime_ir::field::ExtensionFieldOperation<N>> {
public:
  // TODO(chokobole): Support towers of extension field.
  using BaseField = mlir::prime_ir::field::PrimeFieldOperation;
  static constexpr size_t kDegree = N;
};

} // namespace zk_dtypes

namespace mlir::prime_ir::field {

// Extension field operation class for compile-time evaluation.
// Inherits from zk_dtypes CRTP operations for Square/Inverse algorithms.
template <size_t N>
class ExtensionFieldOperation
    : public ExtensionFieldOperationSelector<N>::template Type<
          ExtensionFieldOperation<N>> {
public:
  ExtensionFieldOperation() = default;

  ExtensionFieldOperation(int64_t coeff, ExtensionFieldTypeInterface efType)
      : ExtensionFieldOperation(
            APInt(cast<PrimeFieldType>(efType.getBaseFieldType())
                      .getStorageBitWidth(),
                  coeff),
            efType) {}

  ExtensionFieldOperation(const APInt &coeff,
                          ExtensionFieldTypeInterface efType)
      : efType(efType) {
    auto baseFieldType = cast<PrimeFieldType>(efType.getBaseFieldType());
    coeffs[0] = PrimeFieldOperation(coeff, baseFieldType);
    for (size_t i = 1; i < N; ++i) {
      coeffs[i] = PrimeFieldOperation(uint64_t{0}, baseFieldType);
    }
  }

  // Construct from APInt coefficients (convenient one-liner usage)
  ExtensionFieldOperation(const SmallVector<APInt> &coeffs,
                          ExtensionFieldTypeInterface efType)
      : efType(efType) {
    assert(coeffs.size() == N);
    auto baseFieldType = cast<PrimeFieldType>(efType.getBaseFieldType());
    for (size_t i = 0; i < N; ++i) {
      this->coeffs[i] = PrimeFieldOperation(coeffs[i], baseFieldType);
    }
  }

  ExtensionFieldOperation(const std::array<PrimeFieldOperation, N> &coeffs,
                          ExtensionFieldTypeInterface efType)
      : coeffs(coeffs), efType(efType) {
    assert(coeffs.size() == N);
  }

  static ExtensionFieldOperation
  fromUnchecked(DenseIntElementsAttr attr, ExtensionFieldTypeInterface efType) {
    SmallVector<APInt> coeffs{attr.getValues<APInt>().begin(),
                              attr.getValues<APInt>().end()};
    return fromUnchecked(coeffs, efType);
  }

  static ExtensionFieldOperation
  fromUnchecked(const APInt &coeff, ExtensionFieldTypeInterface efType) {
    SmallVector<APInt> coeffs(
        N, APInt(cast<PrimeFieldType>(efType.getBaseFieldType())
                     .getStorageBitWidth(),
                 0));
    coeffs[0] = coeff;
    return fromUnchecked(coeffs, efType);
  }

  static ExtensionFieldOperation
  fromUnchecked(const SmallVector<APInt> &coeffs,
                ExtensionFieldTypeInterface efType) {
    assert(coeffs.size() == N);
    std::array<PrimeFieldOperation, N> newCoeffs;
    auto baseFieldType = cast<PrimeFieldType>(efType.getBaseFieldType());
    for (size_t i = 0; i < N; ++i) {
      newCoeffs[i] =
          PrimeFieldOperation::fromUnchecked(coeffs[i], baseFieldType);
    }
    return fromUnchecked(newCoeffs, efType);
  }

  static ExtensionFieldOperation
  fromUnchecked(const std::array<PrimeFieldOperation, N> &coeffs,
                ExtensionFieldTypeInterface efType) {
    ExtensionFieldOperation ret;
    ret.coeffs = coeffs;
    ret.efType = efType;
    return ret;
  }

  ExtensionFieldOperation getOne() const {
    return fromUnchecked(coeffs[0].getOne(), efType);
  }

  const std::array<PrimeFieldOperation, N> &getCoeffs() const { return coeffs; }

  // Convert coefficients to APInts (convenient for constant folding one-liners)
  operator SmallVector<APInt>() const {
    SmallVector<APInt> result;
    for (const auto &c : coeffs) {
      result.push_back(static_cast<APInt>(c));
    }
    return result;
  }

  DenseIntElementsAttr getDenseIntElementsAttr() const {
    return DenseIntElementsAttr::get(
        RankedTensorType::get(
            {N},
            cast<PrimeFieldType>(efType.getBaseFieldType()).getStorageType()),
        static_cast<SmallVector<APInt>>(*this));
  }

  ExtensionFieldOperation &operator+=(const ExtensionFieldOperation &other) {
    return *this = *this + other;
  }
  ExtensionFieldOperation &operator-=(const ExtensionFieldOperation &other) {
    return *this = *this - other;
  }
  ExtensionFieldOperation &operator*=(const ExtensionFieldOperation &other) {
    return *this = *this * other;
  }
  ExtensionFieldOperation &operator/=(const ExtensionFieldOperation &other) {
    return *this = *this / other;
  }

  ExtensionFieldOperation dbl() const { return this->Double(); }

  ExtensionFieldOperation square() const { return this->Square(); }

  ExtensionFieldOperation power(const APInt &exponent) const {
    return prime_ir::power(*this, exponent);
  }

  ExtensionFieldOperation inverse() const { return this->Inverse(); }

  bool operator==(const ExtensionFieldOperation &other) const {
    assert(efType == other.efType);
    return coeffs == other.coeffs;
  }
  bool operator!=(const ExtensionFieldOperation &other) const {
    assert(efType == other.efType);
    return coeffs != other.coeffs;
  }

private:
  friend class FieldOperation;
  // PascalCase methods (zk_dtypes compatible)
  template <typename>
  friend class zk_dtypes::QuadraticExtensionFieldOperation;
  template <typename>
  friend class zk_dtypes::CubicExtensionFieldOperation;
  template <typename>
  friend class zk_dtypes::QuarticExtensionFieldOperation;
  template <typename>
  friend class zk_dtypes::KaratsubaOperation;
  template <typename>
  friend class zk_dtypes::ToomCookOperation;
  template <typename>
  friend class zk_dtypes::ExtensionFieldOperation;
  template <typename>
  friend class zk_dtypes::FrobeniusOperation;
  template <typename, size_t>
  friend class zk_dtypes::VandermondeMatrix;

  template <size_t N2>
  friend raw_ostream &operator<<(raw_ostream &os,
                                 const ExtensionFieldOperation<N2> &op);

  const std::array<PrimeFieldOperation, N> &ToCoeffs() const { return coeffs; }

  ExtensionFieldOperation
  FromCoeffs(const std::array<PrimeFieldOperation, N> &c) const {
    return ExtensionFieldOperation(c, efType);
  }

  PrimeFieldOperation NonResidue() const {
    auto baseFieldType = cast<PrimeFieldType>(efType.getBaseFieldType());
    return PrimeFieldOperation::fromUnchecked(
        cast<IntegerAttr>(efType.getNonResidue()), baseFieldType);
  }

  // https://github.com/fractalyze/zk_dtypes/blob/8d5f43c/zk_dtypes/include/field/extension_field.h#L500
  zk_dtypes::ExtensionFieldMulAlgorithm GetSquareAlgorithm() const {
    if constexpr (N == 2) {
      auto baseFieldType = cast<PrimeFieldType>(efType.getBaseFieldType());
      // Heuristic: custom squaring when n² > 2n + C
      unsigned limbNums = (baseFieldType.getStorageBitWidth() + 63) / 64;
      if (limbNums * N >= 2) {
        return zk_dtypes::ExtensionFieldMulAlgorithm::kCustom2;
      }
      return zk_dtypes::ExtensionFieldMulAlgorithm::kKaratsuba;
    } else if constexpr (N == 3) {
      auto baseFieldType = cast<PrimeFieldType>(efType.getBaseFieldType());
      // Heuristic: custom squaring when n² > 4n
      unsigned limbNums = (baseFieldType.getStorageBitWidth() + 63) / 64;
      if (limbNums * N >= 4) {
        return zk_dtypes::ExtensionFieldMulAlgorithm::kCustom;
      }
      return zk_dtypes::ExtensionFieldMulAlgorithm::kKaratsuba;
    } else {
      // NOTE(chokobole): Avoid Toom-Cook algorithm due to caching conflicts.
      // `zk_dtypes::VandermondeMatrix` caches interpolation coefficients.
      // However, field types in `zk_dtypes` are distinct based on their
      // Montgomery property, while `ExtensionFieldOperation` are shared
      // across different domains.
      //
      // If a standard-domain extension field attempts to use a Vandermonde
      // matrix already cached in the Montgomery domain, the arithmetic will be
      // incorrect.
      return zk_dtypes::ExtensionFieldMulAlgorithm::kKaratsuba;
    }
  }

  zk_dtypes::ExtensionFieldMulAlgorithm GetMulAlgorithm() const {
    // NOTE(chokobole): See the comment above why we should not use Toom-Cook
    // algorithm.
    return zk_dtypes::ExtensionFieldMulAlgorithm::kKaratsuba;
  }

  PrimeFieldOperation CreateConstBaseField(int64_t x) const {
    auto baseFieldType = cast<PrimeFieldType>(efType.getBaseFieldType());
    return PrimeFieldOperation(x, baseFieldType);
  }

  // Frobenius coefficients: coeffs[e-1][i-1] = ξ^(i * e * (p - 1) / n)
  std::array<std::array<PrimeFieldOperation, N - 1>, N - 1>
  GetFrobeniusCoeffs() const {
    auto baseFieldType = cast<PrimeFieldType>(efType.getBaseFieldType());
    APInt p = baseFieldType.getModulus().getValue();
    APInt nr = cast<IntegerAttr>(efType.getNonResidue()).getValue();
    PrimeFieldOperation nrOp(nr, baseFieldType);

    unsigned extBitWidth = p.getBitWidth() * 2;
    APInt pMinus1 = (p - 1).zext(extBitWidth);
    APInt nVal(extBitWidth, N);

    std::array<std::array<PrimeFieldOperation, N - 1>, N - 1> result;
    for (size_t e = 1; e < N; ++e) {
      for (size_t i = 1; i < N; ++i) {
        APInt exp =
            APInt(extBitWidth, i) * APInt(extBitWidth, e) * pMinus1.udiv(nVal);
        result[e - 1][i - 1] = nrOp.power(exp);
      }
    }
    return result;
  }

  std::array<PrimeFieldOperation, N> coeffs;
  ExtensionFieldTypeInterface efType;
};

template <size_t N>
raw_ostream &operator<<(raw_ostream &os, const ExtensionFieldOperation<N> &op) {
  llvm::interleaveComma(op.coeffs, os,
                        [&](const PrimeFieldOperation &c) { os << c; });
  return os;
}

extern template class ExtensionFieldOperation<2>;
extern template class ExtensionFieldOperation<3>;
extern template class ExtensionFieldOperation<4>;

extern template raw_ostream &operator<<(raw_ostream &os,
                                        const ExtensionFieldOperation<2> &op);
extern template raw_ostream &operator<<(raw_ostream &os,
                                        const ExtensionFieldOperation<3> &op);
extern template raw_ostream &operator<<(raw_ostream &os,
                                        const ExtensionFieldOperation<4> &op);

using QuadraticExtensionFieldOperation = ExtensionFieldOperation<2>;
using CubicExtensionFieldOperation = ExtensionFieldOperation<3>;
using QuarticExtensionFieldOperation = ExtensionFieldOperation<4>;

} // namespace mlir::prime_ir::field

#endif // PRIME_IR_DIALECT_FIELD_IR_EXTENSIONFIELDOPERATION_H_
