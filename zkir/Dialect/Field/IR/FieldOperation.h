// Copyright 2025 The ZKIR Authors.
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

#ifndef ZKIR_DIALECT_FIELD_IR_FIELDOPERATION_H_
#define ZKIR_DIALECT_FIELD_IR_FIELDOPERATION_H_

#include <array>
#include <cassert>

#include "llvm/Support/ErrorHandling.h"
#include "zkir/Dialect/Field/IR/ExtensionFieldOperationSelector.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"
#include "zkir/Dialect/ModArith/IR/ModArithOperation.h"

namespace mlir::zkir::field {

// PrimeFieldOperation wraps ModArithOperation for prime field arithmetic.
class PrimeFieldOperation {
public:
  PrimeFieldOperation() = default;
  PrimeFieldOperation(APInt value, PrimeFieldType type)
      : op(value, convertPrimeFieldType(type)), type(type) {}
  PrimeFieldOperation(IntegerAttr attr, PrimeFieldType type)
      : op(attr, convertPrimeFieldType(type)), type(type) {}
  PrimeFieldOperation(int64_t value, PrimeFieldType type)
      : op(value, convertPrimeFieldType(type)), type(type) {}
  PrimeFieldOperation(uint64_t value, PrimeFieldType type)
      : op(value, convertPrimeFieldType(type)), type(type) {}

  static PrimeFieldOperation fromUnchecked(APInt value, PrimeFieldType type) {
    return fromUnchecked(mod_arith::ModArithOperation::fromUnchecked(
                             value, convertPrimeFieldType(type)),
                         type);
  }
  static PrimeFieldOperation fromUnchecked(IntegerAttr attr,
                                           PrimeFieldType type) {
    return fromUnchecked(attr.getValue(), type);
  }
  static PrimeFieldOperation fromUnchecked(mod_arith::ModArithOperation op,
                                           PrimeFieldType type) {
    PrimeFieldOperation ret;
    ret.op = op;
    ret.type = type;
    return ret;
  }

  PrimeFieldOperation getOne() const {
    return PrimeFieldOperation::fromUnchecked(op.getOne(), type);
  }

  operator APInt() const { return static_cast<APInt>(op); }
  IntegerAttr getIntegerAttr() const {
    return IntegerAttr::get(type.getStorageType(), op);
  }

  PrimeFieldOperation operator+(const PrimeFieldOperation &other) const {
    assert(type == other.type);
    return PrimeFieldOperation::fromUnchecked(op + other.op, type);
  }
  PrimeFieldOperation &operator+=(const PrimeFieldOperation &other) {
    assert(type == other.type);
    op += other.op;
    return *this;
  }
  PrimeFieldOperation operator-(const PrimeFieldOperation &other) const {
    assert(type == other.type);
    return PrimeFieldOperation::fromUnchecked(op - other.op, type);
  }
  PrimeFieldOperation &operator-=(const PrimeFieldOperation &other) {
    assert(type == other.type);
    op -= other.op;
    return *this;
  }
  PrimeFieldOperation operator*(const PrimeFieldOperation &other) const {
    assert(type == other.type);
    return PrimeFieldOperation::fromUnchecked(op * other.op, type);
  }
  PrimeFieldOperation &operator*=(const PrimeFieldOperation &other) {
    assert(type == other.type);
    op *= other.op;
    return *this;
  }
  PrimeFieldOperation operator/(const PrimeFieldOperation &other) const {
    assert(type == other.type);
    return PrimeFieldOperation::fromUnchecked(op / other.op, type);
  }
  PrimeFieldOperation &operator/=(const PrimeFieldOperation &other) {
    assert(type == other.type);
    op /= other.op;
    return *this;
  }
  PrimeFieldOperation operator-() const {
    return PrimeFieldOperation::fromUnchecked(-op, type);
  }

  PrimeFieldOperation dbl() const {
    return PrimeFieldOperation::fromUnchecked(op.dbl(), type);
  }
  PrimeFieldOperation square() const {
    return PrimeFieldOperation::fromUnchecked(op.square(), type);
  }
  PrimeFieldOperation power(APInt exponent) const {
    return PrimeFieldOperation::fromUnchecked(op.power(exponent), type);
  }
  PrimeFieldOperation inverse() const {
    return PrimeFieldOperation::fromUnchecked(op.inverse(), type);
  }
  PrimeFieldOperation fromMont() const {
    assert(type.isMontgomery());
    auto stdType =
        PrimeFieldType::get(type.getContext(), type.getModulus(), false);
    return PrimeFieldOperation::fromUnchecked(op.fromMont(), stdType);
  }
  PrimeFieldOperation toMont() const {
    assert(!type.isMontgomery());
    auto montType =
        PrimeFieldType::get(type.getContext(), type.getModulus(), true);
    return PrimeFieldOperation::fromUnchecked(op.toMont(), montType);
  }

  bool isOne() const { return op.isOne(); }
  bool isZero() const { return op.isZero(); }
  bool operator==(const PrimeFieldOperation &other) const {
    assert(type == other.type);
    return op == other.op;
  }
  bool operator!=(const PrimeFieldOperation &other) const {
    assert(type == other.type);
    return op != other.op;
  }
  bool operator<(const PrimeFieldOperation &other) const {
    assert(type == other.type);
    return op < other.op;
  }
  bool operator>(const PrimeFieldOperation &other) const {
    assert(type == other.type);
    return op > other.op;
  }
  bool operator<=(const PrimeFieldOperation &other) const {
    assert(type == other.type);
    return op <= other.op;
  }
  bool operator>=(const PrimeFieldOperation &other) const {
    assert(type == other.type);
    return op >= other.op;
  }
  llvm::SmallString<40> toString() const { return op.toString(); }

private:
  // PascalCase methods (zk_dtypes compatible)
  template <size_t>
  friend class ExtensionFieldOperation;
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

  PrimeFieldOperation Double() const { return dbl(); }
  PrimeFieldOperation Square() const { return square(); }
  // Returns Zero() if not invertible.
  PrimeFieldOperation Inverse() const { return inverse(); }
  PrimeFieldOperation CreateConst(int64_t constant) const {
    return PrimeFieldOperation(constant, type);
  }
  // Prime field has extension degree 1 (used by Frobenius)
  static constexpr size_t ExtensionDegree() { return 1; }

  friend raw_ostream &operator<<(raw_ostream &os,
                                 const PrimeFieldOperation &op);

  mod_arith::ModArithOperation op;
  PrimeFieldType type;
};

inline raw_ostream &operator<<(raw_ostream &os, const PrimeFieldOperation &op) {
  return os << op.op;
}

// Extension field operation class for compile-time evaluation.
// Inherits from zk_dtypes CRTP operations for Square/Inverse algorithms.
template <size_t N>
class ExtensionFieldOperation
    : public ExtensionFieldOperationSelector<N>::template Type<
          ExtensionFieldOperation<N>> {
public:
  ExtensionFieldOperation(int64_t coeff, ExtensionFieldTypeInterface efType)
      : efType(efType) {
    auto baseFieldType = cast<PrimeFieldType>(efType.getBaseFieldType());
    coeffs[0] = PrimeFieldOperation(coeff, baseFieldType);
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

  const std::array<PrimeFieldOperation, N> &getCoeffs() const { return coeffs; }

  // Convert coefficients to APInts (convenient for constant folding one-liners)
  SmallVector<APInt> toAPInts() const {
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
        toAPInts());
  }

private:
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

  const std::array<PrimeFieldOperation, N> &ToCoeffs() const { return coeffs; }

  ExtensionFieldOperation
  FromCoeffs(const std::array<PrimeFieldOperation, N> &c) const {
    return ExtensionFieldOperation(c, efType);
  }

  PrimeFieldOperation NonResidue() const {
    auto baseFieldType = cast<PrimeFieldType>(efType.getBaseFieldType());
    return PrimeFieldOperation(cast<IntegerAttr>(efType.getNonResidue()),
                               baseFieldType);
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
      return zk_dtypes::ExtensionFieldMulAlgorithm::kToomCook;
    }
  }

  zk_dtypes::ExtensionFieldMulAlgorithm GetMulAlgorithm() const {
    if constexpr (N == 4) {
      return zk_dtypes::ExtensionFieldMulAlgorithm::kToomCook;
    }
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

} // namespace mlir::zkir::field

namespace zk_dtypes {
template <size_t N>
class ExtensionFieldOperationTraits<
    mlir::zkir::field::ExtensionFieldOperation<N>> {
public:
  using BaseField = mlir::zkir::field::PrimeFieldOperation;
  static constexpr size_t kDegree = N;
};

} // namespace zk_dtypes

#endif // ZKIR_DIALECT_FIELD_IR_FIELDOPERATION_H_
