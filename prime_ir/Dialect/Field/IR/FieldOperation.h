// Copyright 2025 The PrimeIR Authors.
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

#ifndef PRIME_IR_DIALECT_FIELD_IR_FIELDOPERATION_H_
#define PRIME_IR_DIALECT_FIELD_IR_FIELDOPERATION_H_

#include <array>
#include <cassert>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "prime_ir/Dialect/Field/IR/ExtensionFieldOperationSelector.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithOperation.h"

namespace mlir::prime_ir::field {

class FieldOperation;

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

  static PrimeFieldOperation fromUnchecked(int64_t value, PrimeFieldType type) {
    return fromUnchecked(APInt(type.getStorageBitWidth(), value), type);
  }
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
  friend class FieldOperation;
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
    ExtensionFieldOperation ret;
    auto baseFieldType = cast<PrimeFieldType>(efType.getBaseFieldType());
    for (size_t i = 0; i < N; ++i) {
      ret.coeffs[i] =
          PrimeFieldOperation::fromUnchecked(coeffs[i], baseFieldType);
    }
    ret.efType = efType;
    return ret;
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

} // namespace mlir::prime_ir::field

namespace zk_dtypes {
template <size_t N>
class ExtensionFieldOperationTraits<
    mlir::prime_ir::field::ExtensionFieldOperation<N>> {
public:
  using BaseField = mlir::prime_ir::field::PrimeFieldOperation;
  static constexpr size_t kDegree = N;
};

} // namespace zk_dtypes

namespace mlir::prime_ir::field {

extern template class ExtensionFieldOperation<2>;
extern template class ExtensionFieldOperation<3>;
extern template class ExtensionFieldOperation<4>;

using QuadraticExtensionFieldOperation = ExtensionFieldOperation<2>;
using CubicExtensionFieldOperation = ExtensionFieldOperation<3>;
using QuarticExtensionFieldOperation = ExtensionFieldOperation<4>;

class FieldOperation {
public:
  using OperationType =
      std::variant<PrimeFieldOperation, QuadraticExtensionFieldOperation,
                   CubicExtensionFieldOperation,
                   QuarticExtensionFieldOperation>;

  FieldOperation() = default;

  template <typename T, typename = std::enable_if_t<
                            !std::is_same_v<std::decay_t<T>, FieldOperation> &&
                            std::is_constructible_v<OperationType, T>>>
  FieldOperation(T &&operation) // NOLINT(runtime/explicit)
      : operation(std::forward<T>(operation)) {}

  template <typename T>
  FieldOperation(T &&value, Type type) {
    if (auto pfType = dyn_cast<PrimeFieldType>(type)) {
      operation = PrimeFieldOperation(std::forward<T>(value), pfType);
      return;
    }
    createExtFieldOp(std::forward<T>(value),
                     cast<ExtensionFieldTypeInterface>(type));
  }
  FieldOperation(const SmallVector<APInt> &coeffs, Type type) {
    createExtFieldOp(coeffs, cast<ExtensionFieldTypeInterface>(type));
  }
  ~FieldOperation() = default;

  template <typename T>
  static FieldOperation fromUnchecked(T &&value, Type type) {
    FieldOperation ret;
    if (auto pfType = dyn_cast<PrimeFieldType>(type)) {
      ret.operation =
          PrimeFieldOperation::fromUnchecked(std::forward<T>(value), pfType);
      return ret;
    }
    ret.createRawExtFieldOp(std::forward<T>(value),
                            cast<ExtensionFieldTypeInterface>(type));
    return ret;
  }

  static FieldOperation fromUnchecked(const SmallVector<APInt> &coeffs,
                                      Type type) {
    FieldOperation ret;
    ret.createRawExtFieldOp(coeffs, cast<ExtensionFieldTypeInterface>(type));
    return ret;
  }

  operator APInt() const;
  operator SmallVector<APInt>() const;

  const OperationType &getOperation() const { return operation; }

  FieldOperation operator+(const FieldOperation &other) const;
  FieldOperation operator-(const FieldOperation &other) const;
  FieldOperation operator*(const FieldOperation &other) const;
  FieldOperation operator-() const;
  FieldOperation dbl() const;
  FieldOperation square() const;
  FieldOperation inverse() const;

private:
  template <typename T>
  void createExtFieldOp(T &&value, ExtensionFieldTypeInterface efType) {
    TypeSwitch<Type>(efType)
        .template Case<QuadraticExtFieldType>([&](auto) {
          operation =
              QuadraticExtensionFieldOperation(std::forward<T>(value), efType);
        })
        .template Case<CubicExtFieldType>([&](auto) {
          operation =
              CubicExtensionFieldOperation(std::forward<T>(value), efType);
        })
        .template Case<QuarticExtFieldType>([&](auto) {
          operation =
              QuarticExtensionFieldOperation(std::forward<T>(value), efType);
        })
        .Default(
            [](auto) { llvm_unreachable("Unsupported extension field type"); });
  }

  template <typename T>
  void createRawExtFieldOp(T &&value, ExtensionFieldTypeInterface efType) {
    TypeSwitch<Type>(efType)
        .template Case<QuadraticExtFieldType>([&](auto) {
          operation = QuadraticExtensionFieldOperation::fromUnchecked(
              std::forward<T>(value), efType);
        })
        .template Case<CubicExtFieldType>([&](auto) {
          operation = CubicExtensionFieldOperation::fromUnchecked(
              std::forward<T>(value), efType);
        })
        .template Case<QuarticExtFieldType>([&](auto) {
          operation = QuarticExtensionFieldOperation::fromUnchecked(
              std::forward<T>(value), efType);
        })
        .Default(
            [](auto) { llvm_unreachable("Unsupported extension field type"); });
  }

  OperationType operation;
};

} // namespace mlir::prime_ir::field

#endif // PRIME_IR_DIALECT_FIELD_IR_FIELDOPERATION_H_
