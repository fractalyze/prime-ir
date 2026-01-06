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

#include "absl/status/statusor.h"
#include "llvm/Support/ErrorHandling.h"
#include "zk_dtypes/include/field/cubic_extension_field_operation.h"
#include "zk_dtypes/include/field/quadratic_extension_field_operation.h"
#include "zk_dtypes/include/field/quartic_extension_field_operation.h"
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
  absl::StatusOr<PrimeFieldOperation> Inverse() const {
    if (op.isZero())
      return absl::InvalidArgumentError("Cannot invert zero");
    return inverse();
  }
  // Prime field has extension degree 1 (used by Frobenius)
  static constexpr size_t ExtensionDegree() { return 1; }

  friend raw_ostream &operator<<(raw_ostream &os,
                                 const PrimeFieldOperation &op);

  mod_arith::ModArithOperation op;
  field::PrimeFieldType type;
};

inline raw_ostream &operator<<(raw_ostream &os, const PrimeFieldOperation &op) {
  return os << op.op;
}

// Selects the appropriate zk_dtypes extension field operation based on degree.
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

// Extension field operation class for compile-time evaluation.
// Inherits from zk_dtypes CRTP operations for Square/Inverse algorithms.
template <size_t N>
class ExtensionFieldOperation
    : public ExtensionFieldOperationSelector<N>::template Type<
          ExtensionFieldOperation<N>> {
  using Base = typename ExtensionFieldOperationSelector<N>::template Type<
      ExtensionFieldOperation<N>>;

public:
  using Base::operator*;

  // Construct from APInt coefficients (convenient one-liner usage)
  ExtensionFieldOperation(const SmallVector<APInt> &coeffs,
                          const APInt &nonResidue, PrimeFieldType baseFieldType)
      : baseFieldType(baseFieldType) {
    assert(coeffs.size() == N);
    for (size_t i = 0; i < N; ++i) {
      this->coeffs[i] = PrimeFieldOperation(coeffs[i], baseFieldType);
    }
    this->nonResidue = PrimeFieldOperation(nonResidue, baseFieldType);
  }

  // Construct from PrimeFieldOperation coefficients
  ExtensionFieldOperation(const std::array<PrimeFieldOperation, N> &coeffs,
                          const PrimeFieldOperation &nonResidue,
                          PrimeFieldType baseFieldType)
      : coeffs(coeffs), nonResidue(nonResidue), baseFieldType(baseFieldType) {}

  const std::array<PrimeFieldOperation, N> &getCoeffs() const { return coeffs; }

  // Convert coefficients to APInts (convenient for constant folding one-liners)
  SmallVector<APInt> toAPInts() const {
    SmallVector<APInt> result;
    for (const auto &c : coeffs) {
      result.push_back(static_cast<APInt>(c));
    }
    return result;
  }

  // TODO(junbeomlee): Add scalar multiplication (ExtensionField * BaseField)
  // to zk_dtypes and reuse here.
  ExtensionFieldOperation operator*(const PrimeFieldOperation &scalar) const {
    std::array<PrimeFieldOperation, N> result;
    for (size_t i = 0; i < N; ++i) {
      result[i] = coeffs[i] * scalar;
    }
    return ExtensionFieldOperation(result, nonResidue, baseFieldType);
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

  const std::array<PrimeFieldOperation, N> &ToBaseFields() const {
    return coeffs;
  }

  ExtensionFieldOperation
  FromBaseFields(const std::array<PrimeFieldOperation, N> &c) const {
    return ExtensionFieldOperation(c, nonResidue, baseFieldType);
  }

  PrimeFieldOperation NonResidue() const { return nonResidue; }

  // TODO(junbeomlee): Refactor zk_dtypes C() and C2() macros into reusable
  // functions that accept CreateConstBaseField, then implement
  // GetVandermondeInverseMatrix and apply algorithm selection logic based on
  // limb count and non-residue value (similar to ExtensionFieldCodeGen).
  zk_dtypes::ExtensionFieldMulAlgorithm GetSquareAlgorithm() const {
    return zk_dtypes::ExtensionFieldMulAlgorithm::kKaratsuba;
  }

  zk_dtypes::ExtensionFieldMulAlgorithm GetMulAlgorithm() const {
    return zk_dtypes::ExtensionFieldMulAlgorithm::kKaratsuba;
  }

  PrimeFieldOperation CreateZeroBaseField() const {
    APInt zero(baseFieldType.getStorageBitWidth(), 0);
    return PrimeFieldOperation(zero, baseFieldType);
  }

  // Frobenius coefficients: coeffs[e-1][i-1] = ξ^(i * e * (p - 1) / n)
  std::array<std::array<PrimeFieldOperation, N - 1>, N - 1>
  GetFrobeniusCoeffs() const {
    APInt p = baseFieldType.getModulus().getValue();
    APInt nr = static_cast<APInt>(nonResidue);
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

  // Required by ToomCookOperation but never called (we use kKaratsuba).
  std::array<std::array<PrimeFieldOperation, 7>, 7>
  GetVandermondeInverseMatrix() const {
    llvm_unreachable("GetVandermondeInverseMatrix should not be called");
  }

  std::array<PrimeFieldOperation, N> coeffs;
  PrimeFieldOperation nonResidue;
  PrimeFieldType baseFieldType;
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
