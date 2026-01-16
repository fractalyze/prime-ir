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

#ifndef PRIME_IR_DIALECT_FIELD_IR_PRIMEFIELDOPERATION_H_
#define PRIME_IR_DIALECT_FIELD_IR_PRIMEFIELDOPERATION_H_

#include <cassert>

#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithOperation.h"

namespace zk_dtypes {

template <typename>
class QuadraticExtensionFieldOperation;
template <typename>
class CubicExtensionFieldOperation;
template <typename>
class QuarticExtensionFieldOperation;
template <typename>
class KaratsubaOperation;
template <typename>
class ToomCookOperation;
template <typename>
class ExtensionFieldOperation;

} // namespace zk_dtypes

namespace mlir::prime_ir::field {

class FieldOperation;

// PrimeFieldOperation wraps ModArithOperation for prime field arithmetic.
class PrimeFieldOperation {
public:
  PrimeFieldOperation() = default;
  PrimeFieldOperation(const APInt &value, PrimeFieldType type)
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
  static PrimeFieldOperation fromUnchecked(const APInt &value,
                                           PrimeFieldType type) {
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

  PrimeFieldOperation getZero() const {
    return PrimeFieldOperation::fromUnchecked(op.getZero(), type);
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
  PrimeFieldOperation power(const APInt &exponent) const {
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

  friend raw_ostream &operator<<(raw_ostream &os,
                                 const PrimeFieldOperation &op);

  PrimeFieldOperation Double() const { return dbl(); }
  PrimeFieldOperation Square() const { return square(); }
  // Returns Zero() if not invertible.
  PrimeFieldOperation Inverse() const { return inverse(); }
  PrimeFieldOperation CreateConst(int64_t constant) const {
    return PrimeFieldOperation(constant, type);
  }

  mod_arith::ModArithOperation op;
  PrimeFieldType type;
};

inline raw_ostream &operator<<(raw_ostream &os, const PrimeFieldOperation &op) {
  return os << op.op;
}

} // namespace mlir::prime_ir::field

#endif // PRIME_IR_DIALECT_FIELD_IR_PRIMEFIELDOPERATION_H_
