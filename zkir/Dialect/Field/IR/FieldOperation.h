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

#include <cassert>

#include "zkir/Dialect/Field/IR/FieldTypes.h"
#include "zkir/Dialect/ModArith/IR/ModArithOperation.h"

namespace mlir::zkir::field {

class PrimeFieldOperation {
public:
  PrimeFieldOperation(APInt value, PrimeFieldType type)
      : op(value, convertPrimeFieldType(type)), type(type) {}
  PrimeFieldOperation(IntegerAttr attr, PrimeFieldType type)
      : op(attr, convertPrimeFieldType(type)), type(type) {}

  PrimeFieldOperation getOne() const {
    return PrimeFieldOperation(op.getOne(), type);
  }

  operator APInt() const { return static_cast<APInt>(op); }
  IntegerAttr getIntegerAttr() const {
    return IntegerAttr::get(type.getStorageType(), op);
  }

  PrimeFieldOperation operator+(const PrimeFieldOperation &other) const {
    assert(type == other.type);
    return PrimeFieldOperation(op + other.op, type);
  }
  PrimeFieldOperation &operator+=(const PrimeFieldOperation &other) {
    assert(type == other.type);
    op += other.op;
    return *this;
  }
  PrimeFieldOperation operator-(const PrimeFieldOperation &other) const {
    assert(type == other.type);
    return PrimeFieldOperation(op - other.op, type);
  }
  PrimeFieldOperation &operator-=(const PrimeFieldOperation &other) {
    assert(type == other.type);
    op -= other.op;
    return *this;
  }
  PrimeFieldOperation operator*(const PrimeFieldOperation &other) const {
    assert(type == other.type);
    return PrimeFieldOperation(op * other.op, type);
  }
  PrimeFieldOperation &operator*=(const PrimeFieldOperation &other) {
    assert(type == other.type);
    op *= other.op;
    return *this;
  }
  PrimeFieldOperation operator/(const PrimeFieldOperation &other) const {
    assert(type == other.type);
    return PrimeFieldOperation(op / other.op, type);
  }
  PrimeFieldOperation &operator/=(const PrimeFieldOperation &other) {
    assert(type == other.type);
    op /= other.op;
    return *this;
  }
  PrimeFieldOperation operator-() const {
    return PrimeFieldOperation(-op, type);
  }
  PrimeFieldOperation dbl() const {
    return PrimeFieldOperation(op.dbl(), type);
  }
  PrimeFieldOperation square() const {
    return PrimeFieldOperation(op.square(), type);
  }
  PrimeFieldOperation power(APInt exponent) const {
    return PrimeFieldOperation(op.power(exponent), type);
  }
  PrimeFieldOperation inverse() const {
    return PrimeFieldOperation(op.inverse(), type);
  }
  PrimeFieldOperation fromMont() const {
    return PrimeFieldOperation(op.fromMont(), type);
  }
  PrimeFieldOperation toMont() const {
    return PrimeFieldOperation(op.toMont(), type);
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
  friend raw_ostream &operator<<(raw_ostream &os,
                                 const PrimeFieldOperation &op);

  mod_arith::ModArithOperation op;
  field::PrimeFieldType type;
};

inline raw_ostream &operator<<(raw_ostream &os, const PrimeFieldOperation &op) {
  return os << op.op;
}

} // namespace mlir::zkir::field

#endif // ZKIR_DIALECT_FIELD_IR_FIELDOPERATION_H_
