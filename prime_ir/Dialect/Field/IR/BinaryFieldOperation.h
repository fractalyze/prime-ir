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

#ifndef PRIME_IR_DIALECT_FIELD_IR_BINARYFIELDOPERATION_H_
#define PRIME_IR_DIALECT_FIELD_IR_BINARYFIELDOPERATION_H_

#include <cassert>
#include <cstdint>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Utils/ZkDtypes.h"
#include "zk_dtypes/include/field/binary_field_multiplication.h"

namespace mlir::prime_ir::field {

class FieldOperation;

// BinaryFieldOperation provides arithmetic operations for binary tower fields.
//
// Binary tower fields are GF(2^(2^level)) constructed using degree-2
// extensions at each level. Operations use XOR-based arithmetic:
// - Addition and subtraction are XOR
// - Negation is identity
// - Doubling always yields zero
// - Multiplication uses Karatsuba algorithm with tower structure
class BinaryFieldOperation {
public:
  BinaryFieldOperation() = default;

  BinaryFieldOperation(const llvm::APInt &value, BinaryFieldType type)
      : value_(value.zextOrTrunc(type.getBitWidth())), type_(type) {}

  BinaryFieldOperation(uint64_t value, BinaryFieldType type)
      : value_(llvm::APInt(type.getBitWidth(), value)), type_(type) {}

  BinaryFieldOperation(IntegerAttr attr, BinaryFieldType type)
      : value_(attr.getValue().zextOrTrunc(type.getBitWidth())), type_(type) {}

  // Create from unchecked value (no reduction needed for binary fields).
  static BinaryFieldOperation fromUnchecked(int64_t value,
                                            BinaryFieldType type) {
    return fromUnchecked(
        llvm::APInt(type.getBitWidth(), static_cast<uint64_t>(value)), type);
  }

  static BinaryFieldOperation fromUnchecked(const llvm::APInt &value,
                                            BinaryFieldType type) {
    BinaryFieldOperation ret;
    ret.value_ = value.zextOrTrunc(type.getBitWidth());
    ret.type_ = type;
    return ret;
  }

  static BinaryFieldOperation fromUnchecked(IntegerAttr attr,
                                            BinaryFieldType type) {
    return fromUnchecked(attr.getValue(), type);
  }

  template <typename F>
  static BinaryFieldType getBinaryFieldType(MLIRContext *context) {
    return BinaryFieldType::get(context, F::kTowerLevel);
  }

  template <typename F>
  static BinaryFieldOperation fromZkDtype(MLIRContext *context, const F &bf) {
    return fromUnchecked(convertToAPInt(bf.value()),
                         getBinaryFieldType<F>(context));
  }

  // Get zero element.
  BinaryFieldOperation getZero() const {
    return BinaryFieldOperation::fromUnchecked(
        llvm::APInt::getZero(type_.getBitWidth()), type_);
  }

  // Get one element (multiplicative identity).
  BinaryFieldOperation getOne() const {
    return BinaryFieldOperation::fromUnchecked(
        llvm::APInt(type_.getBitWidth(), 1), type_);
  }

  // Convert to APInt.
  operator llvm::APInt() const { return value_; }

  // Get as IntegerAttr.
  IntegerAttr getIntegerAttr() const {
    return IntegerAttr::get(type_.getStorageType(), value_);
  }

  // Get the type.
  Type getType() const { return type_; }

  // Get the tower level.
  unsigned getTowerLevel() const { return type_.getTowerLevel(); }

  //===--------------------------------------------------------------------===//
  // Arithmetic Operations
  //===--------------------------------------------------------------------===//

  // Addition in binary field is XOR.
  BinaryFieldOperation operator+(const BinaryFieldOperation &other) const {
    assert(type_ == other.type_);
    return BinaryFieldOperation::fromUnchecked(value_ ^ other.value_, type_);
  }

  BinaryFieldOperation &operator+=(const BinaryFieldOperation &other) {
    assert(type_ == other.type_);
    value_ ^= other.value_;
    return *this;
  }

  // Subtraction in binary field is XOR (same as addition in characteristic 2).
  BinaryFieldOperation operator-(const BinaryFieldOperation &other) const {
    return *this + other;
  }

  BinaryFieldOperation &operator-=(const BinaryFieldOperation &other) {
    return *this += other;
  }

  // Multiplication using tower field structure.
  BinaryFieldOperation operator*(const BinaryFieldOperation &other) const;

  BinaryFieldOperation &operator*=(const BinaryFieldOperation &other) {
    *this = *this * other;
    return *this;
  }

  // Division (multiply by inverse).
  BinaryFieldOperation operator/(const BinaryFieldOperation &other) const {
    return *this * other.inverse();
  }

  BinaryFieldOperation &operator/=(const BinaryFieldOperation &other) {
    *this = *this / other;
    return *this;
  }

  // Negation in binary field is identity (-a = a in characteristic 2).
  BinaryFieldOperation operator-() const { return *this; }

  // Doubling in binary field is always zero (2a = 0 in characteristic 2).
  BinaryFieldOperation dbl() const { return getZero(); }

  // Squaring using tower field structure.
  BinaryFieldOperation square() const;

  // Power operation using repeated squaring.
  BinaryFieldOperation power(const llvm::APInt &exponent) const;

  // Multiplicative inverse using a^(2ⁿ - 2).
  BinaryFieldOperation inverse() const;

  BinaryFieldOperation fromMont() const { return *this; }
  BinaryFieldOperation toMont() const { return *this; }

  //===--------------------------------------------------------------------===//
  // Comparison Operations
  //===--------------------------------------------------------------------===//

  bool isOne() const { return value_ == 1; }
  bool isZero() const { return value_.isZero(); }

  bool operator==(const BinaryFieldOperation &other) const {
    assert(type_ == other.type_);
    return value_ == other.value_;
  }

  bool operator!=(const BinaryFieldOperation &other) const {
    return !(*this == other);
  }

  bool operator<(const BinaryFieldOperation &other) const {
    assert(type_ == other.type_);
    return value_.ult(other.value_);
  }

  bool operator>(const BinaryFieldOperation &other) const {
    assert(type_ == other.type_);
    return value_.ugt(other.value_);
  }

  bool operator<=(const BinaryFieldOperation &other) const {
    assert(type_ == other.type_);
    return value_.ule(other.value_);
  }

  bool operator>=(const BinaryFieldOperation &other) const {
    assert(type_ == other.type_);
    return value_.uge(other.value_);
  }

  llvm::SmallString<40> toString() const {
    llvm::SmallString<40> result;
    value_.toStringUnsigned(result, 10);
    return result;
  }

private:
  friend class FieldOperation;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const BinaryFieldOperation &op);

  // PascalCase methods for compatibility with zk_dtypes patterns
  BinaryFieldOperation Double() const { return dbl(); }
  BinaryFieldOperation Square() const { return square(); }
  BinaryFieldOperation Inverse() const { return inverse(); }
  BinaryFieldOperation CreateConst(int64_t constant) const {
    return BinaryFieldOperation(constant, type_);
  }

  llvm::APInt value_;
  BinaryFieldType type_;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const BinaryFieldOperation &op) {
  return os << op.value_;
}

} // namespace mlir::prime_ir::field

#endif // PRIME_IR_DIALECT_FIELD_IR_BINARYFIELDOPERATION_H_
