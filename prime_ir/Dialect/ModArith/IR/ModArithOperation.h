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

#ifndef PRIME_IR_DIALECT_MODARITH_IR_MODARITHOPERATION_H_
#define PRIME_IR_DIALECT_MODARITH_IR_MODARITHOPERATION_H_

#include <cassert>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"

namespace mlir::prime_ir::mod_arith {

class ModArithOperation {
public:
  ModArithOperation() = default;
  ModArithOperation(APInt value, ModArithType type) : value(value), type(type) {
    assert(value.ult(type.getModulus().getValue()) && "value must less than P");
    if (type.isMontgomery()) {
      MontgomeryAttr montAttr = type.getMontgomeryAttr();
      operator*=(ModArithOperation::fromUnchecked(
          montAttr.getRSquared().getValue(), type));
    }
  }
  ModArithOperation(IntegerAttr attr, ModArithType type)
      : ModArithOperation(attr.getValue(), type) {}
  ModArithOperation(int64_t value, ModArithType type)
      : ModArithOperation(static_cast<uint64_t>(value < 0 ? -value : value),
                          type) {
    if (value < 0) {
      *this = -(*this);
    }
  }
  ModArithOperation(uint64_t value, ModArithType type)
      : ModArithOperation(APInt(type.getStorageBitWidth(), value), type) {}

  static ModArithOperation fromUnchecked(APInt value, ModArithType type) {
    ModArithOperation ret;
    ret.value = value;
    ret.type = type;
    return ret;
  }

  ModArithOperation getOne() const;

  operator APInt() const { return value; }
  IntegerAttr getIntegerAttr() const {
    return IntegerAttr::get(type.getStorageType(), value);
  }

  ModArithOperation operator+(const ModArithOperation &other) const;
  ModArithOperation &operator+=(const ModArithOperation &other) {
    return *this = *this + other;
  }
  ModArithOperation operator-(const ModArithOperation &other) const;
  ModArithOperation &operator-=(const ModArithOperation &other) {
    return *this = *this - other;
  }
  ModArithOperation operator*(const ModArithOperation &other) const;
  ModArithOperation &operator*=(const ModArithOperation &other) {
    return *this = *this * other;
  }
  ModArithOperation operator/(const ModArithOperation &other) const;
  ModArithOperation &operator/=(const ModArithOperation &other) {
    return *this = *this / other;
  }

  ModArithOperation operator-() const;
  ModArithOperation dbl() const;
  ModArithOperation square() const;
  ModArithOperation power(APInt exponent) const;
  ModArithOperation inverse() const;

  ModArithOperation fromMont() const;
  ModArithOperation toMont() const;

  bool isOne() const;
  bool isZero() const;

  bool operator==(const ModArithOperation &other) const {
    assert(type == other.type);
    return value.eq(other.value);
  }
  bool operator!=(const ModArithOperation &other) const {
    assert(type == other.type);
    return value.ne(other.value);
  }
  bool operator<(const ModArithOperation &other) const {
    assert(type == other.type);
    if (type.isMontgomery()) {
      return fromMont().value.ult(other.fromMont().value);
    }
    return value.ult(other.value);
  }
  bool operator>(const ModArithOperation &other) const {
    assert(type == other.type);
    if (type.isMontgomery()) {
      return fromMont().value.ugt(other.fromMont().value);
    }
    return value.ugt(other.value);
  }
  bool operator<=(const ModArithOperation &other) const {
    assert(type == other.type);
    if (type.isMontgomery()) {
      return fromMont().value.ule(other.fromMont().value);
    }
    return value.ule(other.value);
  }
  bool operator>=(const ModArithOperation &other) const {
    assert(type == other.type);
    if (type.isMontgomery()) {
      return fromMont().value.uge(other.fromMont().value);
    }
    return value.uge(other.value);
  }

  llvm::SmallString<40> toString() const;

private:
  friend raw_ostream &operator<<(raw_ostream &os, const ModArithOperation &op);

  APInt value;
  ModArithType type;
};

raw_ostream &operator<<(raw_ostream &os, const ModArithOperation &op);

} // namespace mlir::prime_ir::mod_arith

#endif // PRIME_IR_DIALECT_MODARITH_IR_MODARITHOPERATION_H_
