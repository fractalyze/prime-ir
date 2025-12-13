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

#ifndef ZKIR_DIALECT_MODARITH_IR_MODARITHOPERATION_H_
#define ZKIR_DIALECT_MODARITH_IR_MODARITHOPERATION_H_

#include <cassert>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"

namespace mlir::zkir::mod_arith {

class ModArithOperation {
public:
  ModArithOperation(APInt value, ModArithType type)
      : value(value), type(type) {}
  ModArithOperation(IntegerAttr attr, ModArithType type)
      : value(attr.getValue()), type(type) {}

  operator APInt() const { return value; }

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
  ModArithOperation Double() const;
  ModArithOperation Square() const;
  ModArithOperation Power(APInt exponent) const;
  ModArithOperation Inverse() const;

  ModArithOperation FromMont() const;
  ModArithOperation ToMont() const;

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
      return FromMont().value.ult(other.FromMont().value);
    }
    return value.ult(other.value);
  }
  bool operator>(const ModArithOperation &other) const {
    assert(type == other.type);
    if (type.isMontgomery()) {
      return FromMont().value.ugt(other.FromMont().value);
    }
    return value.ugt(other.value);
  }
  bool operator<=(const ModArithOperation &other) const {
    assert(type == other.type);
    if (type.isMontgomery()) {
      return FromMont().value.ule(other.FromMont().value);
    }
    return value.ule(other.value);
  }
  bool operator>=(const ModArithOperation &other) const {
    assert(type == other.type);
    if (type.isMontgomery()) {
      return FromMont().value.uge(other.FromMont().value);
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

} // namespace mlir::zkir::mod_arith

#endif // ZKIR_DIALECT_MODARITH_IR_MODARITHOPERATION_H_
