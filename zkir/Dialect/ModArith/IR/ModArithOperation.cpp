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

#include "zkir/Dialect/ModArith/IR/ModArithOperation.h"

#include "zkir/Utils/APIntUtils.h"

namespace mlir::zkir::mod_arith {

ModArithOperation
ModArithOperation::operator+(const ModArithOperation &other) const {
  assert(type == other.type);

  const APInt &a = value;
  const APInt &b = other.value;
  APInt modulus = type.getModulus().getValue();

  unsigned bitWidth = modulus.getBitWidth();
  if (bitWidth > modulus.getActiveBits()) {
    APInt add = a + b;
    if (add.uge(modulus)) {
      add -= modulus;
    }
    return ModArithOperation(add, type);
  }

  APInt extA = a.zext(bitWidth + 1);
  APInt extB = b.zext(bitWidth + 1);
  APInt extModulus = modulus.zext(bitWidth + 1);
  APInt extAdd = extA + extB;
  if (extAdd.uge(extModulus)) {
    extAdd -= extModulus;
  }
  return ModArithOperation(extAdd.trunc(bitWidth), type);
}

ModArithOperation
ModArithOperation::operator-(const ModArithOperation &other) const {
  assert(type == other.type);

  const APInt &a = value;
  const APInt &b = other.value;
  APInt modulus = type.getModulus().getValue();

  if (a.uge(b)) {
    return ModArithOperation(a - b, type);
  }
  unsigned bitWidth = modulus.getBitWidth();
  if (bitWidth > modulus.getActiveBits()) {
    return ModArithOperation(a + modulus - b, type);
  }

  APInt extA = a.zext(bitWidth + 1);
  APInt extB = b.zext(bitWidth + 1);
  APInt extModulus = modulus.zext(bitWidth + 1);
  APInt extSub = extA + extModulus - extB;
  return ModArithOperation(extSub.trunc(bitWidth), type);
}

ModArithOperation
ModArithOperation::operator*(const ModArithOperation &other) const {
  assert(type == other.type);

  const APInt &a = value;
  const APInt &b = other.value;
  APInt modulus = type.getModulus().getValue();

  APInt product = mulMod(a, b, modulus);
  if (type.isMontgomery()) {
    MontgomeryAttr montAttr = type.getMontgomeryAttr();
    product = mulMod(product, montAttr.getRInv().getValue(), modulus);
  }
  return ModArithOperation(product, type);
}

ModArithOperation
ModArithOperation::operator/(const ModArithOperation &other) const {
  assert(type == other.type);

  return *this * other.Inverse();
}

ModArithOperation ModArithOperation::operator-() const {
  APInt modulus = type.getModulus().getValue();

  if (value.isZero()) {
    return ModArithOperation(value, type);
  }
  return ModArithOperation(modulus - value, type);
}

ModArithOperation ModArithOperation::Double() const {
  APInt modulus = type.getModulus().getValue();

  unsigned bitWidth = modulus.getBitWidth();
  if (bitWidth > modulus.getActiveBits()) {
    APInt shl = value.shl(1);
    if (shl.uge(modulus)) {
      shl -= modulus;
    }
    return ModArithOperation(shl, type);
  }

  APInt extValue = value.zext(bitWidth + 1);
  APInt extModulus = modulus.zext(bitWidth + 1);
  APInt extAdd = extValue + extValue;
  if (extAdd.uge(extModulus)) {
    extAdd -= extModulus;
  }
  return ModArithOperation(extAdd.trunc(bitWidth), type);
}

ModArithOperation ModArithOperation::Square() const {
  APInt modulus = type.getModulus().getValue();
  APInt square = mulMod(value, value, modulus);
  if (type.isMontgomery()) {
    MontgomeryAttr montAttr = type.getMontgomeryAttr();
    square = mulMod(square, montAttr.getRInv().getValue(), modulus);
  }
  return ModArithOperation(square, type);
}

ModArithOperation ModArithOperation::Power(APInt exponent) const {
  APInt base = value;
  if (type.isMontgomery()) {
    base = FromMont().value;
  }
  APInt modulus = type.getModulus().getValue();
  APInt power = expMod(base, exponent, modulus);
  if (type.isMontgomery()) {
    MontgomeryAttr montAttr = type.getMontgomeryAttr();
    power = mulMod(power, montAttr.getR().getValue(), modulus);
  }
  return ModArithOperation(power, type);
}

ModArithOperation ModArithOperation::Inverse() const {
  APInt modulus = type.getModulus().getValue();
  auto inverse = multiplicativeInverse(value, modulus);
  if (type.isMontgomery()) {
    MontgomeryAttr montAttr = type.getMontgomeryAttr();
    inverse = mulMod(inverse, montAttr.getRSquared().getValue(), modulus);
  }
  return ModArithOperation(inverse, type);
}

ModArithOperation ModArithOperation::FromMont() const {
  APInt modulus = type.getModulus().getValue();
  MontgomeryAttr montAttr = type.getMontgomeryAttr();
  APInt product = mulMod(value, montAttr.getRInv().getValue(), modulus);
  return ModArithOperation(product, type);
}

ModArithOperation ModArithOperation::ToMont() const {
  APInt modulus = type.getModulus().getValue();
  MontgomeryAttr montAttr = type.getMontgomeryAttr();
  APInt product = mulMod(value, montAttr.getR().getValue(), modulus);
  return ModArithOperation(product, type);
}

bool ModArithOperation::isOne() const {
  if (type.isMontgomery()) {
    MontgomeryAttr montAttr = type.getMontgomeryAttr();
    return value == montAttr.getR().getValue();
  }
  return value.isOne();
}

bool ModArithOperation::isZero() const { return value.isZero(); }

llvm::SmallString<40> ModArithOperation::toString() const {
  llvm::SmallString<40> str;
  APInt val = value;
  if (type.isMontgomery()) {
    val = FromMont().value;
  }
  val.toString(str, 10, false);
  return str;
}

raw_ostream &operator<<(raw_ostream &os, const ModArithOperation &op) {
  if (op.type.isMontgomery()) {
    op.FromMont().value.print(os, false);
  } else {
    op.value.print(os, false);
  }
  return os;
}

} // namespace mlir::zkir::mod_arith
