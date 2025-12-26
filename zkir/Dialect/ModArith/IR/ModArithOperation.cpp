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

#include "zk_dtypes/include/field/modular_operations.h"
#include "zkir/Utils/APIntUtils.h"

namespace mlir::zkir::mod_arith {
namespace {

constexpr size_t kMaximumLimbNum = 9;

struct AddOp {
  template <typename T>
  static void apply(const T &a, const T &b, T &c, const T &mod,
                    bool hasModulusSpareBit) {
    zk_dtypes::ModAdd<T>(a, b, c, mod, hasModulusSpareBit);
  }
};

struct SubOp {
  template <typename T>
  static void apply(const T &a, const T &b, T &c, const T &mod,
                    bool hasModulusSpareBit) {
    zk_dtypes::ModSub<T>(a, b, c, mod, hasModulusSpareBit);
  }
};

struct DoubleOp {
  template <typename T>
  static void apply(const T &a, T &b, const T &mod, bool hasModulusSpareBit) {
    zk_dtypes::ModDouble<T>(a, b, mod, hasModulusSpareBit);
  }
};

template <typename Op, size_t CurrentN, size_t MaxN>
struct UnaryModDispatcher {
  static void dispatch(size_t targetN, const APInt &a, APInt &b,
                       const APInt &mod, bool hasModulusSpareBit) {
    if (targetN == CurrentN) {
      using T = zk_dtypes::BigInt<CurrentN>;
      Op::template apply<T>(
          *reinterpret_cast<const T *>(a.getRawData()),
          *const_cast<T *>(reinterpret_cast<const T *>(b.getRawData())),
          *reinterpret_cast<const T *>(mod.getRawData()), hasModulusSpareBit);
    } else if constexpr (CurrentN < MaxN) {
      UnaryModDispatcher<Op, CurrentN + 1, MaxN>::dispatch(targetN, a, b, mod,
                                                           hasModulusSpareBit);
    }
  }
};

template <typename Op>
APInt executeUnaryModOp(const APInt &a, const ModArithType &type) {
  APInt modulus = type.getModulus().getValue();
  unsigned bitWidth = modulus.getBitWidth();
  bool hasModulusSpareBit = bitWidth > modulus.getActiveBits();

  APInt b(bitWidth, 0);
  size_t neededLimbs = (bitWidth + 63) / 64;

  if (neededLimbs > 1 && neededLimbs <= kMaximumLimbNum) {
    UnaryModDispatcher<Op, 1, kMaximumLimbNum>::dispatch(
        neededLimbs, a, b, modulus, hasModulusSpareBit);
  } else if (neededLimbs == 1) {
    Op::template apply<uint64_t>(a.getZExtValue(),
                                 *const_cast<uint64_t *>(b.getRawData()),
                                 modulus.getZExtValue(), hasModulusSpareBit);
  } else {
    llvm_unreachable("Unsupported limb size");
  }
  return b;
}

template <typename Op, size_t CurrentN, size_t MaxN>
struct BinaryModDispatcher {
  static void dispatch(size_t targetN, const APInt &a, const APInt &b, APInt &c,
                       const APInt &mod, bool hasModulusSpareBit) {
    if (targetN == CurrentN) {
      using T = zk_dtypes::BigInt<CurrentN>;
      Op::template apply<T>(
          *reinterpret_cast<const T *>(a.getRawData()),
          *reinterpret_cast<const T *>(b.getRawData()),
          *const_cast<T *>(reinterpret_cast<const T *>(c.getRawData())),
          *reinterpret_cast<const T *>(mod.getRawData()), hasModulusSpareBit);
    } else if constexpr (CurrentN < MaxN) {
      BinaryModDispatcher<Op, CurrentN + 1, MaxN>::dispatch(
          targetN, a, b, c, mod, hasModulusSpareBit);
    }
  }
};

template <typename Op>
APInt executeBinaryModOp(const APInt &a, const APInt &b,
                         const ModArithType &type) {
  APInt modulus = type.getModulus().getValue();
  unsigned bitWidth = modulus.getBitWidth();
  bool hasModulusSpareBit = bitWidth > modulus.getActiveBits();

  APInt c(bitWidth, 0);
  size_t neededLimbs = (bitWidth + 63) / 64;

  if (neededLimbs > 1 && neededLimbs <= kMaximumLimbNum) {
    BinaryModDispatcher<Op, 1, kMaximumLimbNum>::dispatch(
        neededLimbs, a, b, c, modulus, hasModulusSpareBit);
  } else if (neededLimbs == 1) {
    Op::template apply<uint64_t>(a.getZExtValue(), b.getZExtValue(),
                                 *const_cast<uint64_t *>(c.getRawData()),
                                 modulus.getZExtValue(), hasModulusSpareBit);
  } else {
    llvm_unreachable("Unsupported limb size");
  }
  return c;
}

} // namespace

ModArithOperation
ModArithOperation::operator+(const ModArithOperation &other) const {
  assert(type == other.type);
  return ModArithOperation(executeBinaryModOp<AddOp>(value, other.value, type),
                           type);
}

ModArithOperation
ModArithOperation::operator-(const ModArithOperation &other) const {
  assert(type == other.type);
  return ModArithOperation(executeBinaryModOp<SubOp>(value, other.value, type),
                           type);
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
  return ModArithOperation(executeUnaryModOp<DoubleOp>(value, type), type);
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
