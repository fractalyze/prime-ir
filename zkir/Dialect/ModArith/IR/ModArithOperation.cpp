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

#include "zk_dtypes/include/bit_iterator.h"
#include "zk_dtypes/include/byinverter.h"
#include "zk_dtypes/include/field/modular_operations.h"
#include "zk_dtypes/include/field/mont_multiplication.h"
#include "zkir/Utils/APIntUtils.h"

namespace zk_dtypes {

template <>
class BitTraits<mlir::APInt> {
public:
  static size_t GetNumBits(const mlir::APInt &value) {
    return value.getBitWidth();
  }

  static bool TestBit(const mlir::APInt &value, size_t index) {
    return value[index];
  }

  static void SetBit(mlir::APInt &value, size_t index, bool bitValue) {
    value.setBitVal(index, bitValue);
  }
};

} // namespace zk_dtypes

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

ModArithOperation ModArithOperation::getOne() const {
  APInt one;
  if (type.isMontgomery()) {
    MontgomeryAttr montAttr = type.getMontgomeryAttr();
    one = montAttr.getR().getValue();
  } else {
    one = APInt(value.getBitWidth(), 1);
  }
  return ModArithOperation::fromUnchecked(one, type);
}

ModArithOperation
ModArithOperation::operator+(const ModArithOperation &other) const {
  assert(type == other.type);
  return ModArithOperation::fromUnchecked(
      executeBinaryModOp<AddOp>(value, other.value, type), type);
}

ModArithOperation
ModArithOperation::operator-(const ModArithOperation &other) const {
  assert(type == other.type);
  return ModArithOperation::fromUnchecked(
      executeBinaryModOp<SubOp>(value, other.value, type), type);
}

namespace {

struct MontMulOp {
  template <typename T>
  static void apply(const T &a, const T &b, T &c, const T &mod, uint64_t nPrime,
                    bool hasModulusSpareBit, unsigned bitWidth) {
    if constexpr (std::is_integral_v<T>) {
      if (bitWidth > 32) {
        zk_dtypes::MontMul(a, b, c, mod, nPrime);
      } else if (bitWidth > 16) {
        uint32_t result;
        zk_dtypes::MontMul<uint32_t>(a, b, result, mod, nPrime);
        c = result;
      } else if (bitWidth > 8) {
        uint16_t result;
        zk_dtypes::MontMul<uint16_t>(a, b, result, mod, nPrime);
        c = result;
      } else {
        uint8_t result;
        zk_dtypes::MontMul<uint8_t>(a, b, result, mod, nPrime);
        c = result;
      }
    } else {
      zk_dtypes::MontMul(a, b, c, mod, nPrime, hasModulusSpareBit,
                         zk_dtypes::CanUseNoCarryMulOptimization(mod));
    }
  }
};

template <size_t CurrentN, size_t MaxN>
struct MontMulDispatcher {
  static void dispatch(size_t targetN, const APInt &a, const APInt &b, APInt &c,
                       const APInt &mod, uint64_t nPrime,
                       bool hasModulusSpareBit, unsigned bitWidth) {
    if (targetN == CurrentN) {
      using T = zk_dtypes::BigInt<CurrentN>;
      MontMulOp::apply<T>(
          *reinterpret_cast<const T *>(a.getRawData()),
          *reinterpret_cast<const T *>(b.getRawData()),
          *const_cast<T *>(reinterpret_cast<const T *>(c.getRawData())),
          *reinterpret_cast<const T *>(mod.getRawData()), nPrime,
          hasModulusSpareBit, bitWidth);
    } else if constexpr (CurrentN < MaxN) {
      MontMulDispatcher<CurrentN + 1, MaxN>::dispatch(
          targetN, a, b, c, mod, nPrime, hasModulusSpareBit, bitWidth);
    }
  }
};

APInt executeMontMulOp(const APInt &a, const APInt &b,
                       const ModArithType &type) {
  APInt modulus = type.getModulus().getValue();
  unsigned bitWidth = modulus.getBitWidth();
  bool hasModulusSpareBit = bitWidth > modulus.getActiveBits();
  MontgomeryAttr montAttr = type.getMontgomeryAttr();

  APInt c(bitWidth, 0);
  size_t neededLimbs = (bitWidth + 63) / 64;
  uint64_t nPrime;
  if (bitWidth > 64) {
    nPrime = montAttr.getNPrime().getValue().getZExtValue();
  } else {
    nPrime = montAttr.getNInv().getValue().getZExtValue();
  }

  if (neededLimbs > 1 && neededLimbs <= kMaximumLimbNum) {
    MontMulDispatcher<1, kMaximumLimbNum>::dispatch(
        neededLimbs, a, b, c, modulus, nPrime, hasModulusSpareBit, bitWidth);
  } else if (neededLimbs == 1) {
    MontMulOp::apply<uint64_t>(a.getZExtValue(), b.getZExtValue(),
                               *const_cast<uint64_t *>(c.getRawData()),
                               modulus.getZExtValue(), nPrime,
                               hasModulusSpareBit, bitWidth);
  } else {
    llvm_unreachable("Unsupported limb size");
  }
  return c;
}

} // namespace

ModArithOperation
ModArithOperation::operator*(const ModArithOperation &other) const {
  assert(type == other.type);

  if (type.isMontgomery()) {
    return ModArithOperation::fromUnchecked(
        executeMontMulOp(value, other.value, type), type);
  }

  const APInt &a = value;
  const APInt &b = other.value;
  APInt modulus = type.getModulus().getValue();
  return ModArithOperation::fromUnchecked(mulMod(a, b, modulus), type);
}

ModArithOperation
ModArithOperation::operator/(const ModArithOperation &other) const {
  assert(type == other.type);

  return *this * other.inverse();
}

ModArithOperation ModArithOperation::operator-() const {
  APInt modulus = type.getModulus().getValue();

  if (value.isZero()) {
    return ModArithOperation::fromUnchecked(value, type);
  }
  return ModArithOperation::fromUnchecked(modulus - value, type);
}

ModArithOperation ModArithOperation::dbl() const {
  return ModArithOperation::fromUnchecked(
      executeUnaryModOp<DoubleOp>(value, type), type);
}

namespace {

struct MontSquareOp {
  template <typename T>
  static void apply(const T &a, T &b, const T &mod, uint64_t nPrime,
                    bool hasModulusSpareBit, unsigned bitWidth) {
    if constexpr (std::is_integral_v<T>) {
      MontMulOp::apply<T>(a, a, b, mod, nPrime, hasModulusSpareBit, bitWidth);
    } else {
      zk_dtypes::MontSquare(a, b, mod, nPrime, hasModulusSpareBit);
    }
  }
};

template <size_t CurrentN, size_t MaxN>
struct MontSquareDispatcher {
  static void dispatch(size_t targetN, const APInt &a, APInt &b,
                       const APInt &mod, uint64_t nPrime,
                       bool hasModulusSpareBit, unsigned bitWidth) {
    if (targetN == CurrentN) {
      using T = zk_dtypes::BigInt<CurrentN>;
      MontSquareOp::apply<T>(
          *reinterpret_cast<const T *>(a.getRawData()),
          *const_cast<T *>(reinterpret_cast<const T *>(b.getRawData())),
          *reinterpret_cast<const T *>(mod.getRawData()), nPrime,
          hasModulusSpareBit, bitWidth);
    } else if constexpr (CurrentN < MaxN) {
      MontSquareDispatcher<CurrentN + 1, MaxN>::dispatch(
          targetN, a, b, mod, nPrime, hasModulusSpareBit, bitWidth);
    }
  }
};

APInt executeMontSquareOp(const APInt &a, const ModArithType &type) {
  APInt modulus = type.getModulus().getValue();
  unsigned bitWidth = modulus.getBitWidth();
  bool hasModulusSpareBit = bitWidth > modulus.getActiveBits();
  MontgomeryAttr montAttr = type.getMontgomeryAttr();

  APInt b(bitWidth, 0);
  size_t neededLimbs = (bitWidth + 63) / 64;
  uint64_t nPrime;
  if (bitWidth > 64) {
    nPrime = montAttr.getNPrime().getValue().getZExtValue();
  } else {
    nPrime = montAttr.getNInv().getValue().getZExtValue();
  }

  if (neededLimbs > 1 && neededLimbs <= kMaximumLimbNum) {
    MontSquareDispatcher<1, kMaximumLimbNum>::dispatch(
        neededLimbs, a, b, modulus, nPrime, hasModulusSpareBit, bitWidth);
  } else if (neededLimbs == 1) {
    MontSquareOp::apply<uint64_t>(
        a.getZExtValue(), *const_cast<uint64_t *>(b.getRawData()),
        modulus.getZExtValue(), nPrime, hasModulusSpareBit, bitWidth);
  } else {
    llvm_unreachable("Unsupported limb size");
  }
  return b;
}

} // namespace

ModArithOperation ModArithOperation::square() const {
  if (type.isMontgomery()) {
    return ModArithOperation::fromUnchecked(executeMontSquareOp(value, type),
                                            type);
  }
  APInt modulus = type.getModulus().getValue();
  return ModArithOperation::fromUnchecked(mulMod(value, value, modulus), type);
}

namespace {

ModArithOperation power(const ModArithOperation &value, const APInt &exponent) {
  auto ret = value.getOne();
  auto it = zk_dtypes::BitIteratorBE<APInt>::begin(&exponent, true);
  auto end = zk_dtypes::BitIteratorBE<APInt>::end(&exponent);
  while (it != end) {
    ret = ret.square();
    if (*it) {
      ret *= value;
    }
    ++it;
  }
  return ret;
}

} // namespace

ModArithOperation ModArithOperation::power(APInt exponent) const {
  return mod_arith::power(*this, exponent);
}

namespace {

struct InverseOp {
  template <typename T>
  static void apply(const T &a, T &b, const T &mod, const T &adjuster) {
    if constexpr (std::is_integral_v<T>) {
      using BigInt = zk_dtypes::BigInt<1>;
      auto inverter = zk_dtypes::BYInverter<1>(BigInt(mod), BigInt(adjuster));
      BigInt result;
      [[maybe_unused]] bool ok = inverter.Invert(BigInt(a), result);
      assert(ok);
      b = static_cast<uint64_t>(result);
    } else {
      auto inverter = zk_dtypes::BYInverter<T::kLimbNums>(mod, adjuster);
      [[maybe_unused]] bool ok = inverter.Invert(a, b);
      assert(ok);
    }
  }
};

template <size_t CurrentN, size_t MaxN>
struct InverseDispatcher {
  static void dispatch(size_t targetN, const APInt &a, APInt &b,
                       const APInt &mod, const APInt &adjuster) {
    if (targetN == CurrentN) {
      using T = zk_dtypes::BigInt<CurrentN>;
      InverseOp::apply<T>(
          *reinterpret_cast<const T *>(a.getRawData()),
          *const_cast<T *>(reinterpret_cast<const T *>(b.getRawData())),
          *reinterpret_cast<const T *>(mod.getRawData()),
          *reinterpret_cast<const T *>(adjuster.getRawData()));
    } else if constexpr (CurrentN < MaxN) {
      InverseDispatcher<CurrentN + 1, MaxN>::dispatch(targetN, a, b, mod,
                                                      adjuster);
    }
  }
};

APInt executeInverseOp(const APInt &a, const ModArithType &type) {
  APInt modulus = type.getModulus().getValue();
  unsigned bitWidth = modulus.getBitWidth();

  APInt b(bitWidth, 0);
  size_t neededLimbs = (bitWidth + 63) / 64;
  APInt adjuster;
  if (type.isMontgomery()) {
    MontgomeryAttr montAttr = type.getMontgomeryAttr();
    adjuster = montAttr.getRSquared().getValue();
  } else {
    adjuster = APInt(bitWidth, 1);
  }

  if (neededLimbs > 1 && neededLimbs <= kMaximumLimbNum) {
    InverseDispatcher<1, kMaximumLimbNum>::dispatch(neededLimbs, a, b, modulus,
                                                    adjuster);
  } else if (neededLimbs == 1) {
    InverseOp::apply<uint64_t>(a.getZExtValue(),
                               *const_cast<uint64_t *>(b.getRawData()),
                               modulus.getZExtValue(), adjuster.getZExtValue());
  } else {
    llvm_unreachable("Unsupported limb size");
  }
  return b;
}

} // namespace

ModArithOperation ModArithOperation::inverse() const {
  return ModArithOperation::fromUnchecked(executeInverseOp(value, type), type);
}

namespace {

struct FromMontOp {
  template <typename T>
  static void apply(const T &a, T &b, const T &mod, uint64_t nPrime,
                    unsigned bitWidth) {
    if constexpr (std::is_integral_v<T>) {
      if (bitWidth > 32) {
        zk_dtypes::MontReduce(a, b, mod, nPrime);
      } else if (bitWidth > 16) {
        uint32_t result;
        zk_dtypes::MontReduce<uint32_t>(a, result, mod, nPrime);
        b = result;
      } else if (bitWidth > 8) {
        uint16_t result;
        zk_dtypes::MontReduce<uint16_t>(a, result, mod, nPrime);
        b = result;
      } else {
        uint8_t result;
        zk_dtypes::MontReduce<uint8_t>(a, result, mod, nPrime);
        b = result;
      }
    } else {
      zk_dtypes::MontReduce(a, b, mod, nPrime);
    }
  }
};

template <size_t CurrentN, size_t MaxN>
struct FromMontDispatcher {
  static void dispatch(size_t targetN, const APInt &a, APInt &b,
                       const APInt &mod, uint64_t nPrime, unsigned bitWidth) {
    if (targetN == CurrentN) {
      using T = zk_dtypes::BigInt<CurrentN>;
      FromMontOp::apply<T>(
          *reinterpret_cast<const T *>(a.getRawData()),
          *const_cast<T *>(reinterpret_cast<const T *>(b.getRawData())),
          *reinterpret_cast<const T *>(mod.getRawData()), nPrime, bitWidth);
    } else if constexpr (CurrentN < MaxN) {
      FromMontDispatcher<CurrentN + 1, MaxN>::dispatch(targetN, a, b, mod,
                                                       nPrime, bitWidth);
    }
  }
};

APInt executeFromMontOp(const APInt &a, const ModArithType &type) {
  APInt modulus = type.getModulus().getValue();
  unsigned bitWidth = modulus.getBitWidth();
  MontgomeryAttr montAttr = type.getMontgomeryAttr();

  APInt b(bitWidth, 0);
  size_t neededLimbs = (bitWidth + 63) / 64;
  uint64_t nPrime;
  if (bitWidth > 64) {
    nPrime = montAttr.getNPrime().getValue().getZExtValue();
  } else {
    nPrime = montAttr.getNInv().getValue().getZExtValue();
  }

  if (neededLimbs > 1 && neededLimbs <= kMaximumLimbNum) {
    FromMontDispatcher<1, kMaximumLimbNum>::dispatch(neededLimbs, a, b, modulus,
                                                     nPrime, bitWidth);
  } else if (neededLimbs == 1) {
    FromMontOp::apply<uint64_t>(a.getZExtValue(),
                                *const_cast<uint64_t *>(b.getRawData()),
                                modulus.getZExtValue(), nPrime, bitWidth);
  } else {
    llvm_unreachable("Unsupported limb size");
  }
  return b;
}

} // namespace

ModArithOperation ModArithOperation::fromMont() const {
  return ModArithOperation::fromUnchecked(executeFromMontOp(value, type), type);
}

ModArithOperation ModArithOperation::toMont() const {
  MontgomeryAttr montAttr = type.getMontgomeryAttr();
  return ModArithOperation::fromUnchecked(
      executeMontMulOp(value, montAttr.getRSquared().getValue(), type), type);
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
    val = fromMont().value;
  }
  val.toString(str, 10, false);
  return str;
}

raw_ostream &operator<<(raw_ostream &os, const ModArithOperation &op) {
  if (op.type.isMontgomery()) {
    op.fromMont().value.print(os, false);
  } else {
    op.value.print(os, false);
  }
  return os;
}

} // namespace mlir::zkir::mod_arith
