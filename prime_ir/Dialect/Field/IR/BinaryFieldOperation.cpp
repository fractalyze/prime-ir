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

#include "prime_ir/Dialect/Field/IR/BinaryFieldOperation.h"

#include "prime_ir/Utils/Power.h"
#include "zk_dtypes/include/field/binary_field_multiplication.h"

namespace mlir::prime_ir::field {

namespace {

// Helper to dispatch tower level operations at runtime.
// Converts APInt to native type, performs operation, converts back.

template <size_t TowerLevel>
struct TowerDispatcher {
  using T = typename zk_dtypes::TowerTraits<TowerLevel>::ValueType;

  static llvm::APInt multiply(const llvm::APInt &a, const llvm::APInt &b) {
    T aVal = static_cast<T>(a.getZExtValue());
    T bVal = static_cast<T>(b.getZExtValue());
    T result = zk_dtypes::BinaryMul<TowerLevel>(aVal, bVal);
    return llvm::APInt(1u << TowerLevel, static_cast<uint64_t>(result));
  }

  static llvm::APInt square(const llvm::APInt &a) {
    T aVal = static_cast<T>(a.getZExtValue());
    T result = zk_dtypes::BinarySquare<TowerLevel>(aVal);
    return llvm::APInt(1u << TowerLevel, static_cast<uint64_t>(result));
  }

  static llvm::APInt inverse(const llvm::APInt &a) {
    T aVal = static_cast<T>(a.getZExtValue());
    T result = zk_dtypes::BinaryInverse<TowerLevel>(aVal);
    return llvm::APInt(1u << TowerLevel, static_cast<uint64_t>(result));
  }
};

// Specialization for tower level 7 (128-bit) which uses BigInt<2>
template <>
struct TowerDispatcher<7> {
  static zk_dtypes::BigInt<2> apIntToBigInt(const llvm::APInt &a) {
    uint64_t lo = a.extractBitsAsZExtValue(64, 0);
    uint64_t hi = a.extractBitsAsZExtValue(64, 64);
    return zk_dtypes::BigInt<2>({lo, hi});
  }

  static llvm::APInt bigIntToApint(const zk_dtypes::BigInt<2> &val) {
    llvm::APInt result(128, 0);
    result.insertBits(llvm::APInt(64, val[0]), 0);
    result.insertBits(llvm::APInt(64, val[1]), 64);
    return result;
  }

  static llvm::APInt multiply(const llvm::APInt &a, const llvm::APInt &b) {
    auto aVal = apIntToBigInt(a);
    auto bVal = apIntToBigInt(b);
    auto result = zk_dtypes::BinaryMul<7>(aVal, bVal);
    return bigIntToApint(result);
  }

  static llvm::APInt square(const llvm::APInt &a) {
    auto aVal = apIntToBigInt(a);
    auto result = zk_dtypes::BinarySquare<7>(aVal);
    return bigIntToApint(result);
  }

  static llvm::APInt inverse(const llvm::APInt &a) {
    auto aVal = apIntToBigInt(a);
    auto result = zk_dtypes::BinaryInverse<7>(aVal);
    return bigIntToApint(result);
  }
};

// Runtime dispatch based on tower level using template instantiation table
template <template <size_t> class Op, typename... Args>
llvm::APInt dispatchByTowerLevel(unsigned towerLevel, Args &&...args) {
  assert(towerLevel <= 7 && "Tower level must be 0-7");
  using DispatchFn = llvm::APInt (*)(Args...);
  static constexpr DispatchFn kDispatchTable[] = {
      Op<0>::call, Op<1>::call, Op<2>::call, Op<3>::call,
      Op<4>::call, Op<5>::call, Op<6>::call, Op<7>::call,
  };
  return kDispatchTable[towerLevel](std::forward<Args>(args)...);
}

// Operation wrappers for dispatch table
template <size_t TowerLevel>
struct MultiplyOp {
  static llvm::APInt call(const llvm::APInt &a, const llvm::APInt &b) {
    return TowerDispatcher<TowerLevel>::multiply(a, b);
  }
};

template <size_t TowerLevel>
struct SquareOp {
  static llvm::APInt call(const llvm::APInt &a) {
    return TowerDispatcher<TowerLevel>::square(a);
  }
};

template <size_t TowerLevel>
struct InverseOp {
  static llvm::APInt call(const llvm::APInt &a) {
    return TowerDispatcher<TowerLevel>::inverse(a);
  }
};

} // namespace

BinaryFieldOperation
BinaryFieldOperation::operator*(const BinaryFieldOperation &other) const {
  assert(type_ == other.type_);
  llvm::APInt result = dispatchByTowerLevel<MultiplyOp>(type_.getTowerLevel(),
                                                        value_, other.value_);
  return fromUnchecked(result, type_);
}

BinaryFieldOperation BinaryFieldOperation::square() const {
  llvm::APInt result =
      dispatchByTowerLevel<SquareOp>(type_.getTowerLevel(), value_);
  return fromUnchecked(result, type_);
}

BinaryFieldOperation
BinaryFieldOperation::power(const llvm::APInt &exponent) const {
  return mlir::prime_ir::power(*this, exponent);
}

BinaryFieldOperation BinaryFieldOperation::inverse() const {
  if (isZero()) {
    return getZero();
  }
  llvm::APInt result =
      dispatchByTowerLevel<InverseOp>(type_.getTowerLevel(), value_);
  return fromUnchecked(result, type_);
}

} // namespace mlir::prime_ir::field
