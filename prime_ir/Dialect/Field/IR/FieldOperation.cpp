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

#include "prime_ir/Dialect/Field/IR/FieldOperation.h"

namespace mlir::prime_ir::field {

FieldOperation FieldOperation::getZero() const {
  return std::visit(
      [](const auto &op) -> FieldOperation { return op.getZero(); }, operation);
}

FieldOperation FieldOperation::getOne() const {
  return std::visit(
      [](const auto &op) -> FieldOperation { return op.getOne(); }, operation);
}

FieldOperation::operator APInt() const {
  if (auto pfOperation = std::get_if<PrimeFieldOperation>(&operation)) {
    return static_cast<APInt>(*pfOperation);
  }
  if (auto bfOperation = std::get_if<BinaryFieldOperation>(&operation)) {
    return static_cast<APInt>(*bfOperation);
  }
  llvm_unreachable("Cannot convert ExtensionFieldOperation to APInt");
}

FieldOperation::operator SmallVector<APInt>() const {
  return std::visit(
      [](const auto &op) -> SmallVector<APInt> {
        using T = std::decay_t<decltype(op)>;
        if constexpr (std::is_same_v<T, PrimeFieldOperation> ||
                      std::is_same_v<T, BinaryFieldOperation>) {
          llvm_unreachable("Cannot convert scalar field to SmallVector<APInt>");
        } else {
          // Both non-tower and tower extension fields can be flattened
          // to SmallVector<APInt> using flattenToPrimeCoeffs()
          return op.flattenToPrimeCoeffs();
        }
      },
      operation);
}

Type FieldOperation::getType() const {
  return std::visit([](const auto &op) -> Type { return op.getType(); },
                    operation);
}

namespace {

template <typename F>
FieldOperation applyUnaryOp(const FieldOperation::OperationType &operation,
                            F op) {
  return std::visit([&](const auto &v) -> FieldOperation { return op(v); },
                    operation);
}

template <typename R, typename F>
R applyBinaryOp(const FieldOperation::OperationType &a,
                const FieldOperation::OperationType &b, F op) {
  return std::visit(
      [&](const auto &lhs, const auto &rhs) -> R {
        if constexpr (std::is_same_v<std::decay_t<decltype(lhs)>,
                                     std::decay_t<decltype(rhs)>>) {
          return op(lhs, rhs);
        }
        llvm_unreachable("Unsupported field type in binary operator");
      },
      a, b);
}

// Type trait to detect ExtensionFieldOperation and extract its BaseFieldT.
template <typename T>
struct IsExtensionFieldOp : std::false_type {};

template <size_t N, typename B>
struct IsExtensionFieldOp<ExtensionFieldOperation<N, B>> : std::true_type {
  using BaseFieldT = B;
};

// Mixed-type dispatch: one operand is an extension field, the other is its
// base field. ExtBaseOp handles (ext, base); BaseExtOp handles (base, ext)
// with arguments reordered as (ext, base) for convenience.
template <typename ExtBaseOp, typename BaseExtOp>
FieldOperation applyMixedBinaryOp(const FieldOperation::OperationType &a,
                                  const FieldOperation::OperationType &b,
                                  ExtBaseOp extBaseOp, BaseExtOp baseExtOp) {
  return std::visit(
      [&](const auto &lhs, const auto &rhs) -> FieldOperation {
        using LHS = std::decay_t<decltype(lhs)>;
        using RHS = std::decay_t<decltype(rhs)>;
        if constexpr (IsExtensionFieldOp<LHS>::value) {
          if constexpr (std::is_same_v<
                            typename IsExtensionFieldOp<LHS>::BaseFieldT,
                            RHS>) {
            return extBaseOp(lhs, rhs);
          }
        }
        if constexpr (IsExtensionFieldOp<RHS>::value) {
          if constexpr (std::is_same_v<
                            typename IsExtensionFieldOp<RHS>::BaseFieldT,
                            LHS>) {
            return baseExtOp(rhs, lhs);
          }
        }
        llvm_unreachable("Unsupported mixed-type field operation");
      },
      a, b);
}

} // namespace

FieldOperation FieldOperation::operator+(const FieldOperation &other) const {
  if (operation.index() == other.operation.index()) {
    return applyBinaryOp<FieldOperation>(
        operation, other.operation,
        [](const auto &a, const auto &b) { return a + b; });
  }
  return applyMixedBinaryOp(
      operation, other.operation,
      [](const auto &ext, const auto &base) { return ext + base; },
      [](const auto &ext, const auto &base) { return ext + base; });
}

FieldOperation FieldOperation::operator-(const FieldOperation &other) const {
  if (operation.index() == other.operation.index()) {
    return applyBinaryOp<FieldOperation>(
        operation, other.operation,
        [](const auto &a, const auto &b) { return a - b; });
  }
  return applyMixedBinaryOp(
      operation, other.operation,
      [](const auto &ext, const auto &base) { return ext - base; },
      [](const auto &ext, const auto &base) { return -(ext - base); });
}

FieldOperation FieldOperation::operator*(const FieldOperation &other) const {
  if (operation.index() == other.operation.index()) {
    return applyBinaryOp<FieldOperation>(
        operation, other.operation,
        [](const auto &a, const auto &b) { return a * b; });
  }
  return applyMixedBinaryOp(
      operation, other.operation,
      [](const auto &ext, const auto &base) { return ext * base; },
      [](const auto &ext, const auto &base) { return ext * base; });
}

FieldOperation FieldOperation::operator/(const FieldOperation &other) const {
  if (operation.index() == other.operation.index()) {
    return applyBinaryOp<FieldOperation>(
        operation, other.operation,
        [](const auto &a, const auto &b) { return a / b; });
  }
  return applyMixedBinaryOp(
      operation, other.operation,
      [](const auto &ext, const auto &base) { return ext / base; },
      [](const auto &ext, const auto &base) { return ext.inverse() * base; });
}

FieldOperation FieldOperation::operator-() const {
  return applyUnaryOp(operation, [](const auto &v) { return -v; });
}

FieldOperation FieldOperation::dbl() const {
  return applyUnaryOp(operation, [](const auto &v) { return v.dbl(); });
}

FieldOperation FieldOperation::square() const {
  return applyUnaryOp(operation, [](const auto &v) { return v.square(); });
}

FieldOperation FieldOperation::power(const APInt &exponent) const {
  return applyUnaryOp(operation,
                      [&](const auto &v) { return v.power(exponent); });
}

FieldOperation FieldOperation::inverse() const {
  return applyUnaryOp(operation, [](const auto &v) { return v.inverse(); });
}

FieldOperation FieldOperation::fromMont() const {
  return applyUnaryOp(operation, [](const auto &v) { return v.fromMont(); });
}

FieldOperation FieldOperation::toMont() const {
  return applyUnaryOp(operation, [](const auto &v) { return v.toMont(); });
}

bool FieldOperation::isZero() const {
  return std::visit([](const auto &op) -> bool { return op.isZero(); },
                    operation);
}

bool FieldOperation::isOne() const {
  return std::visit([](const auto &op) -> bool { return op.isOne(); },
                    operation);
}

bool FieldOperation::operator==(const FieldOperation &other) const {
  return applyBinaryOp<bool>(
      operation, other.operation,
      [](const auto &a, const auto &b) { return a == b; });
}

bool FieldOperation::operator!=(const FieldOperation &other) const {
  return applyBinaryOp<bool>(
      operation, other.operation,
      [](const auto &a, const auto &b) { return a != b; });
}

raw_ostream &operator<<(raw_ostream &os, const FieldOperation &op) {
  return std::visit([&](const auto &v) -> raw_ostream & { return os << v; },
                    op.operation);
}

} // namespace mlir::prime_ir::field
