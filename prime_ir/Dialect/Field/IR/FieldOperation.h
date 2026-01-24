// Copyright 2025 The PrimeIR Authors.
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

#ifndef PRIME_IR_DIALECT_FIELD_IR_FIELDOPERATION_H_
#define PRIME_IR_DIALECT_FIELD_IR_FIELDOPERATION_H_

#include <cassert>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "prime_ir/Dialect/EllipticCurve/IR/PointOperationBaseForward.h"
#include "prime_ir/Dialect/Field/IR/ExtensionFieldOperation.h"
#include "prime_ir/Dialect/Field/IR/PrimeFieldOperation.h"
#include "zk_dtypes/include/geometry/point_declarations.h"

namespace mlir::prime_ir::field {

class FieldOperation {
public:
  using OperationType =
      std::variant<PrimeFieldOperation, QuadraticExtensionFieldOperation,
                   CubicExtensionFieldOperation,
                   QuarticExtensionFieldOperation>;

  FieldOperation() = default;

  template <typename T, typename = std::enable_if_t<
                            !std::is_same_v<std::decay_t<T>, FieldOperation> &&
                            std::is_constructible_v<OperationType, T>>>
  FieldOperation(T &&operation) // NOLINT(runtime/explicit)
      : operation(std::forward<T>(operation)) {}

  template <typename T>
  FieldOperation(T &&value, Type type) {
    if (auto pfType = dyn_cast<PrimeFieldType>(type)) {
      operation = PrimeFieldOperation(std::forward<T>(value), pfType);
      return;
    }
    createExtFieldOp(std::forward<T>(value), cast<ExtensionFieldType>(type));
  }
  FieldOperation(const SmallVector<APInt> &coeffs, Type type) {
    createExtFieldOp(coeffs, cast<ExtensionFieldType>(type));
  }
  ~FieldOperation() = default;

  FieldOperation getZero() const;
  FieldOperation getOne() const;

  template <typename T>
  static FieldOperation fromUnchecked(T &&value, Type type) {
    FieldOperation ret;
    if (auto pfType = dyn_cast<PrimeFieldType>(type)) {
      ret.operation =
          PrimeFieldOperation::fromUnchecked(std::forward<T>(value), pfType);
      return ret;
    }
    ret.createRawExtFieldOp(std::forward<T>(value),
                            cast<ExtensionFieldType>(type));
    return ret;
  }
  static FieldOperation fromUnchecked(TypedAttr attr, Type type) {
    FieldOperation ret;
    if (auto pfType = dyn_cast<PrimeFieldType>(type)) {
      ret.operation =
          PrimeFieldOperation::fromUnchecked(cast<IntegerAttr>(attr), pfType);
      return ret;
    }
    ret.createRawExtFieldOp(cast<DenseIntElementsAttr>(attr),
                            cast<ExtensionFieldType>(type));
    return ret;
  }

  static FieldOperation fromUnchecked(const SmallVector<APInt> &coeffs,
                                      Type type) {
    FieldOperation ret;
    ret.createRawExtFieldOp(coeffs, cast<ExtensionFieldType>(type));
    return ret;
  }

  template <typename F>
  static FieldOperation fromZkDtype(MLIRContext *context, const F &f) {
    if constexpr (F::ExtensionDegree() == 1) {
      return PrimeFieldOperation::fromZkDtype(context, f);
    } else {
      constexpr size_t N = F::Config::kDegreeOverBaseField;
      return ExtensionFieldOperation<N>::fromZkDtype(context, f);
    }
  }

  operator APInt() const;
  operator SmallVector<APInt>() const;

  const OperationType &getOperation() const { return operation; }
  Type getType() const;

  FieldOperation operator+(const FieldOperation &other) const;
  FieldOperation &operator+=(const FieldOperation &other) {
    return *this = *this + other;
  }
  FieldOperation operator-(const FieldOperation &other) const;
  FieldOperation &operator-=(const FieldOperation &other) {
    return *this = *this - other;
  }
  FieldOperation operator*(const FieldOperation &other) const;
  FieldOperation &operator*=(const FieldOperation &other) {
    return *this = *this * other;
  }
  FieldOperation operator/(const FieldOperation &other) const;
  FieldOperation &operator/=(const FieldOperation &other) {
    return *this = *this / other;
  }
  FieldOperation operator-() const;
  FieldOperation dbl() const;
  FieldOperation square() const;
  FieldOperation power(const APInt &exponent) const;
  FieldOperation inverse() const;
  FieldOperation fromMont() const;
  FieldOperation toMont() const;

  bool isZero() const;
  bool isOne() const;
  bool operator==(const FieldOperation &other) const;
  bool operator!=(const FieldOperation &other) const;

private:
  template <elliptic_curve::PointKind Kind>
  friend class elliptic_curve::PointOperationBase;
  template <typename, typename>
  friend class zk_dtypes::AffinePointOperation;
  template <typename, typename>
  friend class zk_dtypes::JacobianPointOperation;
  template <typename, typename>
  friend class zk_dtypes::PointXyzzOperation;
  friend raw_ostream &operator<<(raw_ostream &os, const FieldOperation &op);

  FieldOperation Double() const { return dbl(); }
  FieldOperation Square() const { return square(); }
  FieldOperation Inverse() const { return inverse(); }
  bool IsZero() const { return isZero(); }
  FieldOperation CreateConst(int64_t constant) const {
    return FieldOperation(constant, getType());
  }

  template <typename T>
  void createExtFieldOp(T &&value, ExtensionFieldType efType) {
    unsigned degree = efType.getDegree();
    switch (degree) {
    case 2:
      operation = ExtensionFieldOperation<2>(std::forward<T>(value), efType);
      break;
    case 3:
      operation = ExtensionFieldOperation<3>(std::forward<T>(value), efType);
      break;
    case 4:
      operation = ExtensionFieldOperation<4>(std::forward<T>(value), efType);
      break;
    default:
      llvm_unreachable("Unsupported extension field degree");
    }
  }

  template <typename T>
  void createRawExtFieldOp(T &&value, ExtensionFieldType efType) {
    unsigned degree = efType.getDegree();
    switch (degree) {
    case 2:
      operation = ExtensionFieldOperation<2>::fromUnchecked(
          std::forward<T>(value), efType);
      break;
    case 3:
      operation = ExtensionFieldOperation<3>::fromUnchecked(
          std::forward<T>(value), efType);
      break;
    case 4:
      operation = ExtensionFieldOperation<4>::fromUnchecked(
          std::forward<T>(value), efType);
      break;
    default:
      llvm_unreachable("Unsupported extension field degree");
    }
  }

  OperationType operation;
};

raw_ostream &operator<<(raw_ostream &os, const FieldOperation &op);

} // namespace mlir::prime_ir::field

#endif // PRIME_IR_DIALECT_FIELD_IR_FIELDOPERATION_H_
