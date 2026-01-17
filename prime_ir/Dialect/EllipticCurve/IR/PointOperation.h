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

#ifndef PRIME_IR_DIALECT_ELLIPTICCURVE_IR_POINTOPERATION_H_
#define PRIME_IR_DIALECT_ELLIPTICCURVE_IR_POINTOPERATION_H_

#include <array>
#include <cassert>

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Dialect/EllipticCurve/IR/PointKind.h"
#include "prime_ir/Dialect/EllipticCurve/IR/PointOperationBaseForward.h"
#include "prime_ir/Dialect/EllipticCurve/IR/PointOperationSelector.h"
#include "prime_ir/Dialect/Field/IR/FieldOperation.h"
#include "zk_dtypes/include/control_flow_operation.h"

namespace mlir::prime_ir::elliptic_curve {

template <PointKind Kind>
class PointOperationBase;

} // namespace mlir::prime_ir::elliptic_curve

namespace zk_dtypes {

template <mlir::prime_ir::elliptic_curve::PointKind Kind>
class PointTraits<mlir::prime_ir::elliptic_curve::PointOperationBase<Kind>> {
public:
  constexpr static CurveType kType = CurveType::kShortWeierstrass;

  using AffinePoint = mlir::prime_ir::elliptic_curve::PointOperationBase<
      mlir::prime_ir::elliptic_curve::PointKind::kAffine>;
  using JacobianPoint = mlir::prime_ir::elliptic_curve::PointOperationBase<
      mlir::prime_ir::elliptic_curve::PointKind::kJacobian>;
  using PointXyzz = mlir::prime_ir::elliptic_curve::PointOperationBase<
      mlir::prime_ir::elliptic_curve::PointKind::kXYZZ>;
  using BaseField = mlir::prime_ir::field::FieldOperation;
};

} // namespace zk_dtypes

namespace mlir::prime_ir::elliptic_curve {

template <PointKind Kind>
class PointOperationBase : public PointOperationSelector<Kind>::template Type<
                               PointOperationBase<Kind>> {
public:
  static constexpr size_t kNumCoords = static_cast<size_t>(Kind) + 2;

  PointOperationBase() = default;

  PointOperationBase(
      const std::array<field::FieldOperation, kNumCoords> &coords,
      PointTypeInterface pointType)
      : coords(coords), pointType(pointType) {
    assert(coords.size() == kNumCoords);
  }

  PointOperationBase(const SmallVector<APInt> &coords,
                     PointTypeInterface pointType)
      : pointType(pointType) {
    assert(coords.size() == kNumCoords);
    for (size_t i = 0; i < kNumCoords; ++i) {
      this->coords[i] =
          field::FieldOperation(coords[i], pointType.getBaseFieldType());
    }
  }

  static PointOperationBase fromUnchecked(const SmallVector<APInt> &coords,
                                          PointTypeInterface pointType) {
    std::array<field::FieldOperation, kNumCoords> newCoeffs;
    for (size_t i = 0; i < kNumCoords; ++i) {
      newCoeffs[i] = field::FieldOperation::fromUnchecked(
          coords[i], pointType.getBaseFieldType());
    }
    return fromUnchecked(newCoeffs, pointType);
  }

  static PointOperationBase
  fromUnchecked(const std::array<field::FieldOperation, kNumCoords> &coords,
                PointTypeInterface pointType) {
    PointOperationBase ret;
    ret.coords = coords;
    ret.pointType = pointType;
    return ret;
  }

  template <typename T>
  static Type getPointType(MLIRContext *context) {
    if constexpr (zk_dtypes::IsAffinePoint<T>) {
      return createSpecificPointType<AffineType, T>(context);
    } else if constexpr (zk_dtypes::IsJacobianPoint<T>) {
      return createSpecificPointType<JacobianType, T>(context);
    } else {
      return createSpecificPointType<XYZZType, T>(context);
    }
  }

  template <typename Point>
  static PointOperationBase fromZkDtype(MLIRContext *context,
                                        const Point &point) {
    using BaseField = typename Point::BaseField;
    constexpr size_t Degree = BaseField::ExtensionDegree();

    Type pointType = getPointType<Point>(context);
    if constexpr (Degree == 1) {
      return fromUnchecked(convertToAPInts(point.ToCoords()),
                           cast<PointTypeInterface>(pointType));
    } else {
      std::array<field::FieldOperation, kNumCoords> coords;
      for (size_t i = 0; i < kNumCoords; ++i) {
        coords[i] = field::FieldOperation::fromZkDtype(context, point[i]);
      }
      return fromUnchecked(coords, cast<PointTypeInterface>(pointType));
    }
  }

  PointKind getKind() const { return Kind; }

  template <PointKind Kind2 = Kind,
            std::enable_if_t<Kind2 != PointKind::kAffine> * = nullptr>
  PointOperationBase &operator+=(const PointOperationBase &other) {
    return *this = *this + other;
  }
  template <PointKind Kind2 = Kind,
            std::enable_if_t<Kind2 != PointKind::kAffine> * = nullptr>
  PointOperationBase &operator-=(const PointOperationBase &other) {
    return *this = *this - other;
  }

  auto dbl() const { return this->Double(); }

  bool operator==(const PointOperationBase &other) const {
    assert(pointType == other.pointType);
    return coords == other.coords;
  }
  bool operator!=(const PointOperationBase &other) const {
    assert(pointType == other.pointType);
    return coords != other.coords;
  }

  template <PointKind Kind2>
  PointOperationBase<Kind2> convert() const {
    if constexpr (Kind == Kind2) {
      return *this;
    } else if constexpr (Kind2 == PointKind::kAffine) {
      return this->ToAffine();
    } else if constexpr (Kind2 == PointKind::kJacobian) {
      return this->ToJacobian();
    } else {
      static_assert(Kind2 == PointKind::kXYZZ, "Unsupported point kind");
      return this->ToXyzz();
    }
  }

protected:
  // PascalCase methods (zk_dtypes compatible)
  template <typename, typename>
  friend class zk_dtypes::AffinePointOperation;
  template <typename, typename>
  friend class zk_dtypes::JacobianPointOperation;
  template <typename, typename>
  friend class zk_dtypes::PointXyzzOperation;

  template <PointKind Kind2>
  friend raw_ostream &operator<<(raw_ostream &os,
                                 const PointOperationBase<Kind2> &op);

  const std::array<field::FieldOperation, kNumCoords> &ToCoords() const {
    return coords;
  }

  PointOperationBase
  FromCoords(const std::array<field::FieldOperation, kNumCoords> &c) const {
    return PointOperationBase(c, pointType);
  }

  PointOperationBase<PointKind::kAffine>
  CreateAffinePoint(const std::array<field::FieldOperation, 2> &coords) const {
    auto attr = cast<ShortWeierstrassAttr>(pointType.getCurveAttr());
    return PointOperationBase<PointKind::kAffine>(
        coords, AffineType::get(attr.getContext(), attr));
  }

  PointOperationBase<PointKind::kJacobian> CreateJacobianPoint(
      const std::array<field::FieldOperation, 3> &coords) const {
    auto attr = cast<ShortWeierstrassAttr>(pointType.getCurveAttr());
    return PointOperationBase<PointKind::kJacobian>(
        coords, JacobianType::get(attr.getContext(), attr));
  }

  PointOperationBase<PointKind::kXYZZ>
  CreatePointXyzz(const std::array<field::FieldOperation, 4> &coords) const {
    auto attr = cast<ShortWeierstrassAttr>(pointType.getCurveAttr());
    return PointOperationBase<PointKind::kXYZZ>(
        coords, XYZZType::get(attr.getContext(), attr));
  }

  PointOperationBase<PointKind::kAffine> &&
  MaybeConvertToAffine(PointOperationBase<PointKind::kAffine> &&point) const {
    return std::move(point);
  }

  PointOperationBase<PointKind::kJacobian> &&MaybeConvertToJacobian(
      PointOperationBase<PointKind::kJacobian> &&point) const {
    return std::move(point);
  }

  PointOperationBase<PointKind::kXYZZ> &&
  MaybeConvertToXyzz(PointOperationBase<PointKind::kXYZZ> &&point) const {
    return std::move(point);
  }

  bool IsAZero() const { return GetA().isZero(); }

  field::FieldOperation GetA() const {
    return field::FieldOperation::fromUnchecked(
        cast<ShortWeierstrassAttr>(pointType.getCurveAttr()).getA(),
        pointType.getBaseFieldType());
  }

  zk_dtypes::ControlFlowOperation<bool> GetCFOperation() const { return {}; }

  template <typename T>
  static ShortWeierstrassAttr getShortWeierstrassAttr(MLIRContext *context) {
    using BaseField = typename T::BaseField;
    constexpr size_t Degree = BaseField::ExtensionDegree();

    if constexpr (Degree == 1) {
      using UnderlyingType = typename BaseField::UnderlyingType;

      auto getParam = [&](auto config_val) {
        return convertToIntegerAttr(context,
                                    static_cast<UnderlyingType>(config_val));
      };

      return ShortWeierstrassAttr::get(
          context,
          field::PrimeFieldOperation::getPrimeFieldType<BaseField>(context),
          getParam(T::Curve::Config::kA.value()),
          getParam(T::Curve::Config::kB.value()),
          getParam(T::Curve::Config::kX.value()),
          getParam(T::Curve::Config::kY.value()));
    } else {
      using BasePrimeField = typename BaseField::Config::BasePrimeField;
      using UnderlyingType = typename BasePrimeField::UnderlyingType;
      constexpr size_t N = BaseField::Config::kDegreeOverBaseField;

      auto getParam = [&](const auto &config_values) {
        std::array<UnderlyingType, N> raw_values;
        for (size_t i = 0; i < N; ++i) {
          raw_values[i] = config_values[i].value();
        }
        return convertToDenseIntElementsAttr(
            context,
            ArrayRef<UnderlyingType>(raw_values.data(), raw_values.size()));
      };

      return ShortWeierstrassAttr::get(
          context,
          field::ExtensionFieldOperation<N>::template getExtensionFieldType<
              BaseField>(context),
          getParam(T::Curve::Config::kA.values()),
          getParam(T::Curve::Config::kB.values()),
          getParam(T::Curve::Config::kX.values()),
          getParam(T::Curve::Config::kY.values()));
    }
  }

  template <typename MLIRPointType, typename T>
  static MLIRPointType createSpecificPointType(MLIRContext *context) {
    return MLIRPointType::get(context, getShortWeierstrassAttr<T>(context));
  }

  std::array<field::FieldOperation, kNumCoords> coords;
  PointTypeInterface pointType;
};

template <PointKind Kind>
raw_ostream &operator<<(raw_ostream &os, const PointOperationBase<Kind> &op) {
  llvm::interleaveComma(op.coords, os,
                        [&](const field::FieldOperation &c) { os << c; });
  return os;
}

extern template class PointOperationBase<PointKind::kAffine>;
extern template class PointOperationBase<PointKind::kJacobian>;
extern template class PointOperationBase<PointKind::kXYZZ>;

extern template raw_ostream &
operator<<(raw_ostream &os, const PointOperationBase<PointKind::kAffine> &op);
extern template raw_ostream &
operator<<(raw_ostream &os, const PointOperationBase<PointKind::kJacobian> &op);
extern template raw_ostream &
operator<<(raw_ostream &os, const PointOperationBase<PointKind::kXYZZ> &op);

using AffinePointOperation = PointOperationBase<PointKind::kAffine>;
using JacobianPointOperation = PointOperationBase<PointKind::kJacobian>;
using XYZZPointOperation = PointOperationBase<PointKind::kXYZZ>;

class PointOperation {
public:
  using OperationType =
      std::variant<AffinePointOperation, JacobianPointOperation,
                   XYZZPointOperation>;

  template <typename T, typename = std::enable_if_t<
                            !std::is_same_v<std::decay_t<T>, PointOperation> &&
                            std::is_constructible_v<OperationType, T>>>
  PointOperation(T &&cg) // NOLINT(runtime/explicit)
      : operation(std::forward<T>(cg)) {}
  ~PointOperation() = default;

  template <typename Point>
  static PointOperation fromZkDtype(MLIRContext *context, const Point &point) {
    constexpr PointKind Kind = getPointKind<Point>();
    return PointOperationBase<Kind>::fromZkDtype(context, point);
  }

  PointKind getKind() const;

  PointOperation add(const PointOperation &other, PointKind outputKind) const;
  PointOperation dbl(PointKind outputKind) const;
  PointOperation convert(PointKind outputKind) const;

private:
  OperationType operation;
};

} // namespace mlir::prime_ir::elliptic_curve

#endif // PRIME_IR_DIALECT_ELLIPTICCURVE_IR_POINTOPERATION_H_
