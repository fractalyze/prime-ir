/* Copyright 2026 The PrimeIR Authors.

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

#ifndef PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_POINTCODEGEN_H_
#define PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_POINTCODEGEN_H_

#include <array>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/ConversionUtils.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/FieldCodeGen.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/PointCodeGenBaseForward.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Dialect/EllipticCurve/IR/PointOperationSelector.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Utils/BuilderContext.h"
#include "prime_ir/Utils/ControlFlowOperation.h"
#include "zk_dtypes/include/geometry/curve_type.h"
#include "zk_dtypes/include/geometry/point_traits.h"

namespace zk_dtypes {

template <mlir::prime_ir::elliptic_curve::PointKind Kind>
class PointTraits<mlir::prime_ir::elliptic_curve::PointCodeGenBase<Kind>> {
public:
  constexpr static CurveType kType = CurveType::kShortWeierstrass;

  using AffinePoint = mlir::prime_ir::elliptic_curve::PointCodeGenBase<
      mlir::prime_ir::elliptic_curve::PointKind::kAffine>;
  using JacobianPoint = mlir::prime_ir::elliptic_curve::PointCodeGenBase<
      mlir::prime_ir::elliptic_curve::PointKind::kJacobian>;
  using PointXyzz = mlir::prime_ir::elliptic_curve::PointCodeGenBase<
      mlir::prime_ir::elliptic_curve::PointKind::kXYZZ>;
  using BaseField = mlir::prime_ir::elliptic_curve::FieldCodeGen;
};

} // namespace zk_dtypes

namespace mlir::prime_ir::elliptic_curve {

template <PointKind Kind>
class PointCodeGenBase : public PointOperationSelector<Kind>::template Type<
                             PointCodeGenBase<Kind>> {
public:
  static constexpr size_t kNumCoords = static_cast<size_t>(Kind) + 2;

  PointCodeGenBase() = default;
  explicit PointCodeGenBase(Value value) : value(value) {}

  PointKind getKind() const { return Kind; }
  operator Value() const { return value; }

protected:
  template <typename, typename>
  friend class zk_dtypes::AffinePointOperation;
  template <typename, typename>
  friend class zk_dtypes::JacobianPointOperation;
  template <typename, typename>
  friend class zk_dtypes::PointXyzzOperation;

  std::array<FieldCodeGen, kNumCoords> ToCoords() const {
    ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
    Operation::result_range coords = toCoords(*b, value);
    std::array<FieldCodeGen, kNumCoords> ret;
    for (size_t i = 0; i < kNumCoords; ++i) {
      ret[i] = FieldCodeGen(coords[i]);
    }
    return ret;
  }

  PointCodeGenBase
  FromCoords(const std::array<FieldCodeGen, kNumCoords> &coords_array) const {
    SmallVector<Value, kNumCoords> coords(coords_array.begin(),
                                          coords_array.end());
    ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
    return PointCodeGenBase(fromCoords(*b, value.getType(), coords));
  }

  PointCodeGenBase<PointKind::kAffine>
  CreateAffinePoint(const std::array<FieldCodeGen, 2> &coords) const {
    return createPoint<PointKind::kAffine, 2>(coords);
  }

  PointCodeGenBase<PointKind::kJacobian>
  CreateJacobianPoint(const std::array<FieldCodeGen, 3> &coords) const {
    return createPoint<PointKind::kJacobian, 3>(coords);
  }

  PointCodeGenBase<PointKind::kXYZZ>
  CreatePointXyzz(const std::array<FieldCodeGen, 4> &coords) const {
    return createPoint<PointKind::kXYZZ, 4>(coords);
  }

  PointCodeGenBase<PointKind::kAffine>
  MaybeConvertToAffine(mlir::ValueRange values) const {
    return PointCodeGenBase<PointKind::kAffine>(values[0]);
  }

  PointCodeGenBase<PointKind::kJacobian>
  MaybeConvertToJacobian(mlir::ValueRange values) const {
    return PointCodeGenBase<PointKind::kJacobian>(values[0]);
  }

  PointCodeGenBase<PointKind::kXYZZ>
  MaybeConvertToXyzz(mlir::ValueRange values) const {
    return PointCodeGenBase<PointKind::kXYZZ>(values[0]);
  }

  Value IsAZero() const {
    Type baseFieldType = getCurveFromPointLike(value.getType()).getBaseField();
    ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
    Value zero =
        cast<field::FieldTypeInterface>(baseFieldType).createZeroConstant(*b);
    return b->create<field::CmpOp>(arith::CmpIPredicate::eq, GetA(), zero);
  }

  FieldCodeGen GetA() const {
    ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
    ShortWeierstrassAttr swAttr = getCurveFromPointLike(value.getType());
    return FieldCodeGen(
        b->create<field::ConstantOp>(swAttr.getBaseField(), swAttr.getA()));
  }

  zk_dtypes::ControlFlowOperation<Value> GetCFOperation() const { return {}; }

  template <PointKind Kind2, size_t N>
  PointCodeGenBase<Kind2>
  createPoint(const std::array<FieldCodeGen, N> &coords_array) const {
    SmallVector<Value, N> coords(coords_array.begin(), coords_array.end());
    return createPoint<Kind2>(coords);
  }

  template <PointKind Kind2>
  PointCodeGenBase<Kind2> createPoint(mlir::ValueRange coords) const {
    Type type;
    auto curve = getCurveFromPointLike(value.getType());
    if constexpr (Kind2 == PointKind::kAffine) {
      type = AffineType::get(value.getContext(), curve);
    } else if constexpr (Kind2 == PointKind::kJacobian) {
      type = JacobianType::get(value.getContext(), curve);
    } else if constexpr (Kind2 == PointKind::kXYZZ) {
      type = XYZZType::get(value.getContext(), curve);
    }
    ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
    return PointCodeGenBase<Kind2>(fromCoords(*b, type, coords));
  }

  Value value;
};

extern template class PointCodeGenBase<PointKind::kAffine>;
extern template class PointCodeGenBase<PointKind::kJacobian>;
extern template class PointCodeGenBase<PointKind::kXYZZ>;

using AffinePointCodeGen = PointCodeGenBase<PointKind::kAffine>;
using JacobianPointCodeGen = PointCodeGenBase<PointKind::kJacobian>;
using XYZZPointCodeGen = PointCodeGenBase<PointKind::kXYZZ>;

class PointCodeGen {
public:
  using CodeGenType =
      std::variant<AffinePointCodeGen, JacobianPointCodeGen, XYZZPointCodeGen>;

  template <typename T, typename = std::enable_if_t<
                            !std::is_same_v<std::decay_t<T>, FieldCodeGen> &&
                            std::is_constructible_v<CodeGenType, T>>>
  PointCodeGen(T &&cg) // NOLINT(runtime/explicit)
      : codeGen(std::forward<T>(cg)) {}
  PointCodeGen(Type type, Value value);
  ~PointCodeGen() = default;

  operator Value() const;
  PointKind getKind() const;

  PointCodeGen add(const PointCodeGen &other, PointKind outputKind) const;
  PointCodeGen dbl(PointKind outputKind) const;
  PointCodeGen convert(PointKind outputKind) const;

private:
  CodeGenType codeGen;
};

} // namespace mlir::prime_ir::elliptic_curve

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_POINTCODEGEN_H_
