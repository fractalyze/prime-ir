/* Copyright 2019 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

#ifndef ZKX_MLIR_HLO_MHLO_TRANSFORMS_MAP_MHLO_TO_SCALAR_OP_H_
#define ZKX_MLIR_HLO_MHLO_TRANSFORMS_MAP_MHLO_TO_SCALAR_OP_H_

#include <cassert>
#include <optional>
#include <type_traits>

#include "llvm/ADT/APInt.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "zkx/mlir/codegen_utils.h"
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::mhlo {
namespace impl {

// A struct to map MhloBinaryOpTy type to the corresponding integer scalar
// operation types.
template <typename MhloBinaryOpTy>
struct MhloToScalarOp {
  using IOp = void;
  using UOp = void;
  using FOp = void;
  using ECOp = void;
};

template <>
struct MhloToScalarOp<mhlo::AddOp> {
  using IOp = arith::AddIOp;
  using UOp = arith::AddIOp;
  using FOp = prime_ir::field::AddOp;
  using ECOp = prime_ir::elliptic_curve::AddOp;
};
template <>
struct MhloToScalarOp<mhlo::MulOp> {
  using IOp = arith::MulIOp;
  using UOp = arith::MulIOp;
  using FOp = prime_ir::field::MulOp;
};
template <>
struct MhloToScalarOp<mhlo::SubtractOp> {
  using IOp = arith::SubIOp;
  using UOp = arith::SubIOp;
  using FOp = prime_ir::field::SubOp;
  using ECOp = prime_ir::elliptic_curve::SubOp;
};
template <>
struct MhloToScalarOp<mhlo::CompareOp> {
  using IOp = arith::CmpIOp;
  using FOp = prime_ir::field::CmpOp;
  using ECOp = prime_ir::elliptic_curve::CmpOp;
};
template <>
struct MhloToScalarOp<mhlo::AndOp> {
  using IOp = arith::AndIOp;
  using UOp = arith::AndIOp;
};
template <>
struct MhloToScalarOp<mhlo::OrOp> {
  using IOp = arith::OrIOp;
  using UOp = arith::OrIOp;
};
template <>
struct MhloToScalarOp<mhlo::XorOp> {
  using IOp = arith::XOrIOp;
  using UOp = arith::XOrIOp;
};
template <>
struct MhloToScalarOp<mhlo::ClzOp> {
  using IOp = math::CountLeadingZerosOp;
  using UOp = math::CountLeadingZerosOp;
};
template <>
struct MhloToScalarOp<mhlo::PopulationCountOp> {
  using IOp = math::CtPopOp;
  using UOp = math::CtPopOp;
};

// Alias for the map from MHLO binary op type to STD signed integer op type.
template <typename MhloOp>
using ScalarIOp = typename MhloToScalarOp<MhloOp>::IOp;
// Alias for the map from MHLO binary op type to STD unsigned integer op type.
template <typename MhloOp>
using ScalarUOp = typename MhloToScalarOp<MhloOp>::UOp;
// Alias for the map from MHLO binary op type to STD field op type.
template <typename MhloOp>
using ScalarFOp = typename MhloToScalarOp<MhloOp>::FOp;
// Alias for the map from MHLO binary op type to STD elliptic curve op type.
template <typename MhloOp>
using ScalarECOp = typename MhloToScalarOp<MhloOp>::ECOp;

template <typename... Args>
struct MapMhloOpToScalarOpImpl {
  Value operator()(Location /*loc*/, ArrayRef<Type> /*ResultTypes*/,
                   ArrayRef<Type> /*argTypes*/, ValueRange /*args*/,
                   ArrayRef<NamedAttribute> /*attributes*/, OpBuilder * /*b*/) {
    return nullptr;
  }
};

template <typename StdScalarOp>
struct MapMhloOpToScalarOpImpl<StdScalarOp> {
  Value operator()(Location loc, ArrayRef<Type> resultTypes,
                   ArrayRef<Type> /*argTypes*/, ValueRange args,
                   ArrayRef<NamedAttribute> attributes, OpBuilder *b) {
    return b->template create<StdScalarOp>(loc, resultTypes, args, attributes);
  }
};

template <typename SupportedType, typename StdScalarOp, typename... Args>
struct MapMhloOpToScalarOpImpl<SupportedType, StdScalarOp, Args...> {
  Value operator()(Location loc, ArrayRef<Type> resultTypes,
                   ArrayRef<Type> argTypes, ValueRange args,
                   ArrayRef<NamedAttribute> attributes, OpBuilder *b) {
    Type elementType = getElementTypeOrSelf(argTypes.front());
    if (SupportedType{}(elementType)) {
      return b->template create<StdScalarOp>(loc, resultTypes, args,
                                             attributes);
    }
    return MapMhloOpToScalarOpImpl<Args...>{}(loc, resultTypes, argTypes, args,
                                              attributes, b);
  }
};

template <typename SupportedType, typename... Args>
struct MapMhloOpToScalarOpImpl<SupportedType, void, Args...> {
  Value operator()(Location loc, ArrayRef<Type> resultTypes,
                   ArrayRef<Type> argTypes, ValueRange args,
                   ArrayRef<NamedAttribute> attributes, OpBuilder *b) {
    return MapMhloOpToScalarOpImpl<Args...>{}(loc, resultTypes, argTypes, args,
                                              attributes, b);
  }
};

struct IsAnyIntegerType {
  bool operator()(Type t) { return isa<IntegerType>(t); }
};

struct IsSignedIntegerType {
  bool operator()(Type t) {
    // Pretend that signless is signed. This will change eventually.
    return isa<IntegerType>(t) && !t.isUnsignedInteger() &&
           !t.isSignlessInteger(1);
  }
};

struct IsUnsignedIntegerType {
  bool operator()(Type t) {
    return t.isUnsignedInteger() || t.isSignlessInteger(1);
  }
};

struct IsFieldType {
  bool operator()(Type t) {
    return isa<prime_ir::field::PrimeFieldType>(t) ||
           isa<prime_ir::field::ExtensionFieldType>(t);
  }
};

struct IsEllipticCurveType {
  bool operator()(Type t) {
    return isa<prime_ir::elliptic_curve::AffineType>(t) ||
           isa<prime_ir::elliptic_curve::JacobianType>(t) ||
           isa<prime_ir::elliptic_curve::XYZZType>(t);
  }
};

template <template <typename T> class MapTy, typename OpTy,
          typename PredTy = llvm::is_detected<MapTy, OpTy>>
struct MapableIf {
  using type = void;
};
template <template <typename T> class MapTy, typename OpTy>
struct MapableIf<MapTy, OpTy, std::true_type> {
  using type = MapTy<OpTy>;
};

// Inserts the computation that corresponds to the body of the loop for lowered
// MHLO unary/binary op. Returns the value for the result.
template <typename MhloOpTy>
inline Value mapMhloOpToStdScalarOp(Location loc, ArrayRef<Type> resultTypes,
                                    ArrayRef<Type> argTypes,
                                    typename MhloOpTy::Adaptor adaptor,
                                    ArrayRef<NamedAttribute> attributes,
                                    OpBuilder *b) {
  using ScalarIOpOrVoid = typename MapableIf<ScalarIOp, MhloOpTy>::type;
  using ScalarUOpOrVoid = typename MapableIf<ScalarUOp, MhloOpTy>::type;
  using ScalarFOpOrVoid = typename MapableIf<ScalarFOp, MhloOpTy>::type;
  using ScalarECOpOrVoid = typename MapableIf<ScalarECOp, MhloOpTy>::type;
  return MapMhloOpToScalarOpImpl<IsSignedIntegerType, ScalarIOpOrVoid,
                                 IsUnsignedIntegerType, ScalarUOpOrVoid,
                                 IsFieldType, ScalarFOpOrVoid,
                                 IsEllipticCurveType, ScalarECOpOrVoid>{}(
      loc, resultTypes, argTypes, adaptor.getOperands(), attributes, b);
}

// Return a constant for v of type t, splat if t is a shaped type.
inline Value getConstantOrSplat(OpBuilder *b, Location loc, Type t,
                                Attribute v) {
  if (auto shapedType = dyn_cast<ShapedType>(t)) {
    v = SplatElementsAttr::get(shapedType, v);
  }
  return b->create<arith::ConstantOp>(loc, t, cast<TypedAttr>(v));
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::ConvertOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::ConvertOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder *b) {
  assert(resultTypes.size() == 1 && "ConvertOp should return a single result");
  assert(argTypes.size() == 1 && "ConvertOp should take a single argument");

  Type sourceType = getElementTypeOrSelf(argTypes.front());
  Type targetType = getElementTypeOrSelf(resultTypes.front());
  ValueRange args = adaptor.getOperands();

  mlir::ImplicitLocOpBuilder lb(loc, *b);
  if (isa<IntegerType>(sourceType) && isa<IntegerType>(targetType)) {
    return zkx::mlir_utils::ConvertInteger(
        lb, resultTypes, sourceType, targetType, args,
        IsSignedIntegerType{}(sourceType), attributes);
  } else if (isa<prime_ir::field::FieldTypeInterface>(sourceType) ||
             isa<prime_ir::field::FieldTypeInterface>(targetType)) {
    return zkx::mlir_utils::ConvertField(lb, resultTypes, sourceType,
                                         targetType, args, attributes);
  } else if (isa<prime_ir::elliptic_curve::PointTypeInterface>(sourceType) ||
             isa<prime_ir::elliptic_curve::PointTypeInterface>(targetType)) {
    return zkx::mlir_utils::ConvertEcPoint(lb, resultTypes, sourceType,
                                           targetType, args, attributes);
  }
  return nullptr;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::MulOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::MulOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder *b) {
  Type leftType = adaptor.getLhs().getType();
  Type leftElementType = getElementTypeOrSelf(leftType);
  if (IsAnyIntegerType{}(leftElementType)) {
    using ScalarIOpOrVoid = typename MapableIf<ScalarIOp, mhlo::MulOp>::type;
    using ScalarUOpOrVoid = typename MapableIf<ScalarUOp, mhlo::MulOp>::type;
    return MapMhloOpToScalarOpImpl<IsSignedIntegerType, ScalarIOpOrVoid,
                                   IsUnsignedIntegerType, ScalarUOpOrVoid>{}(
        loc, resultTypes, argTypes, adaptor.getOperands(), attributes, b);
  } else if (IsFieldType{}(leftElementType)) {
    Type rightType = adaptor.getRhs().getType();
    Type rightElementType = getElementTypeOrSelf(rightType);
    if (IsEllipticCurveType{}(rightElementType)) {
      return b->create<prime_ir::elliptic_curve::ScalarMulOp>(
          loc, resultTypes, adaptor.getOperands(), attributes);
    } else {
      return b->create<prime_ir::field::MulOp>(
          loc, resultTypes, adaptor.getOperands(), attributes);
    }
  }
  return nullptr;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::DivOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::DivOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder *b) {
  Type type = adaptor.getLhs().getType();
  Type elementType = getElementTypeOrSelf(type);
  if (IsAnyIntegerType{}(elementType)) {
    mlir::ImplicitLocOpBuilder lb(loc, *b);
    Type originalType = getElementTypeOrSelf(argTypes.front());
    bool isSigned = !originalType.isUnsignedInteger();
    return zkx::mlir_utils::DivideInteger(lb, adaptor.getLhs(),
                                          adaptor.getRhs(), isSigned);
  } else if (IsFieldType{}(elementType)) {
    auto inv = b->create<prime_ir::field::InverseOp>(loc, adaptor.getRhs());
    return b->create<prime_ir::field::MulOp>(loc, adaptor.getLhs(), inv);
  }
  return nullptr;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::NegOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::NegOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder *b) {
  Type elementType = getElementTypeOrSelf(adaptor.getOperand().getType());
  if (IsAnyIntegerType{}(elementType)) {
    // lmhlo.neg(x, result) -> result = sub(0, x)
    Value lhs = adaptor.getOperand();
    Value zeroIntval =
        b->create<arith::ConstantOp>(loc, b->getZeroAttr(lhs.getType()));
    return b->create<ScalarIOp<mhlo::SubtractOp>>(loc, zeroIntval, lhs);
  } else if (IsFieldType{}(elementType)) {
    return b->create<prime_ir::field::NegateOp>(loc, adaptor.getOperand());
  } else if (IsEllipticCurveType{}(elementType)) {
    return b->create<prime_ir::elliptic_curve::NegateOp>(loc,
                                                         adaptor.getOperand());
  }
  return nullptr;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::InverseOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::InverseOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder *b) {
  Type elementType = getElementTypeOrSelf(adaptor.getOperand().getType());
  if (IsFieldType{}(elementType)) {
    return b->create<prime_ir::field::InverseOp>(loc, adaptor.getOperand());
  }
  return nullptr;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::PowOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::PowOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder *b) {
  Type type = adaptor.getLhs().getType();
  Type elementType = getElementTypeOrSelf(type);
  if (IsAnyIntegerType{}(elementType)) {
    auto lb = ImplicitLocOpBuilder(loc, *b);
    return zkx::mlir_utils::PowerInteger(lb, adaptor.getLhs(), adaptor.getRhs(),
                                         elementType.isSignedInteger());
  } else if (IsFieldType{}(elementType)) {
    return b->create<prime_ir::field::PowUIOp>(loc, adaptor.getLhs(),
                                               adaptor.getRhs());
  }
  return nullptr;
}

struct IsPrimeFieldType {
  bool operator()(Type t) { return isa<prime_ir::field::PrimeFieldType>(t); }
};

struct IsExtFieldType {
  bool operator()(Type t) {
    return isa<prime_ir::field::ExtensionFieldType>(t);
  }
};

// Converts mhlo::ComparisonDirection to comparison predicate.
template <typename PredicateType>
inline std::optional<PredicateType> getCmpPredicate(mhlo::ComparisonDirection,
                                                    bool) {
  return std::nullopt;
}

template <>
inline std::optional<arith::CmpIPredicate>
getCmpPredicate<arith::CmpIPredicate>(
    mhlo::ComparisonDirection comparisonDirection, bool isSigned) {
  return llvm::StringSwitch<std::optional<arith::CmpIPredicate>>(
             stringifyComparisonDirection(comparisonDirection))
      .Case("EQ", arith::CmpIPredicate::eq)
      .Case("NE", arith::CmpIPredicate::ne)
      .Case("GE",
            isSigned ? arith::CmpIPredicate::sge : arith::CmpIPredicate::uge)
      .Case("GT",
            isSigned ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::ugt)
      .Case("LE",
            isSigned ? arith::CmpIPredicate::sle : arith::CmpIPredicate::ule)
      .Case("LT",
            isSigned ? arith::CmpIPredicate::slt : arith::CmpIPredicate::ult)
      .Default(std::nullopt);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::BitcastConvertOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> /*argTypes*/,
    mhlo::BitcastConvertOp::Adaptor adaptor,
    ArrayRef<NamedAttribute> /*attributes*/, OpBuilder *b) {
  Value operand = adaptor.getOperand();
  Type resultType = resultTypes.front();
  if (operand.getType() == resultType) {
    return operand;
  }
  return b->create<arith::BitcastOp>(loc, resultType, operand);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::CompareOp>(
    Location loc, ArrayRef<Type> /*resultTypes*/, ArrayRef<Type> argTypes,
    mhlo::CompareOp::Adaptor adaptor, ArrayRef<NamedAttribute> /*attributes*/,
    OpBuilder *b) {
  ComparisonDirection comparisonDirection = adaptor.getComparisonDirection();
  const auto &lhs = adaptor.getLhs();
  const auto &rhs = adaptor.getRhs();
  Type elementType = getElementTypeOrSelf(argTypes.front());

  if (mlir::isa<IntegerType>(elementType)) {
    bool isUnsigned = IsUnsignedIntegerType{}(elementType);
    auto predicate =
        getCmpPredicate<arith::CmpIPredicate>(comparisonDirection, !isUnsigned);
    assert(predicate.has_value() && "expected valid comparison direction");
    return b->create<ScalarIOp<mhlo::CompareOp>>(loc, predicate.value(), lhs,
                                                 rhs);
  } else if (IsPrimeFieldType{}(elementType)) {
    auto predicate =
        getCmpPredicate<arith::CmpIPredicate>(comparisonDirection, false);
    assert(predicate.has_value() && "expected valid comparison direction");
    return b->create<ScalarFOp<mhlo::CompareOp>>(loc, predicate.value(), lhs,
                                                 rhs);
  } else if (IsExtFieldType{}(elementType)) {
    // Extension fields only support EQ/NE comparisons.
    if (comparisonDirection != ComparisonDirection::EQ &&
        comparisonDirection != ComparisonDirection::NE) {
      return nullptr;
    }
    auto predicate =
        getCmpPredicate<arith::CmpIPredicate>(comparisonDirection, false);
    assert(predicate.has_value() && "expected valid comparison direction");
    return b->create<ScalarFOp<mhlo::CompareOp>>(loc, predicate.value(), lhs,
                                                 rhs);
  } else if (IsEllipticCurveType{}(elementType)) {
    // Elliptic curve points only support EQ/NE comparisons.
    if (comparisonDirection != ComparisonDirection::EQ &&
        comparisonDirection != ComparisonDirection::NE) {
      return nullptr;
    }
    auto predicate =
        getCmpPredicate<arith::CmpIPredicate>(comparisonDirection, false);
    assert(predicate.has_value() && "expected valid comparison direction");
    return b->create<ScalarECOp<mhlo::CompareOp>>(loc, predicate.value(), lhs,
                                                  rhs);
  }
  return nullptr;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::SelectOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::SelectOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder *b) {
  return MapMhloOpToScalarOpImpl<arith::SelectOp>{}(
      loc, resultTypes, argTypes, adaptor.getOperands(), attributes, b);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::AbsOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::AbsOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder *b) {
  Type elementType = getElementTypeOrSelf(argTypes.front());
  if (elementType.isSignlessInteger() || elementType.isSignedInteger()) {
    // abs(x) = select((x >= 0), x, sub(0, x))
    Value lhs = adaptor.getOperand();
    Value zeroIntval =
        b->create<arith::ConstantOp>(loc, b->getZeroAttr(lhs.getType()));
    auto lhsGeZero = b->create<ScalarIOp<CompareOp>>(
        loc, arith::CmpIPredicate::sge, lhs, zeroIntval);
    auto negVal = b->create<ScalarIOp<mhlo::SubtractOp>>(loc, zeroIntval, lhs);
    return b->create<arith::SelectOp>(loc, lhsGeZero, lhs, negVal);
  }
  return nullptr;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::MaxOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::MaxOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder *b) {
  Type elementType = getElementTypeOrSelf(argTypes.front());
  if (IsPrimeFieldType{}(elementType)) {
    // max(a, b) = select(a >= b, a, b)
    ValueRange operands = adaptor.getOperands();
    auto cmp = b->create<ScalarFOp<mhlo::CompareOp>>(
        loc, arith::CmpIPredicate::uge, operands[0], operands[1]);
    return b->create<arith::SelectOp>(loc, cmp, operands[0], operands[1]);
  }
  return MapMhloOpToScalarOpImpl<IsSignedIntegerType, arith::MaxSIOp,
                                 IsUnsignedIntegerType, arith::MaxUIOp>{}(
      loc, resultTypes, argTypes, adaptor.getOperands(), attributes, b);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::MinOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::MinOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder *b) {
  Type elementType = getElementTypeOrSelf(argTypes.front());
  if (IsPrimeFieldType{}(elementType)) {
    // min(a, b) = select(a <= b, a, b)
    ValueRange operands = adaptor.getOperands();
    auto cmp = b->create<ScalarFOp<mhlo::CompareOp>>(
        loc, arith::CmpIPredicate::ule, operands[0], operands[1]);
    return b->create<arith::SelectOp>(loc, cmp, operands[0], operands[1]);
  }
  return MapMhloOpToScalarOpImpl<IsSignedIntegerType, arith::MinSIOp,
                                 IsUnsignedIntegerType, arith::MinUIOp>{}(
      loc, resultTypes, argTypes, adaptor.getOperands(), attributes, b);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::ClampOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::ClampOp::Adaptor op, ArrayRef<NamedAttribute> attributes,
    OpBuilder *b) {
  // clamp(lb, x, ub) = min(max(lb, x), ub)
  Value maxLbX = mapMhloOpToStdScalarOp<mhlo::MaxOp>(
      loc, resultTypes, argTypes, ValueRange{op.getMin(), op.getOperand()},
      attributes, b);
  return mapMhloOpToStdScalarOp<mhlo::MinOp>(loc, resultTypes, argTypes,
                                             ValueRange{maxLbX, op.getMax()},
                                             attributes, b);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::RemOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::RemOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder *b) {
  // Integer remainder overflow behavior:
  //   X % 0 == X
  //   INT_SMIN %s -1 = 0
  Type originalType = getElementTypeOrSelf(argTypes.front());
  ImplicitLocOpBuilder lb(loc, *b);
  Value lhs = adaptor.getLhs();
  Value rhs = adaptor.getRhs();
  Type type = lhs.getType();
  auto elementType = cast<IntegerType>(getElementTypeOrSelf(type));
  Value zero = lb.create<arith::ConstantOp>(lb.getZeroAttr(type));
  Value one = getConstantOrSplat(
      &lb, lb.getLoc(), type,
      lb.getIntegerAttr(elementType, llvm::APInt(elementType.getWidth(), 1)));
  Value rhsIsZero =
      lb.create<arith::CmpIOp>(arith::CmpIPredicate::eq, rhs, zero);

  if (originalType.isUnsignedInteger()) {
    Value safeRhs = lb.create<arith::SelectOp>(rhsIsZero, one, rhs);
    Value safeRem = lb.create<arith::RemUIOp>(lhs, safeRhs);
    return lb.create<arith::SelectOp>(rhsIsZero, lhs, safeRem);
  }

  // For signed: also check for INT_MIN % -1.
  Value smin = getConstantOrSplat(
      &lb, lb.getLoc(), type,
      lb.getIntegerAttr(
          elementType, llvm::APInt::getSignedMinValue(elementType.getWidth())));
  Value lhsIsSmin =
      lb.create<arith::CmpIOp>(arith::CmpIPredicate::eq, lhs, smin);
  Value minusOne = getConstantOrSplat(
      &lb, lb.getLoc(), type,
      lb.getIntegerAttr(elementType,
                        llvm::APInt::getAllOnes(elementType.getWidth())));
  Value rhsIsMinusOne =
      lb.create<arith::CmpIOp>(arith::CmpIPredicate::eq, rhs, minusOne);
  Value hasIntMinOverflow = lb.create<arith::AndIOp>(lhsIsSmin, rhsIsMinusOne);
  Value rhsIsUnsafe = lb.create<arith::OrIOp>(rhsIsZero, hasIntMinOverflow);
  Value safeRhs = lb.create<arith::SelectOp>(rhsIsUnsafe, one, rhs);
  Value safeRem = lb.create<arith::RemSIOp>(lhs, safeRhs);
  Value safeSmin = lb.create<arith::SelectOp>(hasIntMinOverflow, zero, safeRem);
  return lb.create<arith::SelectOp>(rhsIsZero, lhs, safeSmin);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::NotOp>(
    Location loc, ArrayRef<Type> /*resultTypes*/, ArrayRef<Type> /*argTypes*/,
    mhlo::NotOp::Adaptor adaptor, ArrayRef<NamedAttribute> /*attributes*/,
    OpBuilder *b) {
  Type elementType = getElementTypeOrSelf(adaptor.getOperand().getType());
  if (auto integerType = dyn_cast<IntegerType>(elementType)) {
    // not(x) -> x ^ -1
    Value allOnes = getConstantOrSplat(
        b, loc, adaptor.getOperand().getType(),
        b->getIntegerAttr(integerType,
                          llvm::APInt::getAllOnes(integerType.getWidth())));
    return b->create<arith::XOrIOp>(loc, allOnes, adaptor.getOperand());
  }
  return nullptr;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::SignOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> /*argTypes*/,
    mhlo::SignOp::Adaptor adaptor, ArrayRef<NamedAttribute> /*attributes*/,
    OpBuilder *b) {
  Value operand = adaptor.getOperand();
  Type elementType = getElementTypeOrSelf(operand.getType());
  if (auto integerType = dyn_cast<IntegerType>(elementType)) {
    // sign(x) = x == 0 ? 0 : ((x s>> 31) | 1)
    Value zero =
        b->create<arith::ConstantOp>(loc, b->getZeroAttr(operand.getType()));
    Value bitwidthMinusOne = getConstantOrSplat(
        b, loc, operand.getType(),
        b->getIntegerAttr(integerType, integerType.getWidth() - 1));
    Value one = getConstantOrSplat(b, loc, operand.getType(),
                                   b->getIntegerAttr(integerType, 1));
    Value cmp =
        b->create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, operand, zero);
    Value ashr = b->create<arith::ShRSIOp>(loc, operand, bitwidthMinusOne);
    Value orOp = b->create<arith::OrIOp>(loc, ashr, one);
    return b->create<arith::SelectOp>(loc, cmp, zero, orOp);
  }
  return nullptr;
}

/// Construct operations to select the saturated value if the shift amount is
/// greater than the bitwidth of the type.
inline Value selectShiftedOrSaturated(ImplicitLocOpBuilder &lb, Value rhs,
                                      Value shifted, Value saturated,
                                      Type type) {
  Type etype =
      isa<ShapedType>(type) ? cast<ShapedType>(type).getElementType() : type;
  auto bitWidthInt = etype.getIntOrFloatBitWidth();
  Value bitWidth = getConstantOrSplat(&lb, lb.getLoc(), type,
                                      lb.getIntegerAttr(etype, bitWidthInt));
  Value cmp =
      lb.create<arith::CmpIOp>(arith::CmpIPredicate::ugt, bitWidth, rhs);
  return lb.create<arith::SelectOp>(cmp, shifted, saturated);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::ShiftLeftOp>(
    Location loc, ArrayRef<Type> /*resultTypes*/, ArrayRef<Type> /*argTypes*/,
    mhlo::ShiftLeftOp::Adaptor adaptor, ArrayRef<NamedAttribute> /*attributes*/,
    OpBuilder *b) {
  ImplicitLocOpBuilder lb(loc, *b);
  Value lhs = adaptor.getLhs();
  Value rhs = adaptor.getRhs();
  Type type = lhs.getType();

  // "Saturate" if the shift is greater than the bitwidth of the type
  Value zero = lb.create<arith::ConstantOp>(lb.getZeroAttr(type));
  Value shifted = lb.create<arith::ShLIOp>(lhs, rhs);

  return selectShiftedOrSaturated(lb, rhs, shifted, zero, type);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::ShiftRightLogicalOp>(
    Location loc, ArrayRef<Type> /*resultTypes*/, ArrayRef<Type> /*argTypes*/,
    mhlo::ShiftRightLogicalOp::Adaptor adaptor,
    ArrayRef<NamedAttribute> /*attributes*/, OpBuilder *b) {
  ImplicitLocOpBuilder lb(loc, *b);
  Value lhs = adaptor.getLhs();
  Value rhs = adaptor.getRhs();
  Type type = lhs.getType();

  // "Saturate" if the shift is greater than the bitwidth of the type
  Value zero = lb.create<arith::ConstantOp>(b->getZeroAttr(type));
  Value shifted = lb.create<arith::ShRUIOp>(lhs, rhs);

  return selectShiftedOrSaturated(lb, rhs, shifted, zero, type);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::ShiftRightArithmeticOp>(
    Location loc, ArrayRef<Type> /*resultTypes*/, ArrayRef<Type> /*argTypes*/,
    mhlo::ShiftRightArithmeticOp::Adaptor adaptor,
    ArrayRef<NamedAttribute> /*attributes*/, OpBuilder *b) {
  ImplicitLocOpBuilder lb(loc, *b);
  Value lhs = adaptor.getLhs();
  Value rhs = adaptor.getRhs();
  Type type = lhs.getType();
  Type etype =
      isa<ShapedType>(type) ? cast<ShapedType>(type).getElementType() : type;
  auto bitWidthInt = etype.getIntOrFloatBitWidth();

  // "Saturate" if the shift is greater than the bitwidth of the type
  Value maxShift = getConstantOrSplat(
      b, loc, type, lb.getIntegerAttr(etype, bitWidthInt - 1));
  Value saturatedShifted = lb.create<arith::ShRSIOp>(lhs, maxShift);
  Value shifted = lb.create<arith::ShRSIOp>(lhs, rhs);

  return selectShiftedOrSaturated(lb, rhs, shifted, saturatedShifted, type);
}

} // namespace impl

struct MhloOpToStdScalarOp {
  // Converts mhlo 'op' to linalg and arith ops.
  template <typename MhloOpTy>
  static Value mapOp(MhloOpTy op, ArrayRef<Type> resultTypes, ValueRange args,
                     ArrayRef<NamedAttribute> attributes, OpBuilder *b) {
    auto argTypes = llvm::to_vector(op->getOperandTypes());
    return mapOpWithArgTypes(op, resultTypes, argTypes, args, attributes, b);
  }

  // Converts mhlo 'op' to linalg and arith ops. The types of 'args' may already
  // be converted, 'argTypes' are their original types.
  template <typename MhloOpTy>
  static Value mapOpWithArgTypes(MhloOpTy op, ArrayRef<Type> resultTypes,
                                 ArrayRef<Type> argTypes, ValueRange args,
                                 ArrayRef<NamedAttribute> attributes,
                                 OpBuilder *b) {
    typename MhloOpTy::Adaptor adaptor(args, op->getAttrDictionary(),
                                       op->getPropertiesStorage(),
                                       op->getRegions());
    return mapOpOfType<MhloOpTy>(op.getLoc(), resultTypes, argTypes, adaptor,
                                 attributes, b);
  }

  // Converts mhlo 'op' to linalg and arith ops.
  template <typename MhloOpTy>
  static Value mapOpOfType(Location loc, ArrayRef<Type> resultTypes,
                           ArrayRef<Type> argTypes,
                           typename MhloOpTy::Adaptor adaptor,
                           ArrayRef<NamedAttribute> attributes, OpBuilder *b) {
    return impl::mapMhloOpToStdScalarOp<MhloOpTy>(loc, resultTypes, argTypes,
                                                  adaptor, attributes, b);
  }
};

} // namespace mlir::mhlo

#endif // ZKX_MLIR_HLO_MHLO_TRANSFORMS_MAP_MHLO_TO_SCALAR_OP_H_
