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

#include "mlir/Dialect/Arith/IR/Arith.h"
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

// Return a constant for v of type t, splat if t is a vector type.
inline Value getConstantOrSplat(OpBuilder *b, Location loc, Type t,
                                Attribute v) {
  if (VectorType vecType = dyn_cast<VectorType>(t)) {
    v = SplatElementsAttr::get(vecType, v);
  }
  return b->create<arith::ConstantOp>(loc, t, cast<TypedAttr>(v));
}

inline Value mapConvertOpToStdScalarOp(Location loc, ArrayRef<Type> targetTypes,
                                       ArrayRef<Type> resultTypes,
                                       ArrayRef<Type> argTypes, ValueRange args,
                                       ArrayRef<NamedAttribute> attributes,
                                       OpBuilder *b) {
  assert(targetTypes.size() == 1 && "ConvertOp should return a single result");
  assert(resultTypes.size() == 1 && "ConvertOp should return a single result");
  assert(argTypes.size() == 1 && "ConvertOp should take a single argument");
  assert(args.size() == 1 && "ConvertOp should take a single argument");

  Type sourceType = getElementTypeOrSelf(argTypes.front());
  Type targetType = getElementTypeOrSelf(targetTypes.front());

  mlir::ImplicitLocOpBuilder lb(loc, *b);
  if (isa<IntegerType>(sourceType) && isa<IntegerType>(targetType)) {
    return zkx::mlir_utils::ConvertInteger(
        lb, resultTypes, sourceType, targetType, args,
        IsSignedIntegerType{}(sourceType), attributes);
  } else if (isa<prime_ir::field::PrimeFieldType>(sourceType) ||
             isa<prime_ir::field::ExtensionFieldType>(sourceType) ||
             isa<prime_ir::field::PrimeFieldType>(targetType) ||
             isa<prime_ir::field::ExtensionFieldType>(targetType)) {
    return zkx::mlir_utils::ConvertField(lb, resultTypes, sourceType,
                                         targetType, args, attributes);
  } else if (isa<prime_ir::elliptic_curve::AffineType>(sourceType) ||
             isa<prime_ir::elliptic_curve::JacobianType>(sourceType) ||
             isa<prime_ir::elliptic_curve::XYZZType>(sourceType)) {
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
    static_assert(!std::is_same<MhloOpTy, mhlo::ConvertOp>::value);
    typename MhloOpTy::Adaptor adaptor(args, op->getAttrDictionary(),
                                       op->getPropertiesStorage(),
                                       op->getRegions());
    return mapOpOfType<MhloOpTy>(op.getLoc(), resultTypes, argTypes, adaptor,
                                 attributes, b);
  }
  // Overload for mhlo::ConvertOp.
  static Value mapOpWithArgTypes(mhlo::ConvertOp op, ArrayRef<Type> resultTypes,
                                 ArrayRef<Type> argTypes, ValueRange args,
                                 ArrayRef<NamedAttribute> attributes,
                                 OpBuilder *b) {
    return impl::mapConvertOpToStdScalarOp(
        op.getLoc(), op.getType(), resultTypes, argTypes, args, attributes, b);
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

  static Value
  mapConvertOpToStdScalarOp(Location loc, ArrayRef<Type> targetTypes,
                            ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
                            ValueRange args,
                            ArrayRef<NamedAttribute> attributes, OpBuilder *b) {
    return impl::mapConvertOpToStdScalarOp(loc, targetTypes, resultTypes,
                                           argTypes, args, attributes, b);
  }
};

} // namespace mlir::mhlo

#endif // ZKX_MLIR_HLO_MHLO_TRANSFORMS_MAP_MHLO_TO_SCALAR_OP_H_
