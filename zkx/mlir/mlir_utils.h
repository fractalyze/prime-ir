/* Copyright 2025 The ZKX Authors.

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

#ifndef ZKX_MLIR_MLIR_UTILS_H_
#define ZKX_MLIR_MLIR_UTILS_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithAttributes.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"
#include "zkx/comparison_util.h"
#include "zkx/service/llvm_ir/llvm_util.h"
#include "zkx/shape.h"
#include "zkx/zkx_data.pb.h"

namespace zkx::mlir_utils {

mlir::Value GetConstantOrSplat(mlir::ImplicitLocOpBuilder& b, mlir::Type t,
                               mlir::Attribute v);

// Emits a tensor-level bitcast from 'input' to 'output_type', dispatching
// to the appropriate prime_ir op based on element types:
//
// - EC src → field dst : ec::BitcastOp
// - EC src → int dst   : ec::BitcastOp to field, then field::BitcastOp
// - int/field → EC dst : field::BitcastOp to field (if needed), then
//                        ec::BitcastOp
// - field src or dst   : field::BitcastOp
// - standard           : tensor::BitcastOp
//
// The caller is responsible for any reshape needed before calling this when
// rank changes are not driven by EC coordinate expansion.
mlir::Value BuildZkTensorBitcast(mlir::ImplicitLocOpBuilder& b,
                                 mlir::Value input,
                                 mlir::RankedTensorType output_type);

// Bitcasts a scalar ZK value to an integer type.
//
// Dispatch:
// - Field type    → field::BitcastOp(int_ty, value)
// - EC point type → elliptic_curve::BitcastOp(field_ty, value),
//                   then field::BitcastOp(int_ty, field_val)
//
// The EC chain uses ec::BitcastOp to reinterpret the EC point as its base
// field type, followed by field::BitcastOp to obtain the integer.  Both
// ops accept AnyType and perform pure bit-level reinterpretation.
mlir::Value BuildBitcastToInt(mlir::ImplicitLocOpBuilder& b, mlir::Value value,
                              mlir::IntegerType int_ty);

// Bitcasts an integer scalar back to a ZK type (field or EC).
//
// Dispatch:
// - Field target  → field::BitcastOp(target_ty, value)
// - EC target     → field::BitcastOp(field_ty, value),
//                   then elliptic_curve::BitcastOp(target_ty, field_val)
mlir::Value BuildBitcastFromInt(mlir::ImplicitLocOpBuilder& b,
                                mlir::Value value, mlir::Type target_ty);

void PopulateTypeConverterWithPrimeIR(mlir::LLVMTypeConverter& converter);

template <size_t N>
llvm::APInt ConvertUnderlyingValueToAPInt(const zk_dtypes::BigInt<N>& value) {
  return llvm_ir::ConvertBigIntToAPInt(value);
}

template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
llvm::APInt ConvertUnderlyingValueToAPInt(T value) {
  return {sizeof(T) * 8, value};
}

template <typename T>
mlir::IntegerAttr GetMlirIntegerAttr(mlir::MLIRContext* context,
                                     const T& value) {
  if constexpr (std::is_integral_v<T>) {
    return mlir::IntegerAttr::get(
        mlir::IntegerType::get(context, sizeof(T) * 8),
        ConvertUnderlyingValueToAPInt(value));
  } else {
    auto type = mlir::IntegerType::get(context, T::kBitWidth);
    return mlir::IntegerAttr::get(type, ConvertUnderlyingValueToAPInt(value));
  }
}

template <size_t N>
llvm::SmallVector<llvm::APInt> ConvertUnderlyingValueToAPInt(
    llvm::ArrayRef<zk_dtypes::BigInt<N>> values) {
  llvm::SmallVector<llvm::APInt> ret;
  ret.reserve(values.size());
  for (const auto& value : values) {
    ret.push_back(ConvertUnderlyingValueToAPInt(value));
  }
  return ret;
}

template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
llvm::SmallVector<llvm::APInt> ConvertUnderlyingValueToAPInt(
    llvm::ArrayRef<T> values) {
  llvm::SmallVector<llvm::APInt> ret;
  ret.reserve(values.size());
  for (T value : values) {
    ret.push_back(ConvertUnderlyingValueToAPInt(value));
  }
  return ret;
}

template <typename T>
mlir::DenseIntElementsAttr GetMlirDenseIntElementsAttr(
    mlir::MLIRContext* context, llvm::ArrayRef<T> values) {
  int64_t size = static_cast<int64_t>(values.size());
  if constexpr (std::is_integral_v<T>) {
    return mlir::DenseIntElementsAttr::get(
        mlir::RankedTensorType::get(
            {size}, mlir::IntegerType::get(context, sizeof(T) * 8)),
        ConvertUnderlyingValueToAPInt(values));
  } else {
    auto type = mlir::IntegerType::get(context, T::kBitWidth);
    return mlir::DenseIntElementsAttr::get(
        mlir::RankedTensorType::get({size}, type),
        ConvertUnderlyingValueToAPInt(values));
  }
}

template <typename T>
mlir::prime_ir::mod_arith::ModArithType GetMlirModArithType(
    mlir::MLIRContext* context) {
  return mlir::prime_ir::mod_arith::ModArithType::get(
      context, GetMlirIntegerAttr(context, T::Config::kModulus));
}

template <typename T>
mlir::prime_ir::mod_arith::MontgomeryAttr GetMlirMontgomeryAttr(
    mlir::MLIRContext* context) {
  return mlir::prime_ir::mod_arith::MontgomeryAttr::get(
      context, GetMlirIntegerAttr(context, T::Config::kModulus));
}

template <typename T>
mlir::prime_ir::field::PrimeFieldType GetMlirPrimeFieldType(
    mlir::MLIRContext* context) {
  return mlir::prime_ir::field::PrimeFieldType::get(
      context, GetMlirIntegerAttr(context, T::Config::kModulus),
      T::kUseMontgomery);
}

template <typename T>
mlir::Type GetMlirExtFieldType(mlir::MLIRContext* context) {
  constexpr uint32_t kDegree = T::Config::kDegreeOverBaseField;
  using BaseField = typename T::BaseField;

  return mlir::prime_ir::field::ExtensionFieldType::get(
      context, kDegree, GetMlirPrimeFieldType<BaseField>(context),
      GetMlirIntegerAttr(context, T::Config::kNonResidue.value()));
}

template <typename T>
mlir::prime_ir::elliptic_curve::ShortWeierstrassAttr
GetMlirG1ShortWeierstrassAttr(mlir::MLIRContext* context) {
  using BaseField = typename T::BaseField;
  using UnderlyingType = typename BaseField::UnderlyingType;

  UnderlyingType a_value = T::Curve::Config::kA.value();
  UnderlyingType b_value = T::Curve::Config::kB.value();
  UnderlyingType x_value = T::Curve::Config::kX.value();
  UnderlyingType y_value = T::Curve::Config::kY.value();
  mlir::IntegerAttr a = GetMlirIntegerAttr(context, a_value);
  mlir::IntegerAttr b = GetMlirIntegerAttr(context, b_value);
  mlir::IntegerAttr x = GetMlirIntegerAttr(context, x_value);
  mlir::IntegerAttr y = GetMlirIntegerAttr(context, y_value);
  return mlir::prime_ir::elliptic_curve::ShortWeierstrassAttr::get(
      context, GetMlirPrimeFieldType<BaseField>(context), a, b, x, y);
}

template <typename T>
mlir::prime_ir::elliptic_curve::ShortWeierstrassAttr
GetMlirG2ShortWeierstrassAttr(mlir::MLIRContext* context) {
  using BaseField = typename T::BaseField;
  using BasePrimeField = typename BaseField::Config::BasePrimeField;
  using UnderlyingType = typename BasePrimeField::UnderlyingType;

  auto to_int_array = [](const std::array<BasePrimeField, 2>& values) {
    return std::array<UnderlyingType, 2>{values[0].value(), values[1].value()};
  };

  std::array<UnderlyingType, 2> a_value =
      to_int_array(T::Curve::Config::kA.values());
  std::array<UnderlyingType, 2> b_value =
      to_int_array(T::Curve::Config::kB.values());
  std::array<UnderlyingType, 2> x_value =
      to_int_array(T::Curve::Config::kX.values());
  std::array<UnderlyingType, 2> y_value =
      to_int_array(T::Curve::Config::kY.values());
  mlir::DenseIntElementsAttr a = GetMlirDenseIntElementsAttr(
      context, llvm::ArrayRef<UnderlyingType>(a_value.data(), a_value.size()));
  mlir::DenseIntElementsAttr b = GetMlirDenseIntElementsAttr(
      context, llvm::ArrayRef<UnderlyingType>(b_value.data(), b_value.size()));
  mlir::DenseIntElementsAttr x = GetMlirDenseIntElementsAttr(
      context, llvm::ArrayRef<UnderlyingType>(x_value.data(), x_value.size()));
  mlir::DenseIntElementsAttr y = GetMlirDenseIntElementsAttr(
      context, llvm::ArrayRef<UnderlyingType>(y_value.data(), y_value.size()));
  return mlir::prime_ir::elliptic_curve::ShortWeierstrassAttr::get(
      context, GetMlirExtFieldType<BaseField>(context), a, b, x, y);
}
template <typename T>
mlir::prime_ir::elliptic_curve::AffineType GetMlirAffinePointType(
    mlir::MLIRContext* context) {
  using BaseField = typename T::BaseField;
  if constexpr (BaseField::ExtensionDegree() == 1) {
    return mlir::prime_ir::elliptic_curve::AffineType::get(
        context, GetMlirG1ShortWeierstrassAttr<T>(context));
  } else {
    return mlir::prime_ir::elliptic_curve::AffineType::get(
        context, GetMlirG2ShortWeierstrassAttr<T>(context));
  }
}

template <typename T>
mlir::prime_ir::elliptic_curve::JacobianType GetMlirJacobianPointType(
    mlir::MLIRContext* context) {
  using BaseField = typename T::BaseField;
  if constexpr (BaseField::ExtensionDegree() == 1) {
    return mlir::prime_ir::elliptic_curve::JacobianType::get(
        context, GetMlirG1ShortWeierstrassAttr<T>(context));
  } else {
    return mlir::prime_ir::elliptic_curve::JacobianType::get(
        context, GetMlirG2ShortWeierstrassAttr<T>(context));
  }
}

template <typename T>
mlir::prime_ir::elliptic_curve::XYZZType GetMlirPointXyzzType(
    mlir::MLIRContext* context) {
  using BaseField = typename T::BaseField;
  if constexpr (BaseField::ExtensionDegree() == 1) {
    return mlir::prime_ir::elliptic_curve::XYZZType::get(
        context, GetMlirG1ShortWeierstrassAttr<T>(context));
  } else {
    return mlir::prime_ir::elliptic_curve::XYZZType::get(
        context, GetMlirG2ShortWeierstrassAttr<T>(context));
  }
}

template <typename T>
mlir::Type GetMlirEcPointType(mlir::MLIRContext* context) {
  if constexpr (zk_dtypes::IsAffinePoint<T>) {
    return GetMlirAffinePointType<T>(context);
  } else if constexpr (zk_dtypes::IsJacobianPoint<T>) {
    return GetMlirJacobianPointType<T>(context);
  } else {
    return GetMlirPointXyzzType<T>(context);
  }
}

// Converts a ZKX primitive type to the corresponding MLIR type.
//
// - Signed/unsigned ZKX primitive types → signless MLIR types
mlir::Type PrimitiveTypeToMlirType(PrimitiveType element_type,
                                   mlir::MLIRContext* context);

// Converts a ZKX primitive type to the corresponding MLIR type,
// preserving signedness where applicable.
//
// - Unsigned ZKX primitive types → unsigned MLIR types
// - Other types → same as PrimitiveTypeToMLIRType()
mlir::Type PrimitiveTypeToMlirTypeWithSign(PrimitiveType element_type,
                                           mlir::MLIRContext* context);

// Returns the MLIR memref type which represents the given ZKX shape. For
// example, if "shape" is [5 x [10 x i32]], the function returns [5 x 10 x i32].
mlir::MemRefType ShapeToMlirMemRefType(const Shape& shape,
                                       mlir::MLIRContext* context);

// Returns the MLIR tensor type which represents the given ZKX shape. For
// example, if "shape" is [5 x [10 x i32]], the function returns [5 x 10 x i32].
mlir::RankedTensorType ShapeToMlirTensorType(const Shape& shape,
                                             mlir::MLIRContext* context);

llvm::SmallVector<mlir::Type> ShapeToMlirTypes(const Shape& shape,
                                               mlir::MLIRContext* context);

// Returns a ZKX PrimitiveType equivalent of an MLIR Type that represents
// a primitive type (e.g., i8), else returns PRIMITIVE_TYPE_INVALID.
// Signless MLIR types are converted to signed ZKX primitive types.
PrimitiveType MlirTypeToPrimitiveTypeWithSign(mlir::Type type);

template <typename T>
mlir::Value CreateMlirPrimeFieldConstant(mlir::ImplicitLocOpBuilder& b,
                                         const T& value) {
  return b.create<mlir::prime_ir::field::ConstantOp>(
      GetMlirPrimeFieldType<T>(b.getContext()),
      llvm_ir::ConvertBigIntToAPInt(value.value()));
}

template <typename T>
mlir::Value CreateMlirQuadraticExtFieldConstant(mlir::ImplicitLocOpBuilder& b,
                                                const T& value) {
  using BaseField = typename T::BaseField;
  using UnderlyingType = typename BaseField::UnderlyingType;

  std::array<UnderlyingType, 2> values{value[0].value(), value[1].value()};
  return b.create<mlir::prime_ir::field::ConstantOp>(
      GetMlirExtFieldType<T>(b.getContext()),
      GetMlirDenseIntElementsAttr(
          b.getContext(),
          llvm::ArrayRef<UnderlyingType>(values.data(), values.size())));
}

template <typename T>
mlir::Value CreateMlirFieldConstant(mlir::ImplicitLocOpBuilder& b,
                                    const T& value) {
  if constexpr (T::ExtensionDegree() == 1) {
    return CreateMlirPrimeFieldConstant(b, value);
  } else {
    static_assert(T::ExtensionDegree() == 2);
    return CreateMlirQuadraticExtFieldConstant(b, value);
  }
}

template <typename T>
mlir::Value CreateMlirEcPointConstant(mlir::ImplicitLocOpBuilder& b,
                                      const T& value) {
  if constexpr (zk_dtypes::IsAffinePoint<T>) {
    llvm::SmallVector<mlir::Value, 2> values;
    values.push_back(CreateMlirFieldConstant(b, value.x()));
    values.push_back(CreateMlirFieldConstant(b, value.y()));
    return b.create<mlir::prime_ir::elliptic_curve::FromCoordsOp>(
        GetMlirAffinePointType<T>(b.getContext()), values);
  } else if constexpr (zk_dtypes::IsJacobianPoint<T>) {
    llvm::SmallVector<mlir::Value, 3> values;
    values.push_back(CreateMlirFieldConstant(b, value.x()));
    values.push_back(CreateMlirFieldConstant(b, value.y()));
    values.push_back(CreateMlirFieldConstant(b, value.z()));
    return b.create<mlir::prime_ir::elliptic_curve::FromCoordsOp>(
        GetMlirJacobianPointType<T>(b.getContext()), values);
  } else {
    static_assert(zk_dtypes::IsPointXyzz<T>);
    llvm::SmallVector<mlir::Value, 4> values;
    values.push_back(CreateMlirFieldConstant(b, value.x()));
    values.push_back(CreateMlirFieldConstant(b, value.y()));
    values.push_back(CreateMlirFieldConstant(b, value.zz()));
    values.push_back(CreateMlirFieldConstant(b, value.zzz()));
    return b.create<mlir::prime_ir::elliptic_curve::FromCoordsOp>(
        GetMlirPointXyzzType<T>(b.getContext()), values);
  }
}

mlir::arith::CmpIPredicate CreateMlirArithCmpIPredicate(
    ComparisonDirection direction, bool is_signed);

}  // namespace zkx::mlir_utils

#endif  // ZKX_MLIR_MLIR_UTILS_H_
