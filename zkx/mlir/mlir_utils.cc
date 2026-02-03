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

#include "zkx/mlir/mlir_utils.h"

#include "absl/base/optimization.h"
#include "absl/log/log.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"

#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToLLVM/EllipticCurveToLLVM.h"
#include "prime_ir/Dialect/Field/Conversions/ExtFieldToLLVM/ExtFieldToLLVM.h"
#include "zk_dtypes/include/all_types.h"
#include "zkx/layout_util.h"
#include "zkx/primitive_util.h"

namespace zkx::mlir_utils {

mlir::Value GetConstantOrSplat(mlir::ImplicitLocOpBuilder& b, mlir::Type t,
                               mlir::Attribute v) {
  if (auto shaped_type = mlir::dyn_cast<mlir::ShapedType>(t)) {
    v = mlir::SplatElementsAttr::get(shaped_type, v);
  }
  return b.create<mlir::arith::ConstantOp>(t, mlir::cast<mlir::TypedAttr>(v));
}

void PopulateTypeConverterWithPrimeIR(mlir::LLVMTypeConverter& converter) {
  mlir::prime_ir::field::populateExtFieldToLLVMTypeConversion(converter);
  mlir::prime_ir::elliptic_curve::populateEllipticCurveToLLVMTypeConversion(
      converter);
}

mlir::Type PrimitiveTypeToMlirType(PrimitiveType element_type,
                                   mlir::MLIRContext* context) {
  switch (element_type) {
    case PRED:
    case S1:
    case U1:
    case S2:
    case U2:
    case S4:
    case U4:
    case S8:
    case U8:
    case S16:
    case U16:
    case S32:
    case U32:
    case S64:
    case U64:
    case U128:
    case U256:
      return mlir::IntegerType::get(context,
                                    primitive_util::BitWidth(element_type));
    // TODO(chokobole): For Tuple, see the comments in
    // ShapeToMlirMemRefType().
    case TUPLE:
    // An Opaque is like a void*, use i8*.
    case OPAQUE_TYPE:
      return mlir::MemRefType::get({1}, mlir::IntegerType::get(context, 8));
    case TOKEN:
      // Tokens do not have a physical representation, but the compiler needs
      // some placeholder type, so use int8_t*.
      return mlir::MemRefType::get({1}, mlir::IntegerType::get(context, 8));
#define ZK_DTYPES_CASE(cpp_type, unused, enum, unused2) \
  case enum:                                            \
    return GetMlirPrimeFieldType<cpp_type>(context);
      ZK_DTYPES_PUBLIC_PRIME_FIELD_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE

#define ZK_DTYPES_CASE(cpp_type, unused, enum, unused2) \
  case enum:                                            \
    return GetMlirEcPointType<cpp_type>(context);
      ZK_DTYPES_PUBLIC_EC_POINT_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE

#define ZK_DTYPES_CASE(cpp_type, unused, enum, unused2) \
  case enum:                                            \
    return GetMlirExtFieldType<cpp_type>(context);
      ZK_DTYPES_PUBLIC_EXT_FIELD_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE
    default:
      LOG(FATAL) << "unsupported type " << element_type;
  }
}

mlir::Type PrimitiveTypeToMlirTypeWithSign(PrimitiveType element_type,
                                           mlir::MLIRContext* context) {
  if (element_type == PRED) {
    return mlir::IntegerType::get(context, 1);
  } else if (primitive_util::IsIntegralType(element_type)) {
    return mlir::IntegerType::get(
        context,
        /*width=*/primitive_util::BitWidth(element_type),
        /*signed=*/
        primitive_util::IsUnsignedIntegralType(element_type)
            ? mlir::IntegerType::Unsigned
            : mlir::IntegerType::Signless);
  }
  // Delegate to the other function for non-integer and signed integer
  // types.
  return PrimitiveTypeToMlirType(element_type, context);
}

mlir::MemRefType ShapeToMlirMemRefType(const Shape& shape,
                                       mlir::MLIRContext* context) {
  CHECK(shape.IsArray());
  CHECK(shape.is_static())
      << "ShapeToMlirMemRefType only supports static shapes.";
  const Layout& layout = shape.layout();
  mlir::Type element_type =
      PrimitiveTypeToMlirType(shape.element_type(), context);
  // TODO(chokobole): Take `major_to_minor` into account.
  auto dimensions_span = shape.dimensions();
  llvm::ArrayRef<int64_t> dimensions(dimensions_span.data(),
                                     dimensions_span.size());
  if (LayoutUtil::IsMonotonicWithDim0Major(layout)) {
    return mlir::MemRefType::get(dimensions, element_type);
  }

  llvm::SmallVector<int64_t> strides(dimensions.size(), 0);
  int64_t running = 1;
  for (int64_t d : layout.minor_to_major()) {
    strides[d] = running;
    // NOTE(chokobole): CHECK_GE(dimensions[d], 0) is used to allow for cases
    // where a dimension can legitimately have a size of zero. For example, an
    // empty array [2, 0, 3] is a valid shape.
    CHECK_GE(dimensions[d], 0);
    running *= dimensions[d];
  }

  auto strided = mlir::StridedLayoutAttr::get(context, /*offset=*/0, strides);
  return mlir::MemRefType::get(dimensions, element_type, strided);
}

namespace {

mlir::DenseIntElementsAttr CreateDenseIntElementsAttrFromVector(
    mlir::MLIRContext* context, const llvm::ArrayRef<int64_t> vector) {
  return mlir::DenseIntElementsAttr::get(
      mlir::RankedTensorType::get(vector.size(),
                                  mlir::IntegerType::get(context, 64)),
      vector);
}

}  // namespace

mlir::RankedTensorType ShapeToMlirTensorType(const Shape& shape,
                                             mlir::MLIRContext* context) {
  CHECK(shape.IsArray());
  const Layout& layout = shape.layout();
  mlir::Type element_type =
      PrimitiveTypeToMlirType(shape.element_type(), context);

  auto dimensions_span = shape.dimensions();
  llvm::ArrayRef<int64_t> dimensions(dimensions_span.data(),
                                     dimensions_span.size());
  if (LayoutUtil::IsSparse(layout)) {
    auto convert_to_mlir_level = [](DimLevelType dlt, bool ordered,
                                    bool unique) {
      switch (dlt) {
        case DimLevelType::DIM_DENSE:
          return *mlir::sparse_tensor::buildLevelType(
              mlir::sparse_tensor::LevelFormat::Dense, ordered, unique);
        case DimLevelType::DIM_COMPRESSED:
          return *mlir::sparse_tensor::buildLevelType(
              mlir::sparse_tensor::LevelFormat::Compressed, ordered, unique);
        case DimLevelType::DIM_SINGLETON:
          return *mlir::sparse_tensor::buildLevelType(
              mlir::sparse_tensor::LevelFormat::Singleton, ordered, unique);
        case DimLevelType::DIM_LOOSE_COMPRESSED:
          return *mlir::sparse_tensor::buildLevelType(
              mlir::sparse_tensor::LevelFormat::LooseCompressed, ordered,
              unique);
        case DimLevelType_INT_MIN_SENTINEL_DO_NOT_USE_:
        case DimLevelType_INT_MAX_SENTINEL_DO_NOT_USE_:
          break;
      }
      ABSL_UNREACHABLE();
      return mlir::sparse_tensor::LevelType(0);
    };

    llvm::SmallVector<mlir::sparse_tensor::LevelType> lts;
    for (int i = 0; i < layout.dim_level_types_size(); ++i) {
      DimLevelType dlt = layout.dim_level_type(i);
      bool ordered =
          i < layout.dim_ordered_size() ? layout.dim_ordered(i) : true;
      bool unique = i < layout.dim_unique_size() ? layout.dim_unique(i) : true;
      lts.push_back(convert_to_mlir_level(dlt, ordered, unique));
    }
    absl::Span<const int64_t> ordering = layout.minor_to_major();
    llvm::SmallVector<int64_t> major_to_minor = {ordering.rbegin(),
                                                 ordering.rend()};
    mlir::AffineMap id_map =
        mlir::AffineMap::getPermutationMap(major_to_minor, context);
    auto encoding = mlir::sparse_tensor::SparseTensorEncodingAttr::get(
        context, lts, id_map, mlir::AffineMap(), 32, 32);
    return mlir::RankedTensorType::get(dimensions, element_type, encoding);
  } else {
    // Default layouts create a lot of clutter in the IR, so only add an
    // encoding when needed.
    mlir::Attribute layout = {};
    if (!LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
      layout = CreateDenseIntElementsAttrFromVector(
          context, llvm::to_vector(shape.layout().minor_to_major()));
    }
    return mlir::RankedTensorType::get(dimensions, element_type, layout);
  }
}

llvm::SmallVector<mlir::Type> ShapeToMlirTypes(const Shape& shape,
                                               mlir::MLIRContext* context) {
  llvm::SmallVector<mlir::Type> types;
  types.reserve(shape.IsTuple() ? shape.tuple_shapes_size() : 1);
  if (shape.IsTuple()) {
    for (const Shape& tuple_shape : shape.tuple_shapes()) {
      if (tuple_shape.IsTuple()) {
        types.append(ShapeToMlirTypes(tuple_shape, context));
      } else {
        types.push_back(ShapeToMlirTensorType(tuple_shape, context));
      }
    }
  } else {
    types.push_back(ShapeToMlirTensorType(shape, context));
  }
  return types;
}

namespace {
PrimitiveType GetPrimitiveTypeOfPrimeFieldType(
    mlir::prime_ir::field::PrimeFieldType field_type) {
#define ZK_DTYPES_CASE(cpp_type, unused, enum, unused2)           \
  {                                                               \
    mlir::prime_ir::field::PrimeFieldType this_field_type =       \
        GetMlirPrimeFieldType<cpp_type>(field_type.getContext()); \
    if (field_type == this_field_type) {                          \
      return PrimitiveType::enum;                                 \
    }                                                             \
  }
  ZK_DTYPES_PUBLIC_PRIME_FIELD_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE
  return PrimitiveType::PRIMITIVE_TYPE_INVALID;
}

PrimitiveType GetPrimitiveTypeOfAffineType(
    mlir::prime_ir::elliptic_curve::AffineType affine_type) {
#define ZK_DTYPES_CASE(cpp_type, unused, enum, unused2)             \
  {                                                                 \
    mlir::prime_ir::elliptic_curve::AffineType this_affine_type =   \
        GetMlirAffinePointType<cpp_type>(affine_type.getContext()); \
    if (affine_type == this_affine_type) {                          \
      return PrimitiveType::enum;                                   \
    }                                                               \
  }
  ZK_DTYPES_PUBLIC_EC_POINT_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE
  return PrimitiveType::PRIMITIVE_TYPE_INVALID;
}

PrimitiveType GetPrimitiveTypeOfJacobianType(
    mlir::prime_ir::elliptic_curve::JacobianType jacobian_type) {
#define ZK_DTYPES_CASE(cpp_type, unused, enum, unused2)                 \
  {                                                                     \
    mlir::prime_ir::elliptic_curve::JacobianType this_jacobian_type =   \
        GetMlirJacobianPointType<cpp_type>(jacobian_type.getContext()); \
    if (jacobian_type == this_jacobian_type) {                          \
      return PrimitiveType::enum;                                       \
    }                                                                   \
  }
  ZK_DTYPES_PUBLIC_EC_POINT_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE
  return PrimitiveType::PRIMITIVE_TYPE_INVALID;
}

PrimitiveType GetPrimitiveTypeOfXYZZType(
    mlir::prime_ir::elliptic_curve::XYZZType xyzz_type) {
#define ZK_DTYPES_CASE(cpp_type, unused, enum, unused2)         \
  {                                                             \
    mlir::prime_ir::elliptic_curve::XYZZType this_xyzz_type =   \
        GetMlirPointXyzzType<cpp_type>(xyzz_type.getContext()); \
    if (xyzz_type == this_xyzz_type) {                          \
      return PrimitiveType::enum;                               \
    }                                                           \
  }
  ZK_DTYPES_PUBLIC_EC_POINT_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE
  return PrimitiveType::PRIMITIVE_TYPE_INVALID;
}

}  // namespace

PrimitiveType MlirTypeToPrimitiveTypeWithSign(mlir::Type type) {
  if (auto integer_type = mlir::dyn_cast<mlir::IntegerType>(type)) {
    bool is_unsigned = integer_type.isUnsigned();
    if (integer_type.getWidth() == 1) {
      return PrimitiveType::PRED;
    }
    return is_unsigned ? primitive_util::UnsignedIntegralTypeForBitWidth(
                             integer_type.getWidth())
                       : primitive_util::SignedIntegralTypeForBitWidth(
                             integer_type.getWidth());
  } else if (auto field_type =
                 mlir::dyn_cast<mlir::prime_ir::field::PrimeFieldType>(type)) {
    return GetPrimitiveTypeOfPrimeFieldType(field_type);
  } else if (auto affine_type =
                 mlir::dyn_cast<mlir::prime_ir::elliptic_curve::AffineType>(
                     type)) {
    return GetPrimitiveTypeOfAffineType(affine_type);
  } else if (auto jacobian_type =
                 mlir::dyn_cast<mlir::prime_ir::elliptic_curve::JacobianType>(
                     type)) {
    return GetPrimitiveTypeOfJacobianType(jacobian_type);
  } else if (auto xyzz_type =
                 mlir::dyn_cast<mlir::prime_ir::elliptic_curve::XYZZType>(
                     type)) {
    return GetPrimitiveTypeOfXYZZType(xyzz_type);
  }
  return PrimitiveType::PRIMITIVE_TYPE_INVALID;
}

mlir::arith::CmpIPredicate CreateMlirArithCmpIPredicate(
    ComparisonDirection direction, bool is_signed) {
  switch (direction) {
    case ComparisonDirection::kEq:
      return mlir::arith::CmpIPredicate::eq;
    case ComparisonDirection::kNe:
      return mlir::arith::CmpIPredicate::ne;
    case ComparisonDirection::kLt:
      return is_signed ? mlir::arith::CmpIPredicate::slt
                       : mlir::arith::CmpIPredicate::ult;
    case ComparisonDirection::kLe:
      return is_signed ? mlir::arith::CmpIPredicate::sle
                       : mlir::arith::CmpIPredicate::ule;
    case ComparisonDirection::kGt:
      return is_signed ? mlir::arith::CmpIPredicate::sgt
                       : mlir::arith::CmpIPredicate::ugt;
    case ComparisonDirection::kGe:
      return is_signed ? mlir::arith::CmpIPredicate::sge
                       : mlir::arith::CmpIPredicate::uge;
  }
}

}  // namespace zkx::mlir_utils
