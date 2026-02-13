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

// This file defines helpers useful when creating or manipulating lhlo/hlo.

#include "zkx/hlo/translate/hlo_to_mhlo/hlo_utils.h"

#include <assert.h>

#include "absl/strings/str_format.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"

#include "prime_ir/IR/Attributes.h"
#include "prime_ir/Utils/ZkDtypes.h"
#include "zkx/primitive_util.h"

namespace zkx {
namespace {

using mlir::AffineMap;

template <typename CppType>
absl::StatusOr<mlir::DenseElementsAttr> CreateDenseAttrFromLiteral(
    const mlir::ShapedType& type, const LiteralBase& literal) {
  auto data_span = literal.data<CppType>();
  if constexpr (zk_dtypes::IsPrimeField<CppType> ||
                zk_dtypes::IsBinaryField<CppType>) {
    // Prime and binary fields: single value converted to APInt.
    llvm::SmallVector<mlir::APInt> apints;
    apints.reserve(data_span.size());
    for (const auto& elem : data_span) {
      apints.push_back(
          mlir::prime_ir::convertToAPInt(elem.value(), CppType::kBitWidth));
    }
    auto builtin_type = mlir::prime_ir::maybeConvertPrimeIRToBuiltinType(type);
    return mlir::DenseElementsAttr::get(builtin_type, llvm::ArrayRef(apints));
  } else if constexpr (zk_dtypes::IsExtensionField<CppType>) {
    // Extension fields: each element has multiple base field coefficients.
    llvm::SmallVector<mlir::APInt> apints;
    for (const auto& elem : data_span) {
      auto coeffs = mlir::prime_ir::convertToAPInts(elem.values());
      apints.append(coeffs.begin(), coeffs.end());
    }
    auto builtin_type = mlir::prime_ir::maybeConvertPrimeIRToBuiltinType(type);
    return mlir::DenseElementsAttr::get(builtin_type, llvm::ArrayRef(apints));
  } else if constexpr (zk_dtypes::IsEcPoint<CppType>) {
    return absl::UnimplementedError("Not implemented for ec point types");
  } else if constexpr (is_big_int_v<CppType>) {
    // BigInt types need to be converted to APInt for MLIR.
    auto data_span = literal.data<CppType>();
    constexpr unsigned bit_width = CppType::kBitWidth;
    llvm::SmallVector<llvm::APInt, 4> apint_values;
    apint_values.reserve(data_span.size());
    for (const auto& val : data_span) {
      // Construct APInt from raw limbs.
      llvm::ArrayRef<uint64_t> limbs(reinterpret_cast<const uint64_t*>(&val),
                                     CppType::kLimbNums);
      apint_values.push_back(llvm::APInt(bit_width, limbs));
    }
    return mlir::DenseElementsAttr::get(type, llvm::ArrayRef(apint_values));
  } else {
    return mlir::DenseElementsAttr::get(
        type, llvm::ArrayRef(data_span.data(), data_span.size()));
  }
}

absl::StatusOr<AffineMap> GetPermutationIfAvailable(const Shape& shape,
                                                    mlir::Builder builder) {
  // N.B. IsMonotonicWithDim0Major ignores tiling, and I can't change it because
  // some ZKX code relies on it treating tiled layouts as equivalent to untiled
  // layouts, so the check to rule out tiling has to come /before/ the
  // early-return branch, or we'd miss tiled monotonic layouts.
  if (!shape.layout().tiles().empty()) {
    return absl::InternalError("Tiled layouts are not yet supported");
  }
  if (!shape.has_layout() ||
      LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    return AffineMap();
  }
  if (!shape.is_static()) {
    return absl::InternalError(
        "Permutations for dynamic shapes are not yet supported");
  }
  int64_t accumulated_stride = 1;
  llvm::SmallVector<int64_t, 4> strides(shape.rank(), 1);
  for (int64_t dim : LayoutUtil::MinorToMajor(shape)) {
    strides[dim] = accumulated_stride;
    accumulated_stride *= shape.dimensions(dim);
  }
  if (accumulated_stride == 0) {
    return AffineMap();
  }
  return makeStridedLinearLayoutMap(strides, /*offset=*/0,
                                    builder.getContext());
}

}  // namespace

absl::StatusOr<mlir::MemRefType> ConvertTensorShapeToMemRefType(
    const Shape& shape, mlir::Builder builder) {
  mlir::Type element_type = mlir_utils::PrimitiveTypeToMlirTypeWithSign(
      shape.element_type(), builder.getContext());

  absl::Span<const int64_t> dimensions = shape.dimensions();
  llvm::SmallVector<int64_t, 4> array(dimensions.begin(), dimensions.end());
  absl::StatusOr<AffineMap> permutation_or =
      GetPermutationIfAvailable(shape, builder);
  if (!permutation_or.ok()) return permutation_or.status();
  return mlir::MemRefType::get(array, element_type, permutation_or.value());
}

absl::StatusOr<mlir::DenseElementsAttr> CreateDenseElementsAttrFromLiteral(
    const LiteralBase& literal, mlir::Builder builder) {
  TF_ASSIGN_OR_RETURN(auto type,
                      ConvertTensorShapeToType<mlir::RankedTensorType>(
                          literal.shape(), builder));

  // TODO(hinsu): Support remaining ZKX primitive types.
  PrimitiveType element_type = literal.shape().element_type();
  return primitive_util::PrimitiveTypeSwitch<
      absl::StatusOr<mlir::DenseElementsAttr>>(
      [&](auto primitive_type_constant)
          -> absl::StatusOr<mlir::DenseElementsAttr> {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant) &&
                      !primitive_util::IsEcPointType(primitive_type_constant)) {
          return CreateDenseAttrFromLiteral<
              primitive_util::NativeTypeOf<primitive_type_constant>>(type,
                                                                     literal);
        }
        return absl::InternalError(absl::StrFormat(
            "Unsupported type: %s", PrimitiveType_Name(element_type)));
      },
      element_type);
}

mlir::DenseIntElementsAttr CreateDenseIntElementsAttrFromVector(
    const llvm::ArrayRef<int64_t> vector, mlir::Builder builder,
    llvm::ArrayRef<int64_t> shape) {
  return mlir::DenseIntElementsAttr::get(
      mlir::RankedTensorType::get(shape.empty() ? vector.size() : shape,
                                  builder.getIntegerType(64)),
      vector);
}

mlir::Value CreateTupleValue(mlir::OpBuilder* func_builder, mlir::Location loc,
                             mlir::ValueRange& flatten_values,
                             mlir::Type type) {
  auto tuple_type = llvm::dyn_cast<mlir::TupleType>(type);
  if (!tuple_type) {
    assert(!flatten_values.empty());
    auto retval = flatten_values.front();
    flatten_values = flatten_values.drop_front();
    return retval;
  }

  llvm::SmallVector<mlir::Value> flatten_sub_values;
  for (mlir::Type child_type : tuple_type.getTypes())
    flatten_sub_values.push_back(
        CreateTupleValue(func_builder, loc, flatten_values, child_type));

  return func_builder->create<mlir::mhlo::TupleOp>(loc, flatten_sub_values)
      .getResult();
}

mlir::Operation* CreateTupleFromOpResults(mlir::OpBuilder* func_builder,
                                          mlir::Location loc,
                                          mlir::Operation* op,
                                          mlir::Type type) {
  if (!llvm::isa<mlir::TupleType>(type)) return op;

  mlir::ValueRange flattened_results_ref(op->getResults());
  auto result =
      CreateTupleValue(func_builder, loc, flattened_results_ref, type);
  auto defining_tuple_op = result.getDefiningOp<mlir::mhlo::TupleOp>();
  assert(defining_tuple_op && "builder didn't return the right type");
  mlir::Operation* tuple_op = defining_tuple_op.getOperation();
  return tuple_op;
}

mlir::Operation* WrapVariadicResultsInTuple(mlir::OpBuilder* builder,
                                            mlir::Location loc,
                                            mlir::Operation* op) {
  auto result_types = op->getResultTypes();
  // Consider skipping wrapping result type of size 1.
  assert(result_types.size() != 1 ||
         !llvm::isa<mlir::TupleType>(result_types[0]) &&
             "Cannot wrap single tuple arg in tuple");

  mlir::TupleType tuple_type = builder->getTupleType(result_types);
  return CreateTupleFromOpResults(builder, loc, op, tuple_type);
}

bool IsEmptyTuple(const mlir::Type& type) {
  if (auto tuple_type = llvm::dyn_cast<mlir::TupleType>(type)) {
    return tuple_type.getTypes().empty();
  }
  return false;
}

mlir::TypeRange Untuple(const mlir::Type& type) {
  if (llvm::isa<mlir::TupleType>(type)) {
    return llvm::dyn_cast<mlir::TupleType>(type).getTypes();
  }
  return type;
}

}  // namespace zkx
