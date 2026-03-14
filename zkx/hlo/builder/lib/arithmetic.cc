/* Copyright 2017 The OpenXLA Authors.
Copyright 2026 The ZKX Authors.

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

#include "zkx/hlo/builder/lib/arithmetic.h"

#include <climits>
#include <memory>
#include <numeric>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/builder/lib/constants.h"
#include "zkx/primitive_util.h"
#include "zkx/shape_util.h"

namespace zkx {

ZkxComputation CreateScalarComputation(const std::string& name,
                                       PrimitiveType type, ZkxBuilder* builder,
                                       ZkxOpGenerator generator) {
  std::unique_ptr<ZkxBuilder> b;
  if (type == PRED) {
    b = builder->CreateSubBuilder(name);
  } else {
    b = builder->CreateSubBuilder(
        absl::StrCat(name, "_", PrimitiveType_Name(type)));
  }

  const Shape scalar = ShapeUtil::MakeShape(type, {});
  auto lhs = Parameter(b.get(), 0, scalar, "lhs");
  auto rhs = Parameter(b.get(), 1, scalar, "rhs");
  generator(lhs, rhs);
  return b->BuildAndNoteError();
}

ZkxComputation CreateScalarAddComputation(PrimitiveType type,
                                          ZkxBuilder* builder) {
  return CreateScalarComputation(
      "add", type, builder, [](ZkxOp lhs, ZkxOp rhs) { return Add(lhs, rhs); });
}

ZkxComputation CreateScalarMultiplyComputation(PrimitiveType type,
                                               ZkxBuilder* builder) {
  return CreateScalarComputation(
      "mul", type, builder, [](ZkxOp lhs, ZkxOp rhs) { return Mul(lhs, rhs); });
}

ZkxComputation CreateScalarGeComputation(PrimitiveType type,
                                         ZkxBuilder* builder) {
  return CreateScalarComputation(
      "ge", type, builder, [](ZkxOp lhs, ZkxOp rhs) { return Ge(lhs, rhs); });
}

ZkxComputation CreateScalarMaxComputation(PrimitiveType type,
                                          ZkxBuilder* builder) {
  return CreateScalarComputation(
      "max", type, builder, [](ZkxOp lhs, ZkxOp rhs) { return Max(lhs, rhs); });
}

ZkxComputation CreateScalarMinComputation(PrimitiveType type,
                                          ZkxBuilder* builder) {
  return CreateScalarComputation(
      "min", type, builder, [](ZkxOp lhs, ZkxOp rhs) { return Min(lhs, rhs); });
}

ZkxComputation CreateScalarAndComputation(PrimitiveType type,
                                          ZkxBuilder* builder) {
  return CreateScalarComputation(
      "and", type, builder, [](ZkxOp lhs, ZkxOp rhs) { return And(lhs, rhs); });
}

ZkxComputation CreateScalarOrComputation(PrimitiveType type,
                                         ZkxBuilder* builder) {
  return CreateScalarComputation(
      "or", type, builder, [](ZkxOp lhs, ZkxOp rhs) { return Or(lhs, rhs); });
}

ZkxComputation CreateScalarIdentityWithZeroComputation(PrimitiveType type,
                                                       ZkxBuilder* builder) {
  return (primitive_util::IsIntegralType(type) || type == PRED)
             ? CreateScalarOrComputation(type, builder)
             : CreateScalarAddComputation(type, builder);
}

ZkxOp Any(ZkxOp predicates) {
  ZkxBuilder* builder = predicates.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    auto f = ConstantR0<bool>(builder, false);
    ZkxComputation logical_or = CreateScalarOrComputation(PRED, builder);
    TF_ASSIGN_OR_RETURN(const Shape& predicates_shape,
                        builder->GetShape(predicates));
    std::vector<int64_t> all_dimensions(predicates_shape.rank());
    std::iota(all_dimensions.begin(), all_dimensions.end(), 0);
    return Reduce(predicates, f, logical_or, all_dimensions);
  });
}

namespace {
ZkxComputation CreateMinMaxComputation(ZkxBuilder* outer_builder,
                                       PrimitiveType value_type,
                                       PrimitiveType index_type, bool is_min) {
  auto sub_builder = outer_builder->CreateSubBuilder("minmax_func");
  ZkxBuilder* b = sub_builder.get();
  ZkxOp lhs_value =
      Parameter(b, 0, ShapeUtil::MakeShape(value_type, {}), "lhs_value");
  ZkxOp lhs_index =
      Parameter(b, 1, ShapeUtil::MakeShape(index_type, {}), "lhs_index");
  ZkxOp rhs_value =
      Parameter(b, 2, ShapeUtil::MakeShape(value_type, {}), "rhs_value");
  ZkxOp rhs_index =
      Parameter(b, 3, ShapeUtil::MakeShape(index_type, {}), "rhs_index");

  ZkxOp cmp = is_min ? Le(lhs_value, rhs_value) : Ge(lhs_value, rhs_value);
  ZkxOp max = Select(cmp, lhs_value, rhs_value);
  ZkxOp arg_max = Select(cmp, lhs_index, rhs_index);
  ZkxOp eq = Eq(lhs_value, rhs_value);
  ZkxOp tie_id = Min(lhs_index, rhs_index);
  arg_max = Select(eq, tie_id, arg_max);
  Tuple(b, {max, arg_max});
  return b->BuildAndNoteError();
}
}  // namespace

ZkxOp ArgMinMax(ZkxOp input, PrimitiveType output_type, int axis, bool is_min) {
  ZkxBuilder* builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(Shape input_shape, builder->GetShape(input));
    ZkxOp value_init_value =
        is_min ? MaxValue(builder, input_shape.element_type())
               : MinValue(builder, input_shape.element_type());
    int64_t dimension_size = input_shape.dimensions(axis);
    auto index_type = dimension_size <= INT32_MAX ? S32 : output_type;
    ZkxOp index_init_value = Zero(builder, index_type);
    auto iota_shape =
        ShapeUtil::MakeShape(index_type, input_shape.dimensions());
    ZkxOp iota = Iota(builder, iota_shape, axis);

    ZkxComputation reducer = CreateMinMaxComputation(
        builder, input_shape.element_type(), index_type, is_min);
    ZkxOp max_argmax =
        Reduce(builder, {input, iota}, {value_init_value, index_init_value},
               reducer, /*dimensions_to_reduce=*/{axis});
    ZkxOp argmax = GetTupleElement(max_argmax, 1);
    if (index_type != output_type) {
      argmax = ConvertElementType(argmax, output_type);
    }
    return argmax;
  });
}

ZkxOp ArgMax(ZkxOp input, PrimitiveType output_type, int axis) {
  return ArgMinMax(input, output_type, axis, /*is_min=*/false);
}

}  // namespace zkx
