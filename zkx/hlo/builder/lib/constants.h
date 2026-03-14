/* Copyright 2018 The OpenXLA Authors.
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

#ifndef ZKX_HLO_BUILDER_LIB_CONSTANTS_H_
#define ZKX_HLO_BUILDER_LIB_CONSTANTS_H_

#include "absl/status/statusor.h"

#include "zkx/hlo/builder/zkx_builder.h"
#include "zkx/primitive_util.h"
#include "zkx/shape.h"
#include "zkx/shape_util.h"
#include "zkx/zkx_data.pb.h"

namespace zkx {

// Returns scalar 'value' as a scalar of 'type'. Unlike ConstantR0, 'type' is
// determined at C++ run-time, rather than C++ compile-time.
template <typename T>
ZkxOp ConstantR0WithType(ZkxBuilder* builder, PrimitiveType type, T value) {
  return primitive_util::PrimitiveTypeSwitch<ZkxOp>(
      [&](auto primitive_type_constant) -> ZkxOp {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          return ConstantR0<NativeT>(builder, static_cast<NativeT>(value));
        }
        return builder->ReportError(absl::InvalidArgumentError(
            absl::StrCat("Invalid type for ConstantR0WithType (",
                         PrimitiveType_Name(type), ").")));
      },
      type);
}

// Returns a scalar containing 'value' cast to the same run-time type as
// 'prototype'.
template <typename T>
ZkxOp ScalarLike(ZkxOp prototype, T value) {
  ZkxBuilder* builder = prototype.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(prototype));
    return ConstantR0WithType(builder, shape.element_type(), value);
  });
}

// Returns an array or scalar containing copies of `value` cast to the same
// run-time type as `prototype` and broadcast to the same dimensions as
// `prototype`.
//
// If `prototype` is not a scalar or array, returns an error.
template <typename T>
ZkxOp FullLike(ZkxOp prototype, T value) {
  ZkxBuilder* builder = prototype.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(prototype));
    if (ShapeUtil::IsScalar(shape) || shape.IsArray()) {
      return Broadcast(ScalarLike(prototype, value), shape.dimensions());
    } else {
      return absl::InvalidArgumentError(absl::StrCat(
          "Prototype shape for FullLike must be a scalar or array, but was ",
          shape.ToString()));
    }
  });
}

// Returns a scalar with value '0' of 'type'.
ZkxOp Zero(ZkxBuilder* builder, PrimitiveType type);

// Returns a zero-filled tensor with shape `shape`.
ZkxOp Zeros(ZkxBuilder* builder, const Shape& shape);

// Returns a zero-filled tensor with the same shape as `prototype`.
ZkxOp ZerosLike(ZkxOp prototype);

// Returns a scalar with value '1' of 'type'.
ZkxOp One(ZkxBuilder* builder, PrimitiveType type);

// Returns the minimum representable value for 'type'.
ZkxOp MinValue(ZkxBuilder* builder, PrimitiveType type);

// Returns the maximum representable value for 'type'.
ZkxOp MaxValue(ZkxBuilder* builder, PrimitiveType type);

}  // namespace zkx

#endif  // ZKX_HLO_BUILDER_LIB_CONSTANTS_H_
