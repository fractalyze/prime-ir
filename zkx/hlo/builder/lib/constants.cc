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

#include "zkx/hlo/builder/lib/constants.h"

#include "zkx/literal_util.h"

namespace zkx {

ZkxOp Zero(ZkxBuilder* builder, PrimitiveType type) {
  return ConstantLiteral(builder, LiteralUtil::Zero(type));
}

ZkxOp Zeros(ZkxBuilder* builder, const Shape& shape) {
  return Broadcast(Zero(builder, shape.element_type()), shape.dimensions());
}

ZkxOp ZerosLike(ZkxOp prototype) {
  ZkxBuilder* builder = prototype.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<ZkxOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(prototype));
    return Zeros(builder, shape);
  });
}

ZkxOp One(ZkxBuilder* builder, PrimitiveType type) {
  return ConstantLiteral(builder, LiteralUtil::One(type));
}

ZkxOp MinValue(ZkxBuilder* builder, PrimitiveType type) {
  return ConstantLiteral(builder, LiteralUtil::MinValue(type));
}

ZkxOp MaxValue(ZkxBuilder* builder, PrimitiveType type) {
  return ConstantLiteral(builder, LiteralUtil::MaxValue(type));
}

}  // namespace zkx
