/* Copyright 2025 The PrimeIR Authors.

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

#include "prime_ir/Utils/OpUtils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::prime_ir {

Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto shapedType = dyn_cast<ShapedType>(type))
    return shapedType.cloneWith(std::nullopt, i1Type);
  if (isa<UnrankedTensorType>(type))
    return UnrankedTensorType::get(i1Type);
  return i1Type;
}

} // namespace mlir::prime_ir
