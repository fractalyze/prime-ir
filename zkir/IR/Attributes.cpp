/* Copyright 2025 The ZKIR Authors.

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

#include "zkir/IR/Attributes.h"

#include "zkir/Dialect/Field/IR/FieldTypes.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"

namespace mlir::zkir {

ShapedType maybeConvertZkirToBuiltinType(ShapedType type) {
  if (auto modArithType =
          dyn_cast<mod_arith::ModArithType>(type.getElementType())) {
    return type.clone(modArithType.getStorageType());
  } else if (auto fieldType =
                 dyn_cast<field::PrimeFieldType>(type.getElementType())) {
    return type.clone(fieldType.getStorageType());
  }
  return type;
}

} // namespace mlir::zkir
