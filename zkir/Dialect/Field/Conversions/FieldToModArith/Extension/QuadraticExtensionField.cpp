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

#include "zkir/Dialect/Field/Conversions/FieldToModArith/Extension/QuadraticExtensionField.h"

#include "zkir/Dialect/Field/Conversions/FieldToModArith/ExtensionFieldCodeGen.h"

namespace mlir::zkir::field {

Value QuadraticExtensionField::square(Value x) {
  ExtensionFieldCodeGen<2> xGen(&b, x.getType(), x, nonResidue);
  return xGen.Square().getValue();
}

Value QuadraticExtensionField::mul(Value x, Value y) {
  ExtensionFieldCodeGen<2> xGen(&b, x.getType(), x, nonResidue);
  ExtensionFieldCodeGen<2> yGen(&b, y.getType(), y, nonResidue);
  return (xGen * yGen).getValue();
}

Value QuadraticExtensionField::inverse(Value x) {
  ExtensionFieldCodeGen<2> xGen(&b, x.getType(), x, nonResidue);
  return xGen.Inverse().getValue();
}

} // namespace mlir::zkir::field
