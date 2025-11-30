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

#include "zkir/Dialect/Field/Conversions/FieldToModArith/Extension/ExtensionField.h"

#include "llvm/Support/ErrorHandling.h"
#include "zkir/Dialect/Field/Conversions/FieldToModArith/Extension/CubicExtensionField.h"
#include "zkir/Dialect/Field/IR/FieldAttributes.h"
#include "zkir/Dialect/ModArith/IR/ModArithOps.h"

namespace mlir::zkir::field {

// static
std::unique_ptr<ExtensionField>
ExtensionField::create(ImplicitLocOpBuilder &b,
                       ExtensionFieldTypeInterface type,
                       const TypeConverter *converter) {
  std::unique_ptr<ExtensionField> ret;
  if (isa<CubicExtFieldType>(type)) {
    ret.reset(new CubicExtensionField(b, type, converter));
  } else {
    llvm_unreachable("Unsupported extension field type");
  }
  return ret;
}

ExtensionField::ExtensionField(ImplicitLocOpBuilder &b,
                               ExtensionFieldTypeInterface type,
                               const TypeConverter *converter)
    : b(b), type(type) {
  // TODO(chokobole): Support towers of extension field.
  nonResidue = b.create<mod_arith::ConstantOp>(
      converter->convertType(type.getBaseFieldType()),
      cast<PrimeFieldAttr>(type.getNonResidue()).getValue());
}

} // namespace mlir::zkir::field
