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
#include "mlir/Support/LLVM.h"
#include "zkir/Dialect/Field/Conversions/FieldToModArith/ConversionUtils.h"
#include "zkir/Dialect/Field/Conversions/FieldToModArith/Extension/ExtensionFieldImpl.h"
#include "zkir/Dialect/Field/IR/FieldAttributes.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"
#include "zkir/Dialect/ModArith/IR/ModArithOps.h"

namespace mlir::zkir::field {

// static
std::unique_ptr<ExtensionField>
ExtensionField::create(ImplicitLocOpBuilder &b,
                       ExtensionFieldTypeInterface type,
                       const TypeConverter *converter) {
  static_assert(kMaxExtDegree == 4,
                "Update switch cases below when changing kMaxExtDegree");
  std::unique_ptr<ExtensionField> ret;
  if (auto efType = dyn_cast<ExtensionFieldType>(type)) {
    switch (efType.getDegree()) {
    case 2:
      ret.reset(new QuadraticExtensionField(b, type, converter));
      break;
    case 3:
      ret.reset(new CubicExtensionField(b, type, converter));
      break;
    case 4:
      ret.reset(new QuarticExtensionField(b, type, converter));
      break;
    default:
      llvm_unreachable("Unsupported extension field degree");
    }
  } else if (isa<QuadraticExtFieldType>(type)) {
    ret.reset(new QuadraticExtensionField(b, type, converter));
  } else if (isa<CubicExtFieldType>(type)) {
    ret.reset(new CubicExtensionField(b, type, converter));
  } else if (isa<QuarticExtFieldType>(type)) {
    ret.reset(new QuarticExtensionField(b, type, converter));
  } else {
    llvm_unreachable("Unsupported extension field type");
  }
  return ret;
}

ExtensionField::ExtensionField(ImplicitLocOpBuilder &b,
                               ExtensionFieldTypeInterface type,
                               const TypeConverter *converter)
    : b(b), type(type), converter(converter) {
  // TODO(chokobole): Support towers of extension field.
  nonResidue = b.create<mod_arith::ConstantOp>(
      converter->convertType(type.getBaseFieldType()),
      cast<IntegerAttr>(type.getNonResidue()));
}

Value ExtensionField::add(Value x, Value y) {
  auto xCoeffs = toCoeffs(b, x);
  auto yCoeffs = toCoeffs(b, y);
  SmallVector<Value, kMaxDegreeOverBaseField> retCoeffs;
  for (unsigned i = 0; i < type.getDegreeOverBase(); ++i) {
    // TODO(chokobole): Support towers of extension field.
    retCoeffs.push_back(b.create<mod_arith::AddOp>(xCoeffs[i], yCoeffs[i]));
  }
  return fromCoeffs(b, type, retCoeffs);
}

Value ExtensionField::sub(Value x, Value y) {
  auto xCoeffs = toCoeffs(b, x);
  auto yCoeffs = toCoeffs(b, y);
  SmallVector<Value, kMaxDegreeOverBaseField> retCoeffs;
  for (unsigned i = 0; i < type.getDegreeOverBase(); ++i) {
    // TODO(chokobole): Support towers of extension field.
    retCoeffs.push_back(b.create<mod_arith::SubOp>(xCoeffs[i], yCoeffs[i]));
  }
  return fromCoeffs(b, type, retCoeffs);
}

Value ExtensionField::dbl(Value x) {
  auto coeffs = toCoeffs(b, x);
  SmallVector<Value, kMaxDegreeOverBaseField> retCoeffs;
  for (unsigned i = 0; i < type.getDegreeOverBase(); ++i) {
    // TODO(chokobole): Support towers of extension field.
    retCoeffs.push_back(b.create<mod_arith::DoubleOp>(coeffs[i]));
  }
  return fromCoeffs(b, type, retCoeffs);
}

Value ExtensionField::negate(Value x) {
  auto coeffs = toCoeffs(b, x);
  SmallVector<Value, kMaxDegreeOverBaseField> retCoeffs;
  for (unsigned i = 0; i < type.getDegreeOverBase(); ++i) {
    // TODO(chokobole): Support towers of extension field.
    retCoeffs.push_back(b.create<mod_arith::NegateOp>(coeffs[i]));
  }
  return fromCoeffs(b, type, retCoeffs);
}

Value ExtensionField::frobeniusMap(Value x, const APInt &exponent) {
  llvm_unreachable("frobeniusMap not implemented for this extension field");
}

template class ExtensionFieldImpl<2>;
template class ExtensionFieldImpl<3>;
template class ExtensionFieldImpl<4>;

} // namespace mlir::zkir::field
