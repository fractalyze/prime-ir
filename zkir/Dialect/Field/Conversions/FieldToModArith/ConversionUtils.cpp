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

#include "zkir/Dialect/Field/Conversions/FieldToModArith/ConversionUtils.h"

#include "mlir/Support/LLVM.h"
#include "zkir/Dialect/Field/IR/FieldOperation.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkir/Dialect/ModArith/IR/ModArithOps.h"

namespace mlir::zkir::field {
namespace {

SmallVector<Type> coeffsTypeRange(Type type) {
  auto extField = cast<ExtensionFieldTypeInterface>(type);
  auto baseField = cast<PrimeFieldType>(extField.getBaseFieldType());
  return SmallVector<Type>(extField.getDegreeOverBase(),
                           convertPrimeFieldType(baseField));
}

} // namespace

Operation::result_range toCoeffs(ImplicitLocOpBuilder &b,
                                 Value extFieldElement) {
  return b
      .create<ExtToCoeffsOp>(coeffsTypeRange(extFieldElement.getType()),
                             extFieldElement)
      .getResults();
}

Value fromCoeffs(ImplicitLocOpBuilder &b, Type type, ValueRange coeffs) {
  return b.create<ExtFromCoeffsOp>(type, coeffs);
}

Value createConst(ImplicitLocOpBuilder &b, PrimeFieldType baseField,
                  int64_t n) {
  return b.create<mod_arith::ConstantOp>(
      convertPrimeFieldType(baseField),
      PrimeFieldOperation(n, baseField).getIntegerAttr());
}

Value createInvConst(ImplicitLocOpBuilder &b, PrimeFieldType baseField,
                     int64_t n) {
  return b.create<mod_arith::ConstantOp>(
      convertPrimeFieldType(baseField),
      PrimeFieldOperation(n, baseField).inverse().getIntegerAttr());
}

Value createRationalConst(ImplicitLocOpBuilder &b, PrimeFieldType baseField,
                          int64_t num, int64_t denom) {
  return b.create<mod_arith::ConstantOp>(convertPrimeFieldType(baseField),
                                         (PrimeFieldOperation(num, baseField) /
                                          PrimeFieldOperation(denom, baseField))
                                             .getIntegerAttr());
}

} // namespace mlir::zkir::field
