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

#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/ConversionUtils.h"

#include "mlir/Support/LLVM.h"
#include "prime_ir/Dialect/Field/IR/FieldOperation.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithOps.h"

namespace mlir::prime_ir::field {
namespace {

SmallVector<Type> coeffsTypeRange(Type type) {
  auto extField = cast<ExtensionFieldType>(type);
  Type baseField = extField.getBaseField();

  // For tower extensions, the coefficients are extension field elements
  if (isa<ExtensionFieldType>(baseField)) {
    return SmallVector<Type>(extField.getDegree(), baseField);
  }

  // For direct extensions over prime field, convert to mod_arith type
  auto pfType = cast<PrimeFieldType>(baseField);
  return SmallVector<Type>(extField.getDegree(), convertPrimeFieldType(pfType));
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

Value fromPrimeCoeffs(ImplicitLocOpBuilder &b, ExtensionFieldType efType,
                      ArrayRef<Value> primeCoeffs) {
  Type baseField = efType.getBaseField();
  unsigned degree = efType.getDegree();

  // Non-tower: primeCoeffs should have exactly `degree` elements
  if (isa<PrimeFieldType>(baseField)) {
    assert(primeCoeffs.size() == degree &&
           "Expected degree prime coefficients for non-tower extension");
    return fromCoeffs(b, efType, primeCoeffs);
  }

  // Tower extension: recursively build nested structure
  auto baseEf = cast<ExtensionFieldType>(baseField);
  unsigned baseDegreeOverPrime = baseEf.getDegreeOverPrime();

  assert(primeCoeffs.size() == degree * baseDegreeOverPrime &&
         "Expected degreeOverPrime prime coefficients for tower extension");

  SmallVector<Value> baseCoeffs;
  for (unsigned i = 0; i < degree; ++i) {
    ArrayRef<Value> baseCoeffSlice =
        primeCoeffs.slice(i * baseDegreeOverPrime, baseDegreeOverPrime);
    baseCoeffs.push_back(fromPrimeCoeffs(b, baseEf, baseCoeffSlice));
  }

  return fromCoeffs(b, efType, baseCoeffs);
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

} // namespace mlir::prime_ir::field
