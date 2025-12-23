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
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkir/Dialect/ModArith/IR/ModArithOperation.h"
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
  APInt modulus = baseField.getModulus().getValue();
  unsigned bitWidth = baseField.getStorageBitWidth();
  auto convertedType = convertPrimeFieldType(baseField);

  assert(n > -modulus.getSExtValue() && n < modulus.getSExtValue() &&
         "n must be in range (-P, P)");

  APInt nVal;
  if (n < 0) {
    nVal = modulus - APInt(bitWidth, -n);
  } else {
    nVal = APInt(bitWidth, n);
  }

  return b.create<mod_arith::ConstantOp>(
      convertedType, IntegerAttr::get(baseField.getStorageType(), nVal));
}

Value createInvConst(ImplicitLocOpBuilder &b, PrimeFieldType baseField,
                     int64_t n) {
  APInt modulus = baseField.getModulus().getValue();
  unsigned bitWidth = baseField.getStorageBitWidth();
  auto convertedType = convertPrimeFieldType(baseField);

  assert(n > -modulus.getSExtValue() && n < modulus.getSExtValue() &&
         "n must be in range (-P, P)");

  APInt nVal;
  if (n < 0) {
    nVal = modulus - APInt(bitWidth, -n);
  } else {
    nVal = APInt(bitWidth, n);
  }

  APInt inv = mod_arith::ModArithOperation(nVal, convertedType).Inverse();
  return b.create<mod_arith::ConstantOp>(
      convertedType, IntegerAttr::get(baseField.getStorageType(), inv));
}

Value createRationalConst(ImplicitLocOpBuilder &b, PrimeFieldType baseField,
                          int64_t num, int64_t denom) {
  APInt modulus = baseField.getModulus().getValue();
  unsigned bitWidth = baseField.getStorageType().getWidth();
  auto convertedType = convertPrimeFieldType(baseField);

  assert(num > -modulus.getSExtValue() && num < modulus.getSExtValue() &&
         "num must be in range (-P, P)");
  assert(denom > -modulus.getSExtValue() && denom < modulus.getSExtValue() &&
         "denom must be in range (-P, P)");

  APInt numVal;
  if (num < 0) {
    numVal = modulus - APInt(bitWidth, -num);
  } else {
    numVal = APInt(bitWidth, num);
  }

  APInt denomVal;
  if (denom < 0) {
    denomVal = modulus - APInt(bitWidth, -denom);
  } else {
    denomVal = APInt(bitWidth, denom);
  }

  // Compute num * denom⁻¹ mod modulus
  mod_arith::ModArithOperation numOp(numVal, convertedType);
  mod_arith::ModArithOperation denomOp(denomVal, convertedType);
  APInt result = numOp / denomOp;

  return b.create<mod_arith::ConstantOp>(
      convertedType, IntegerAttr::get(baseField.getStorageType(), result));
}

} // namespace mlir::zkir::field
