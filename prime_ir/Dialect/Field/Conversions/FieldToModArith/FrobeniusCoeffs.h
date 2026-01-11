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

#ifndef PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FROBENIUSCOEFFS_H_
#define PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FROBENIUSCOEFFS_H_

#include <array>
#include <cstddef>

#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/ConversionUtils.h"
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/PrimeFieldCodeGen.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithOperation.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithOps.h"

namespace mlir::prime_ir::field {

// Mixin providing Frobenius coefficients for extension field operations.
//
// Frobenius coefficients: coeffs[e - 1][i - 1] = ξ^(i * e * (p - 1) / n)
// where e = 1..n - 1, i = 1..n - 1, n is extension degree, p is modulus
//
// See:
// https://fractalyze.gitbook.io/intro/primitives/abstract-algebra/extension-field/inversion#id-2.2.-optimized-computation-when
//
// Derived class must provide:
//   - getBuilder(): returns ImplicitLocOpBuilder&
//   - getType(): returns Type (extension field type)
template <typename Derived, size_t N>
class FrobeniusCoeffs {
public:
  std::array<std::array<PrimeFieldCodeGen, N - 1>, N - 1>
  GetFrobeniusCoeffs() const {
    const auto &self = static_cast<const Derived &>(*this);
    auto extField = cast<ExtensionFieldTypeInterface>(self.getType());
    auto baseField = cast<PrimeFieldType>(extField.getBaseFieldType());
    APInt modulus = baseField.getModulus().getValue();
    APInt xi = cast<IntegerAttr>(extField.getNonResidue()).getValue();
    auto convertedType = convertPrimeFieldType(baseField);

    std::array<std::array<PrimeFieldCodeGen, N - 1>, N - 1> coeffs;

    // Use larger bit width to avoid overflow
    unsigned extBitWidth = modulus.getBitWidth() * 2;
    APInt pMinus1 = (modulus - 1).zext(extBitWidth);
    APInt nVal(extBitWidth, N);

    for (size_t e = 1; e < N; ++e) {
      for (size_t i = 1; i < N; ++i) {
        // exp = i * e * (p - 1) / n
        APInt exp =
            APInt(extBitWidth, i) * APInt(extBitWidth, e) * pMinus1.udiv(nVal);
        // Compute ξ^exp mod p
        APInt coeff =
            mod_arith::ModArithOperation::fromUnchecked(xi, convertedType)
                .power(exp);

        Value constant =
            self.getBuilder().template create<mod_arith::ConstantOp>(
                convertedType,
                IntegerAttr::get(baseField.getStorageType(), coeff));
        coeffs[e - 1][i - 1] = PrimeFieldCodeGen(&self.getBuilder(), constant);
      }
    }
    return coeffs;
  }
};

} // namespace mlir::prime_ir::field

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDTOMODARITH_FROBENIUSCOEFFS_H_
